"""
The Section 5 rankings (results_v2.html) repeated with a Claude model as the ranker.

Same users, candidate orders and inputs as run_llm_demographics_v2.py: for every user and band
(ranks 1-20 and 81-100) the candidates in the user's random order, ranked from no user information,
from GPT-4o's demographic description, or from the names of 12, 25 or 50 followed accounts. The
prompts are the exact chat messages dspy renders for the authors' signatures (system + user), sent
through the Anthropic SDK; the answer's ranked_ids field is scored like the GPT-6 Luna runs.

    python run_llm_claude.py --pilot 50       # a few synchronous calls, prints the token use and cost
    python run_llm_claude.py --batch          # everything not yet cached, through the Message Batches API

The API key is read from ANTHROPIC_API_KEY or .env.local, and requests go to api.anthropic.com.
Per-user scores go to results/llm/demographics_v2_<model>_per_user.csv.gz.
"""

import argparse
import json
import random
import re
import time

import pandas as pd

from reproduction.config import RESULTS_DIR, ROOT
from reproduction.data import load_categories, load_users
from reproduction.embeddings import load_metadata
from reproduction.llm import (ResponseCache, candidates_json, llm_average_precision, names_of, ranked_ids, ranker_messages,
                              user_samples)
from reproduction.llm_demographics import make_demographic_programs
from run_llm_demographics_bands import shuffled_order
from run_llm_demographics_v2 import BANDS, PRIVATE, select_users
from run_llm_experiments import band_setup

OUT = RESULTS_DIR / "llm"
MODEL = "claude-haiku-5-5"
PRICE = {"claude-haiku-5-5": (0.10, 0.50)}  # USD per million input / output tokens (prompts under 100K tokens)
CONDITIONS = ["baseline", "gpt-4o", "k12", "k25", "k50"]
MAX_TOKENS = 8000  # thinking counts toward max_tokens


def load_anthropic_key():
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ANTHROPIC_API_KEY":
            return value.strip().strip('"').strip("'")
    raise RuntimeError("No ANTHROPIC_API_KEY in the environment or .env.local")


def answer_text(message):
    """The ranked_ids field of a response (the text after dspy's field marker), or the whole text."""
    text = "".join(b.text for b in message.content if b.type == "text")
    m = re.search(r"\[\[\s*##\s*ranked_ids\s*##\s*\]\](.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)", text, re.S)
    return (m.group(1) if m else text).strip()


def build_jobs(users, categories, names, descriptions):
    """(key, band, row, topic, relevant, candidate ids, messages) for every user, band and condition."""
    _, _, render_demo = make_demographic_programs(None, single_value=True)
    sample = select_users(users, categories)
    jobs = []
    for label, (spec, _) in BANDS.items():
        candidates, exclude, _ = band_setup(users, categories, spec)
        prompts = user_samples(sample, categories, names, exclude=exclude)
        for row, user in sample.iterrows():
            cands = candidates[user["topic"]]
            relevant = [f for f in user["friends_ids"] if f in set(cands["twitter_user_id"])]
            if not relevant or row not in descriptions.index:  # descriptions come from run_llm_demographics_v2.py
                continue
            order = shuffled_order(cands, label, row)
            text = candidates_json(cands, order=order)
            for cond in CONDITIONS:
                if cond == "baseline":
                    msgs = ranker_messages("default", potential_accounts=text)
                elif cond == "gpt-4o":
                    msgs = render_demo("rank", user_description=descriptions[row], potential_accounts=text)
                else:
                    msgs = ranker_messages("personal", followed_accounts=names_of(prompts.at[row, f"llm_{cond}"], names),
                                           potential_accounts=text)
                key = f"b{label.replace('-', '_')}-{cond.replace('-', '')}-{row}"
                jobs.append((key, label, row, user["topic"], relevant, order, msgs))
    return jobs


def request_params(msgs):
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "output_config": {"effort": "low"},
            "system": msgs[0]["content"], "messages": [{"role": m["role"], "content": m["content"]} for m in msgs[1:]]}


def cost(usage_in, usage_out, batch):
    p_in, p_out = PRICE[MODEL]
    return (usage_in * p_in + usage_out * p_out) / 1e6 * (0.5 if batch else 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", type=int, default=0, help="run this many random uncached jobs synchronously")
    p.add_argument("--batch", action="store_true", help="submit every uncached job as one message batch and wait for it")
    p.add_argument("--max-cost", type=float, default=15.0, help="refuse to submit a batch estimated above this (USD)")
    args = p.parse_args()

    import anthropic
    client = anthropic.Anthropic(api_key=load_anthropic_key(), base_url="https://api.anthropic.com")
    categories = load_categories()
    users = load_users(categories)
    metadata = load_metadata()
    names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))
    descriptions = pd.read_csv(PRIVATE / "descriptions_gpt-4o_v2.csv").set_index("row")["description"]
    jobs = build_jobs(users, categories, names, descriptions)
    cache = ResponseCache(OUT / "cache" / f"{MODEL}.jsonl")
    usage_path = OUT / "cache" / f"{MODEL}_usage.jsonl"
    todo = [j for j in jobs if cache.get(j[0]) is None]
    print(f"{len(jobs)} jobs, {len(todo)} not cached")

    spent_in = spent_out = n_calls = 0
    if args.pilot:
        for key, *_, msgs in random.Random(0).sample(todo, min(args.pilot, len(todo))):
            msg = client.messages.create(**request_params(msgs))
            cache.call(key, lambda: answer_text(msg))
            spent_in += msg.usage.input_tokens
            spent_out += msg.usage.output_tokens
            n_calls += 1
            with open(usage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "in": msg.usage.input_tokens, "out": msg.usage.output_tokens,
                                    "stop": msg.stop_reason, "batch": False}) + "\n")
        n = min(args.pilot, len(todo))
        per_call = cost(spent_in, spent_out, batch=True) / n
        print(f"pilot: {n} calls, {spent_in / n:.0f} input and {spent_out / n:.0f} output tokens per call, "
              f"${cost(spent_in, spent_out, False):.4f}; the remaining {len(todo) - n} by batch: about ${per_call * (len(todo) - n):.2f}")
        mode = "pilot"
    elif args.batch and todo:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        usage = [json.loads(l) for l in usage_path.read_text(encoding="utf-8").splitlines()] if usage_path.exists() else []
        if usage:
            estimate = cost(sum(u["in"] for u in usage), sum(u["out"] for u in usage), True) / len(usage) * len(todo)
            print(f"estimated batch cost: ${estimate:.2f}")
            if estimate > args.max_cost:
                raise SystemExit(f"estimate above --max-cost ${args.max_cost}; not submitted")
        batch = client.messages.batches.create(requests=[
            Request(custom_id=key, params=MessageCreateParamsNonStreaming(**request_params(msgs))) for key, *_, msgs in todo])
        (OUT / "cache" / f"{MODEL}_batch_id.txt").write_text(batch.id)
        print("batch", batch.id, "submitted")
        while (batch := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            c = batch.request_counts
            print(f"  processing {c.processing}, succeeded {c.succeeded}, errored {c.errored}", flush=True)
            time.sleep(60)
        failed = 0
        with open(usage_path, "a", encoding="utf-8") as f:
            for result in client.messages.batches.results(batch.id):
                if result.result.type != "succeeded":
                    failed += 1
                    continue
                msg = result.result.message
                cache.call(result.custom_id, lambda m=msg: answer_text(m))
                spent_in += msg.usage.input_tokens
                spent_out += msg.usage.output_tokens
                n_calls += 1
                f.write(json.dumps({"key": result.custom_id, "in": msg.usage.input_tokens, "out": msg.usage.output_tokens,
                                    "stop": msg.stop_reason, "batch": True}) + "\n")
        print(f"batch done: {failed} failed (rerun --batch to retry them), ${cost(spent_in, spent_out, True):.4f}")
        mode = "batch"
    else:
        mode = None
    if mode and (spent_in or spent_out):
        info_path = OUT / "run_info.json"
        runs = json.loads(info_path.read_text())
        runs.append({"model": MODEL, "reasoning_effort": "low", "parts": [f"demographics-v2-claude-{mode}"], "calls": n_calls,
                     "input_tokens": spent_in, "output_tokens": spent_out,
                     "cost_usd": round(cost(spent_in, spent_out, mode == "batch"), 4)})
        info_path.write_text(json.dumps(runs, indent=2))

    # per-user scores from whatever is cached
    rows = {}
    for key, label, row, topic, relevant, order, _ in jobs:
        response = cache.get(key)
        if response is None:
            continue
        r = rows.setdefault((label, row), {"user": row, "topic": topic, "band": label, "n_relevant": len(relevant)})
        cond = key.split("-")[1]
        cond = "gpt-4o" if cond == "gpt4o" else cond
        r[f"ap_{cond}"] = llm_average_precision(response, relevant)
        r[f"returned_{cond}"] = len(set(ranked_ids(response)) & set(order))
    res = pd.DataFrame(rows.values())
    if len(res):
        res.to_csv(OUT / f"demographics_v2_{MODEL}_per_user.csv.gz", index=False)
        print(res.groupby("band")[[c for c in res.columns if c.startswith("ap_")]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
