"""
Demographics-only LLM ranking for the users of the ranks 81-100 LLM sample
(30 per category, see run_llm_experiments.py).

For each user, the LLM first writes a short description of the user's likely
gender, age, ethnicity and education, deduced from the names of the accounts
listed in the user's k=50 prompt (with all of the target category's entities
removed). It then ranks the candidates given only that description.

    python run_llm_demographics.py --debug        # one user, every step printed
    python run_llm_demographics.py                # all sampled users

Usernames and the generated descriptions describe real people, so they are written
to data/llm_demographics/ (not tracked by git). Scores, keyed by internal user row,
go to results/llm/.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from reproduction.config import CATEGORY_LABELS, DATA_DIR, RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import load_metadata
from reproduction.llm import (
    ResponseCache,
    SpendingCap,
    candidates_json,
    llm_average_precision,
    load_openai_key,
    names_of,
    ranked_ids,
    user_samples,
)
from reproduction.llm_demographics import PROMPT_VERSION, make_demographic_programs
from reproduction.report import markdown_table
from run_llm_experiments import by_category, band_sample, p_value

PRIVATE_DIR = DATA_DIR / "llm_demographics"
OUT = RESULTS_DIR / "llm"
FULL_USERS_PICKLE = DATA_DIR / "users_selected_for_the_experiment2.pkl"


def write_sampled_users(sample):
    """The sampled users' Twitter ids and screen names (from the non-anonymized user file)."""
    full = pd.read_pickle(FULL_USERS_PICKLE).reset_index(drop=True)
    users = pd.DataFrame({
        "row": sample.index,
        "category": sample["topic"].map(CATEGORY_LABELS).values,
        "twitter_user_id": full.loc[sample.index, "user_id"].values,
        "screen_name": full.loc[sample.index, "screen_name"].values,
    })
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    users.to_csv(PRIVATE_DIR / "sampled_users.csv", index=False)
    return users.set_index("row")


def show_messages(messages):
    return "\n".join(f"--- [{m['role']}] ---\n{m['content']}" for m in messages)


def debug_user(row, sample, candidates, followed, users_file, describe, rank, render, lm, cache, names):
    """Run one user step by step and return a markdown walkthrough."""
    user = sample.loc[row]
    topic = user["topic"]
    cands = candidates[topic]
    cand_ids = list(cands["twitter_user_id"])
    relevant = [f for f in user["friends_ids"] if f in set(cand_ids)]
    ids = followed["ids"][row]
    text = followed["text"][row]
    out = []

    out.append(f"## Step 1: the user\n\nInternal row {row}, sampled for **{CATEGORY_LABELS[topic]}**. "
               f"Twitter id `{users_file.at[row, 'twitter_user_id']}`, screen name `@{users_file.at[row, 'screen_name']}`. "
               f"Follows {len(user['friends_ids']):,} accounts. Follows {len(relevant)} of the {len(cand_ids)} candidates "
               f"(the relevant answers): {', '.join(names.get(str(i), str(i)) for i in relevant)}.")

    out.append(f"## Step 2: the accounts given to the LLM\n\nThe same accounts as in the user's LLM k = 50 prompt: 50 accounts "
               f"sampled at random (seed 76) from everything the user follows, after removing every listed entity of the "
               f"target category, keeping only the {len(ids)} that have a known name:\n\n> {text}")

    def call(key, kind, fn, inputs):
        before = len(lm.history)
        cached = cache.get(key) is not None
        response = cache.call(key, fn)
        entry = lm.history[-1] if len(lm.history) > before else None
        messages = entry["messages"] if entry else render(kind, **inputs)
        raw = entry["outputs"][0] if entry else "(cached response)"
        usage = (f"{entry['usage'].get('prompt_tokens')} prompt + {entry['usage'].get('completion_tokens')} completion tokens, "
                 f"${entry.get('cost') or 0:.5f}") if entry else "served from cache"
        return response, messages, raw, usage, cached

    desc, msgs, raw, usage, _ = call(f"{PROMPT_VERSION}|describe|{row}", "describe", lambda: describe(text),
                                     {"followed_accounts": text})
    out.append("## Step 3: LLM call 1, the demographic description\n\nPrompt (the chat messages dspy sends to GPT-6 Luna):\n\n"
               f"```text\n{show_messages(msgs)}\n```\n\nRaw model output ({usage}):\n\n```text\n{raw}\n```\n\n"
               f"Parsed description:\n\n> {desc}")

    cjson = candidates_json(cands, by_popularity=True)
    ranked, msgs, raw, usage, _ = call(f"{PROMPT_VERSION}|rank|{row}", "rank", lambda: rank(desc, cjson),
                                       {"user_description": desc, "potential_accounts": cjson})
    out.append("## Step 4: LLM call 2, ranking from the description only\n\nThe candidates are listed by follower count, as "
               "in the original personalized prompts. Prompt:\n\n"
               f"```text\n{show_messages(msgs)}\n```\n\nRaw model output ({usage}):\n\n```text\n{raw}\n```")

    order = ranked_ids(ranked)
    pos = {a: i + 1 for i, a in enumerate(order)}
    rows = "\n".join(f"| {pos.get(a, '–')} | {names.get(str(a), a)} | {'✓' if a in relevant else ''} |" for a in order)
    found = sorted(pos[a] for a in relevant if a in pos)
    terms = " + ".join(f"{i + 1}/{p}" for i, p in enumerate(found))
    ap = llm_average_precision(ranked, relevant)
    out.append(f"## Step 5: scoring\n\nThe model's ranking ({len(order)} of {len(cand_ids)} candidates returned); "
               f"✓ marks candidates the user follows:\n\n| Rank | Candidate | Followed |\n|---|---|---|\n{rows}\n\n"
               f"Average precision = mean of (i / rank of the i-th followed candidate) = ({terms}) / {len(found)} = **{ap:.3f}**.")

    band = pd.read_csv(OUT / "band_per_user.csv.gz").set_index("user")
    if row in band.index:
        b = band.loc[row]
        out.append("## Step 6: this user under the other methods\n\n| Method | AP |\n|---|---|\n"
                   + "\n".join(f"| {n} | {b[c]:.3f} |" for n, c in [
                       ("Random ranking (expected)", "random"), ("Popularity", "popularity"),
                       ("SocialVec, all entities", "socialvec"), ("LLM, no user information", "llm_baseline"),
                       ("LLM, 50 followed accounts", "llm_k50")])
                   + f"\n| **LLM, demographic description only** | **{ap:.3f}** |")
    return "\n\n".join(out)


def run_all(sample, candidates, followed, users_file, describe, rank, cache, workers):
    cjson = {t: candidates_json(c, by_popularity=True) for t, c in candidates.items()}

    def job(row):
        text = followed["text"][row]
        desc = cache.call(f"{PROMPT_VERSION}|describe|{row}", lambda: describe(text))
        topic = sample.at[row, "topic"]
        ranked = cache.call(f"{PROMPT_VERSION}|rank|{row}", lambda: rank(desc, cjson[topic]))
        return row, desc, ranked

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(job, sample.index), total=len(sample), desc="Demographic LLM"))

    rows, private = [], []
    for row, desc, ranked in results:
        user = sample.loc[row]
        cand_ids = set(candidates[user["topic"]]["twitter_user_id"])
        relevant = [f for f in user["friends_ids"] if f in cand_ids]
        rows.append({"user": row, "topic": user["topic"], "n_relevant": len(relevant),
                     "n_accounts_described": len(followed["ids"][row]),
                     "llm_demographics": llm_average_precision(ranked, relevant)})
        private.append({"row": row, "category": CATEGORY_LABELS[user["topic"]],
                        "twitter_user_id": users_file.at[row, "twitter_user_id"],
                        "screen_name": users_file.at[row, "screen_name"],
                        "description": desc, "ranking_response": ranked})
    pd.DataFrame(private).to_csv(PRIVATE_DIR / "descriptions.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "demographics_per_user.csv.gz", index=False)
    return res


def summarize(res):
    band = pd.read_csv(OUT / "band_per_user.csv.gz")
    m = band.merge(res[["user", "llm_demographics", "n_accounts_described"]], on="user")
    cols = ["random", "popularity", "socialvec", "llm_baseline", "llm_demographics", "llm_k12", "llm_k25", "llm_k50"]
    table = by_category(m, cols)
    table.to_csv(OUT / "demographics_by_category.csv")
    tests = {f"llm_demographics vs {c}": p_value(m["llm_demographics"], m[c])
             for c in ["popularity", "llm_baseline", "llm_k12", "llm_k50", "socialvec"]}
    pd.Series(tests).to_csv(OUT / "demographics_tests.csv")
    shown = table.copy()
    shown.index = [CATEGORY_LABELS.get(i, "**Overall**") for i in shown.index]
    text = (markdown_table(shown) + "\n\nPaired Wilcoxon p: "
            + ", ".join(f"{k} {v:.3g}" for k, v in tests.items()))
    (OUT / "demographics_tables.md").write_text("# Demographics-only LLM ranking (ranks 81-100)\n\n" + text + "\n",
                                                encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debug", action="store_true", help="run one user and print every step")
    p.add_argument("--user", type=int, default=None, help="internal row of the debug user (default: first musical-artist user)")
    p.add_argument("--user-index", type=int, default=0,
                   help="with no --user, take the n-th sampled musical-artist user (0 = first)")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="low")
    p.add_argument("--budget", type=float, default=0.5, help="stop calling the model at this cost (USD)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    categories = load_categories()
    users = load_users(categories)
    sample, candidates, exclude, _ = band_sample(users, categories)
    users_file = write_sampled_users(sample)
    metadata = load_metadata()
    names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))
    ids = user_samples(sample, categories, names, exclude=exclude)["llm_k50"]
    followed = {"ids": ids, "text": ids.apply(lambda a: names_of(a, names))}

    import dspy
    lm = dspy.LM(f"openai/{args.model}", api_key=load_openai_key(), reasoning_effort=args.effort,
                 cache=False, num_retries=8)
    cap = SpendingCap(lm, args.budget)
    describe, rank, render = make_demographic_programs(lm)
    describe, rank = cap.guard(describe), cap.guard(rank)
    cache = ResponseCache(OUT / "cache" / "demographics.jsonl")

    try:
        if args.debug:
            row = args.user if args.user is not None else sample[sample["topic"] == "MusicalArtist"].index[args.user_index]
            report = debug_user(row, sample, candidates, followed, users_file, describe, rank, render, lm, cache, names)
            path = PRIVATE_DIR / f"debug_user_{row}_{PROMPT_VERSION}.md"
            path.write_text(report, encoding="utf-8")
            print(report)
            print(f"\n(saved to {path})")
        else:
            summarize(run_all(sample, candidates, followed, users_file, describe, rank, cache, args.workers))
    finally:
        cap.update()
        info_path = OUT / "run_info.json"
        runs = json.loads(info_path.read_text()) if info_path.exists() else []
        if cap.calls:
            runs.append({"model": args.model, "reasoning_effort": args.effort,
                         "parts": ["demographics-debug" if args.debug else "demographics"],
                         "calls": cap.calls, "cost_usd": round(cap.spent, 4), "budget_usd": args.budget})
            info_path.write_text(json.dumps(runs, indent=2))
        print(f"{cap.calls} model calls, ${cap.spent:.4f}")


if __name__ == "__main__":
    main()
