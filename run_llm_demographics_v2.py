"""
Demographics-only LLM ranking, second version (results_v2.html, Section 5).

Users: the users of the ranks 81-100 LLM table (Table 9; run_band_more_users.py, 200 per category),
so that every user count matches Tables 9-11. The summary also reports the subset of users who
follow fewer than MAX_FOLLOWS accounts in all (fewer likely bots) and at least MIN_ENTITIES of the
experiment's category entities (enough evidence for an informed guess).

GPT-4o describes each user's gender, age, ethnicity and education from the names of the
accounts in the user's k = 50 prompt, with the instruction to state exactly one value per
attribute (llm_demographics.SINGLE_VALUE). GPT-6 Luna then ranks the candidates of ranks 1-20
and 81-100 in one random order per user and band, from no user information, the description,
or 12/25/50 followed account names (the protocol of run_llm_demographics_bands.py; the no-information
and account-name calls are those of run_shuffled_personal.py, reused from the cache).

    python run_llm_demographics_v2.py --budget 3.5

Per-user scores and a summary go to results/llm/demographics_v2_*; the descriptions stay in
data/llm_demographics/.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from reproduction.config import CATEGORY_LABELS, DATA_DIR, PAPER_CATEGORY_ORDER, RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import load_metadata
from reproduction.popularity_bands import load_category_sheets, load_extended_sheets
from reproduction.llm import (ResponseCache, SpendingCap, candidates_json, llm_average_precision, load_openai_key,
                              make_dspy_rankers, names_of, ranked_ids, user_samples)
from reproduction.llm_demographics import (ATTRIBUTE_VALUES, PROMPT_VERSION, TOPICS, make_demographic_programs,
                                           stated_values)
from run_llm_demographics_bands import shuffled_order
from run_llm_experiments import DEFAULT_BAND, band_setup, p_value

OUT = RESULTS_DIR / "llm"
PRIVATE = DATA_DIR / "llm_demographics"
MAX_FOLLOWS = 2000
MIN_ENTITIES = 30
DESCRIBE_VERSION = "v4"  # the single-value description prompt
BANDS = {"1-20": ((1, 20, "original"), "popularity_band_1_20"),
         "81-100": ((81, 100, "original"), "popularity_band_81_100")}
N_EXAMPLES = 4


def select_users(users, categories):
    """The users of Table 9 (ranks 81-100, 200 per category)."""
    return users.loc[pd.read_csv(OUT / "band_per_user_n200.csv.gz")["user"]]


def passes_filter(users):
    """Fewer than MAX_FOLLOWS accounts followed and at least MIN_ENTITIES category entities."""
    return (follows(users) < MAX_FOLLOWS) & (users["experiment_accounts"].apply(len) >= MIN_ENTITIES)


def follows(users):
    """Accounts followed: the larger of the profile's friends_count and the collected list (they rarely disagree)."""
    return np.maximum(pd.to_numeric(users["friends_count"]).fillna(0), users["friends_ids"].apply(len))


def stats(values):
    v = pd.Series(values)
    return {"min": int(v.min()), "mean": round(float(v.mean()), 1), "median": float(v.median()), "max": int(v.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="low")
    p.add_argument("--describe-model", default="gpt-4o")
    p.add_argument("--budget", type=float, default=3.5, help="stop calling the models at this cost (USD)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    categories = load_categories()
    users = load_users(categories)
    sample = select_users(users, categories)
    metadata = load_metadata()
    names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))
    _, exclude, _ = band_setup(users, categories, DEFAULT_BAND)
    followed_ids = user_samples(sample, categories, names, exclude=exclude)["llm_k50"]
    followed = followed_ids.apply(lambda a: names_of(a, names))

    import dspy
    key = load_openai_key()
    lm = dspy.LM(f"openai/{args.model}", api_key=key, reasoning_effort=args.effort, cache=False, num_retries=8)
    describe_lm = dspy.LM(f"openai/{args.describe_model}", api_key=key, cache=False, num_retries=8)
    cap = SpendingCap([lm, describe_lm], args.budget)
    describe, rank, _ = make_demographic_programs(lm, describe_lm, single_value=True)
    rank_personal, rank_default = make_dspy_rankers(lm=lm)
    describe, rank, rank_personal, rank_default = (cap.guard(f) for f in (describe, rank, rank_personal, rank_default))
    describe_cache = ResponseCache(OUT / "cache" / "demographics_v2.jsonl")
    rank_cache = ResponseCache(OUT / "cache" / "demographics_bands.jsonl")

    jobs = []  # (band, row, topic, candidates, order, relevant, followed-account prompts)
    for label, (spec, _) in BANDS.items():
        candidates, band_exclude, _ = band_setup(users, categories, spec)
        prompts = user_samples(sample, categories, names, exclude=band_exclude)
        for row, user in sample.iterrows():
            cands = candidates[user["topic"]]
            relevant = [f for f in user["friends_ids"] if f in set(cands["twitter_user_id"])]
            if relevant:
                jobs.append((label, row, user["topic"], cands, shuffled_order(cands, label, row), relevant, prompts.loc[row]))

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            descriptions = dict(tqdm(pool.map(
                lambda row: (row, describe_cache.call(f"{DESCRIBE_VERSION}|describe|{row}", lambda: describe(followed[row]))),
                sample.index), total=len(sample), desc="GPT-4o descriptions"))

        def run(job):
            label, row, topic, cands, order, relevant, prompt = job
            text = candidates_json(cands, order=order)
            tag = f"{PROMPT_VERSION}|{label}"
            out = {"baseline": rank_cache.call(f"{tag}|baseline|{row}", lambda: rank_default(text)),
                   "gpt-4o": rank_cache.call(f"{DESCRIBE_VERSION}|{label}|gpt-4o|{row}", lambda: rank(descriptions[row], text))}
            for k in (12, 25, 50):
                names_k = names_of(prompt[f"llm_k{k}"], names)
                out[f"k{k}"] = rank_cache.call(f"{tag}|k{k}|{row}", lambda f=names_k: rank_personal(f, text))
            return job, out

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(run, jobs), total=len(jobs), desc="Shuffled rankings"))
    finally:
        cap.update()
        info_path = OUT / "run_info.json"
        runs = json.loads(info_path.read_text())
        if cap.calls:
            runs.append({"model": args.model, "reasoning_effort": args.effort, "describe_model": args.describe_model,
                         "parts": ["demographics-v2"], "calls": cap.calls, "cost_usd": round(cap.spent, 4),
                         "budget_usd": args.budget, "cost_by_model": {k: round(v, 4) for k, v in cap.spent_by_model.items()},
                         "calls_by_model": cap.calls_by_model})
            info_path.write_text(json.dumps(runs, indent=2))
        print(f"{cap.calls} model calls, ${cap.spent:.4f}", {k: round(v, 4) for k, v in cap.spent_by_model.items()})

    # descriptions (private) and per-user scores
    PRIVATE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": sample.index, "topic": sample["topic"].values, "followed_accounts": followed.values,
                  "description": [descriptions[r] for r in sample.index]}).to_csv(PRIVATE / "descriptions_gpt-4o_v2.csv", index=False)
    rows = []
    for (label, row, topic, cands, order, relevant, _), out in results:
        r = {"user": row, "topic": topic, "band": label, "n_relevant": len(relevant), "n_candidates": len(order)}
        for kind, response in out.items():
            r[f"ap_{kind}"] = llm_average_precision(response, relevant)
            r[f"returned_{kind}"] = len(set(ranked_ids(response)) & set(order))
        rows.append(r)
    res = pd.DataFrame(rows)
    merged = []
    for label, (_, folder) in BANDS.items():
        other = pd.read_csv(RESULTS_DIR / folder / "per_user_ap.csv.gz")
        other = other[(other["population"] == "category") & (other["candidates"] == "band")
                      & (other["exclusion"] == "exclude_category")][["user", "random", "popularity", "socialvec"]]
        merged.append(res[res["band"] == label].merge(other, on="user", how="left"))
    res = pd.concat(merged, ignore_index=True)
    res.to_csv(OUT / "demographics_v2_per_user.csv.gz", index=False)

    # summary for the results page: MAP, tests, user statistics, stated attributes, examples
    desc = pd.Series(descriptions).reindex(sample.index)
    methods = ["random", "popularity", "ap_baseline", "ap_gpt-4o", "ap_k12", "ap_k25", "ap_k50", "socialvec"]
    filtered_rows = set(sample.index[passes_filter(sample)])

    def band_summary(b):
        return {"users": len(b), "users_by_category": b.groupby("topic").size().to_dict(),
                "map": b.groupby("topic")[methods].mean().mean().round(4).to_dict(),
                "tests": {name: p_value(b[x], b[y]) for name, x, y in [
                    ("4o_vs_baseline", "ap_gpt-4o", "ap_baseline"), ("baseline_vs_popularity", "ap_baseline", "popularity"),
                    ("4o_vs_socialvec", "ap_gpt-4o", "socialvec"), ("k12_vs_4o", "ap_k12", "ap_gpt-4o"),
                    ("k50_vs_socialvec", "ap_k50", "socialvec")]}}

    shuffled = {label: band_summary(res[res["band"] == label]) for label in BANDS}
    shuffled_filtered = {label: band_summary(res[(res["band"] == label) & res["user"].isin(filtered_rows)]) for label in BANDS}
    attributes = pd.DataFrame({a: desc.map(lambda t, a=a: (stated_values(t, a) or ["Not stated"])[0]) for a in ATTRIBUTE_VALUES})
    # examples: per category, the user whose names given to GPT-4o include the most known entities (accounts of the
    # original or extended category lists); the page shows those entities, the description is the user's own
    known = {}
    for sheets in (load_category_sheets(), load_extended_sheets()):
        for topic, sheet in sheets.items():
            for i, name, handle in zip(sheet["twitter_user_id"], sheet["twitter_name"], sheet["twitter_screen_name"]):
                known.setdefault(int(i), (name if isinstance(name, str) else handle, handle, CATEGORY_LABELS.get(topic, topic)))
    examples = []
    for topic in ["MusicalArtist", "SportsTeam", "Politician", "news"][:N_EXAMPLES]:
        rows = sample.index[sample["topic"] == topic]
        n_known = {r: sum(int(i) in known for i in followed_ids[r]) for r in rows}
        row = max(rows, key=lambda r: (n_known[r], -list(rows).index(r)))
        examples.append({"user": int(row), "category": CATEGORY_LABELS[topic], "n_names": len(followed_ids[row]),
                         "known_entities": [list(known[int(i)]) for i in followed_ids[row] if int(i) in known],
                         "description": desc[row]})
    # a complete Stage 2 call for the first example user at ranks 81-100: the inputs, the answer and its score
    row = examples[0]["user"]
    (label, _, topic, cands, order, relevant, _), out = next(r for r in results if r[0][0] == "81-100" and r[0][1] == row)
    name_of = dict(zip(cands["twitter_user_id"].astype(int), cands["twitter_name"]))
    ranked = [i for i in ranked_ids(out["gpt-4o"]) if i in name_of]
    stage2 = {"user": int(row), "band": label, "description": desc[row], "candidates_json": candidates_json(cands, order=order),
              "response": str(out["gpt-4o"]), "ranked": [[int(i), name_of[i], int(i) in set(relevant)] for i in ranked],
              "relevant": [name_of[int(i)] for i in relevant], "ap": llm_average_precision(out["gpt-4o"], relevant)}
    summary = {
        "filter": {"max_follows": MAX_FOLLOWS, "min_entities": MIN_ENTITIES, "users": len(filtered_rows)},
        "users": len(sample), "users_by_category": sample.groupby("topic").size().reindex(
            [c for c in PAPER_CATEGORY_ORDER if c in set(sample["topic"])]).to_dict(),
        "user_stats": {"follows": stats(follows(sample)),
                       "category_entities": stats(sample["experiment_accounts"].apply(len)),
                       "names_given": stats(followed_ids.apply(len))},
        "leak_4o": int(desc.str.contains(TOPICS, case=False, regex=True).sum()),
        "shuffled": shuffled,
        "shuffled_filtered": shuffled_filtered,
        "attributes": {a: {"values": [v for v, _ in ATTRIBUTE_VALUES[a]] + ["Not stated"],
                           "counts": attributes[a].value_counts().to_dict(),
                           "multiple": int(desc.map(lambda t, a=a: len(stated_values(t, a)) > 1).sum()),
                           "by_category": {t: g.value_counts().to_dict() for t, g in attributes[a].groupby(sample["topic"])}}
                       for a in ATTRIBUTE_VALUES},
        "examples": examples,
        "stage2_example": stage2,
    }
    (OUT / "demographics_v2_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for label, s in shuffled.items():
        print(label, s["users"], s["map"])
    print({a: v["multiple"] for a, v in summary["attributes"].items()}, "leak:", summary["leak_4o"])


if __name__ == "__main__":
    main()
