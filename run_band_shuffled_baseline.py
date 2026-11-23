"""
The non-personalized LLM baseline with shuffled candidates for a popularity band's LLM table.

The band tables (run_llm_experiments.py --parts band) call the baseline once per category with the
candidates in list order, which the model partly copies. Here the baseline is called TRIALS times per
category, each with a different random order, and each user's AP is averaged over the trials. The
baseline does not depend on the user, so the same calls score every user of the category.

    python run_band_shuffled_baseline.py --band-start 181 --band-end 200 --band-source extended

Per-user scores go to results/llm/band_shuffled_baseline<suffix>.csv.gz.
"""

import argparse
import json
import random

import numpy as np
import pandas as pd

from reproduction.config import RESULTS_DIR, SEED
from reproduction.data import load_categories, load_users
from reproduction.llm import (ResponseCache, SpendingCap, candidates_json, llm_average_precision, load_openai_key,
                              make_dspy_rankers)
from run_llm_experiments import band_sample, band_suffix

OUT = RESULTS_DIR / "llm"
TRIALS = 10


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band-start", type=int, default=181)
    p.add_argument("--band-end", type=int, default=200)
    p.add_argument("--band-source", default="extended", choices=["original", "extended"])
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="low")
    p.add_argument("--budget", type=float, default=0.2, help="stop calling the model at this cost (USD)")
    args = p.parse_args()
    band = (args.band_start, args.band_end, args.band_source)

    categories = load_categories()
    users = load_users(categories)
    sample, candidates, _, _ = band_sample(users, categories, band)

    import dspy
    lm = dspy.LM(f"openai/{args.model}", api_key=load_openai_key(), reasoning_effort=args.effort, cache=False, num_retries=8)
    cap = SpendingCap(lm, args.budget)
    rank_default = cap.guard(make_dspy_rankers(lm=lm)[1])
    cache = ResponseCache(OUT / "cache" / f"band_shuffled_baseline{band_suffix(band)}.jsonl")

    responses = {}
    try:
        for topic, cands in candidates.items():
            for trial in range(TRIALS):
                order = [int(i) for i in cands["twitter_user_id"]]
                random.Random(f"{SEED}:baseline:{band}:{topic}:{trial}").shuffle(order)
                text = candidates_json(cands, order=order)
                responses[(topic, trial)] = cache.call(f"{topic}|{trial}", lambda: rank_default(text))
    finally:
        cap.update()
        if cap.calls:
            info_path = OUT / "run_info.json"
            runs = json.loads(info_path.read_text())
            runs.append({"model": args.model, "reasoning_effort": args.effort, "parts": ["band-shuffled-baseline"], "band": list(band),
                         "calls": cap.calls, "cost_usd": round(cap.spent, 4), "budget_usd": args.budget})
            info_path.write_text(json.dumps(runs, indent=2))
        print(f"{cap.calls} model calls, ${cap.spent:.4f}")

    rows = []
    for row, user in sample.iterrows():
        topic = user["topic"]
        relevant = [f for f in user["friends_ids"] if f in set(candidates[topic]["twitter_user_id"])]
        aps = [llm_average_precision(responses[(topic, t)], relevant) for t in range(TRIALS)]
        rows.append({"user": row, "topic": topic, "llm_baseline_shuffled": float(np.mean(aps))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"band_shuffled_baseline{band_suffix(band)}.csv.gz", index=False)
    print(res.groupby("topic")["llm_baseline_shuffled"].mean().round(3).to_string())
    print("overall (mean over categories):", round(res.groupby("topic")["llm_baseline_shuffled"].mean().mean(), 3))


if __name__ == "__main__":
    main()
