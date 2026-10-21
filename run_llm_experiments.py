"""
LLM experiments on stratified user samples:

    main   - Table 3 rerun: baseline, k=12, k=25, k=50 for 50 random users per category
    order  - does the candidate order matter? k=25 for the same users, repeated in the
             original order and with shuffled candidates; plus 10 original-order and
             10 shuffled baseline rankings per category
    band   - the Table 3 conditions for the candidates ranked 81-100 by popularity
             (30 random users per category from the band experiment)

    python run_llm_experiments.py                        # all parts, then the analysis
    python run_llm_experiments.py --parts main order     # selected parts
    python run_llm_experiments.py --analyze-only         # recompute tables from saved results
    python run_llm_experiments.py --dry-run              # offline check with a fake model

The OpenAI key is read from OPENAI_API_KEY / OPENAI_KEY or .env.local. Model calls
stop once the reported cost reaches --budget. Outputs go to results/llm/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from reproduction import paper_results as paper
from reproduction.config import CATEGORY_LABELS, PAPER_CATEGORY_ORDER, RESULTS_DIR, SEED
from reproduction.data import load_categories, load_users
from reproduction.embeddings import SocialVec, load_metadata
from reproduction.llm import (
    BudgetExceeded,
    SpendingCap,
    baseline_order_trials,
    coverage,
    llm_average_precision,
    load_openai_key,
    make_dspy_rankers,
    order_agreement,
    ranked_ids,
    run_llm,
    user_prompts,
)
from reproduction.popularity_bands import load_category_sheets, popularity_band
from reproduction.report import markdown_table

USERS_PER_CATEGORY = 50
BAND_USERS_PER_CATEGORY = 30
BASELINE_TRIALS = 10
BAND_START, BAND_END = 81, 100
LLM_COLUMNS = ["llm_baseline", "llm_k12", "llm_k25", "llm_k50"]


# ---------------------------------------------------------------- runs
def main_sample(users):
    return users.groupby("topic").sample(n=USERS_PER_CATEGORY, random_state=SEED)


def run_main(users, categories, names, rankers, out, workers):
    sample = main_sample(users)
    prompts = user_prompts(sample, categories, names)
    res = run_llm(sample, categories, prompts, *rankers, out / "cache" / "main.jsonl", workers=workers)
    ap = pd.read_csv(RESULTS_DIR / "per_user_ap.csv.gz", usecols=["user", "popularity", "se1_legacy_k50"])
    res.merge(ap, on="user").to_csv(out / "main_per_user.csv.gz", index=False)


def run_order(users, categories, names, rankers, out, workers):
    sample = main_sample(users)
    prompts = user_prompts(sample, categories, names)
    for name, shuffle in [("repeat", False), ("shuffled", True)]:
        res = run_llm(sample, categories, prompts, *rankers, out / "cache" / f"order_{name}.jsonl",
                      shuffle=shuffle, workers=workers, conditions=["llm_k25"], baseline=False)
        res.to_csv(out / f"order_{name}_per_user.csv.gz", index=False)
    trials = baseline_order_trials(categories, rankers[1], out / "cache" / "baseline_trials.jsonl",
                                   repeats=BASELINE_TRIALS, workers=workers)
    trials.to_csv(out / "baseline_trials.csv", index=False)


def band_setup(users, categories):
    """Band candidates (with SocialVec embeddings, in spreadsheet order) and per-topic exclusions."""
    sheets = load_category_sheets()
    vocab = SocialVec("2020")
    candidates, exclude = {}, {}
    for topic in PAPER_CATEGORY_ORDER:
        if topic not in sheets or len(sheets[topic]) < BAND_END:
            continue
        band = popularity_band(sheets[topic], BAND_START, BAND_END)
        band = band[[i in vocab for i in band["twitter_user_id"]]].sort_values("sheet_row")
        candidates[topic] = band[["twitter_user_id", "twitter_name", "followers_count"]].reset_index(drop=True)
        exclude[topic] = ({int(i) for i in sheets[topic]["twitter_user_id"]}
                          | {int(i) for i in categories[topic]["twitter_user_id"]})
    return candidates, exclude


def run_band(users, categories, names, rankers, out, workers):
    candidates, exclude = band_setup(users, categories)
    eligible = users[users["topic"].isin(candidates)].copy()
    ids = {t: set(c["twitter_user_id"]) for t, c in candidates.items()}
    eligible = eligible[[not ids[t].isdisjoint(f) for t, f in zip(eligible["topic"], eligible["friends_ids"])]]
    sample = eligible.groupby("topic").sample(n=BAND_USERS_PER_CATEGORY, random_state=SEED)
    prompts = user_prompts(sample, categories, names, exclude=exclude)
    res = run_llm(sample, candidates, prompts, *rankers, out / "cache" / "band.jsonl", workers=workers)

    band_ap = pd.read_csv(RESULTS_DIR / f"popularity_band_{BAND_START}_{BAND_END}" / "per_user_ap.csv.gz")
    band_ap = band_ap[(band_ap["population"] == "category") & (band_ap["candidates"] == "band")
                      & (band_ap["exclusion"] == "exclude_category")][["user", "random", "popularity", "socialvec"]]
    res.merge(band_ap, on="user").to_csv(out / "band_per_user.csv.gz", index=False)


# ---------------------------------------------------------------- analysis
def by_category(df, columns, order=PAPER_CATEGORY_ORDER):
    t = df.groupby("topic")[columns].mean()
    t = t.reindex([c for c in order if c in t.index])
    t.insert(0, "users", df.groupby("topic").size())
    t.loc["Overall"] = t.mean()
    t.loc["Overall", "users"] = len(df)
    return t


def p_value(a, b):
    d = np.asarray(a) - np.asarray(b)
    return wilcoxon(d).pvalue if (d != 0).any() else np.nan


def popularity_order(categories, topic):
    return [int(i) for i in categories[topic].sort_values("followers_count", ascending=False)["twitter_user_id"]]


def labelled(t):
    t = t.copy()
    t.index = [CATEGORY_LABELS.get(i, "**Overall**") for i in t.index]
    if "users" in t:
        t["users"] = t["users"].map(lambda v: f"{v:,.0f}")
    return t


def analyze(out, categories, main_users):
    sections = ["# LLM experiments: summary tables\n"]

    # ---- main
    main = pd.read_csv(out / "main_per_user.csv.gz")
    t3 = by_category(main, ["popularity", "se1_legacy_k50"] + LLM_COLUMNS)
    t3.to_csv(out / "table3_main.csv")
    overall = t3.loc["Overall"]
    gains = {c: overall[c] / overall["llm_baseline"] - 1 for c in ["llm_k12", "llm_k25", "llm_k50"]}
    tests = {c: p_value(main[c], main["llm_baseline"]) for c in ["llm_k12", "llm_k25", "llm_k50"]}
    complete = {c: np.mean([coverage(r, popularity_order(categories, t)) == 1.0
                            for r, t in zip(main[f"resp_{c}"], main["topic"])]) for c in LLM_COLUMNS}
    stats = pd.DataFrame({
        "overall MAP": [overall[c] for c in LLM_COLUMNS],
        "gain vs LLM baseline": [np.nan] + [gains[c] for c in ["llm_k12", "llm_k25", "llm_k50"]],
        "p vs LLM baseline": [np.nan] + [tests[c] for c in ["llm_k12", "llm_k25", "llm_k50"]],
        "complete rankings": [complete[c] for c in LLM_COLUMNS],
        "paper (GPT-4o)": [paper.TABLE3[c].mean() for c in LLM_COLUMNS],
    }, index=LLM_COLUMNS)
    stats.to_csv(out / "table3_stats.csv")
    sections += ["## Table 3 rerun (per category MAP)\n", markdown_table(labelled(t3)), "",
                 "## Table 3 rerun: overall\n", markdown_table(stats, floatfmt="{:.4g}"), ""]

    # ---- order
    rep = pd.read_csv(out / "order_repeat_per_user.csv.gz").set_index("user")
    shf = pd.read_csv(out / "order_shuffled_per_user.csv.gz").set_index("user")
    m = main.set_index("user").loc[rep.index]
    order_df = pd.DataFrame({"topic": m["topic"], "original": m["llm_k25"], "repeat": rep["llm_k25"],
                             "shuffled": shf["llm_k25"], "popularity": m["popularity"]})
    t_order = by_category(order_df.reset_index(), ["popularity", "original", "repeat", "shuffled"])
    t_order.to_csv(out / "order_by_category.csv")

    def taus(responses, references):
        return np.nanmean([order_agreement(r, ref) for r, ref in zip(responses, references)])

    pop_orders = [popularity_order(categories, t) for t in m["topic"]]
    shf_orders = [json.loads(o) for o in shf["order_llm_k25"]]
    out_orders = {"original": [ranked_ids(r) for r in m["resp_llm_k25"]],
                  "repeat": [ranked_ids(r) for r in rep["resp_llm_k25"]]}
    rows = {
        "MAP": [order_df["original"].groupby(m["topic"]).mean().mean(),
                order_df["repeat"].groupby(m["topic"]).mean().mean(),
                order_df["shuffled"].groupby(m["topic"]).mean().mean()],
        "p vs original (Wilcoxon)": [np.nan, p_value(order_df["repeat"], order_df["original"]),
                                     p_value(order_df["shuffled"], order_df["original"])],
        "mean |ΔAP| vs original": [0.0, (order_df["repeat"] - order_df["original"]).abs().mean(),
                                   (order_df["shuffled"] - order_df["original"]).abs().mean()],
        "τ with presented order": [taus(m["resp_llm_k25"], pop_orders), taus(rep["resp_llm_k25"], pop_orders),
                                   taus(shf["resp_llm_k25"], shf_orders)],
        "τ with popularity order": [taus(m["resp_llm_k25"], pop_orders), taus(rep["resp_llm_k25"], pop_orders),
                                    taus(shf["resp_llm_k25"], pop_orders)],
        "τ with original-run output": [1.0, taus(rep["resp_llm_k25"], out_orders["original"]),
                                       taus(shf["resp_llm_k25"], out_orders["original"])],
        "output = presented order": [
            np.mean([ranked_ids(r) == o for r, o in zip(m["resp_llm_k25"], pop_orders)]),
            np.mean([ranked_ids(r) == o for r, o in zip(rep["resp_llm_k25"], pop_orders)]),
            np.mean([ranked_ids(r) == o for r, o in zip(shf["resp_llm_k25"], shf_orders)])],
    }
    order_stats = pd.DataFrame(rows, index=["original", "repeat", "shuffled"]).T
    order_stats.to_csv(out / "order_stats.csv")
    sections += ["## Order experiment, k = 25 (per category MAP)\n", markdown_table(labelled(t_order)), "",
                 "## Order experiment, k = 25: agreement\n", markdown_table(order_stats, floatfmt="{:.4g}"), ""]

    # ---- baseline trials
    trials = pd.read_csv(out / "baseline_trials.csv")
    relevant = {u: [f for f in fr if f in set(categories[t]["twitter_user_id"])]
                for u, t, fr in zip(main_users.index, main_users["topic"], main_users["friends_ids"])}
    trial_rows = []
    for _, tr in trials.iterrows():
        users_t = main_users[main_users["topic"] == tr["topic"]].index
        trial_rows.append({
            "topic": tr["topic"], "order_type": tr["order_type"], "trial": tr["trial"],
            "MAP": np.mean([llm_average_precision(tr["response"], relevant[u]) for u in users_t]),
            "tau_presented": order_agreement(tr["response"], json.loads(tr["order"])),
            "tau_popularity": order_agreement(tr["response"], popularity_order(categories, tr["topic"])),
            "ranking": json.dumps(ranked_ids(tr["response"])),
        })
    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(out / "baseline_trials_scored.csv", index=False)
    per_trial = trial_df.groupby(["order_type", "trial"])["MAP"].mean()  # macro over categories
    base_stats = pd.DataFrame({
        "overall MAP (mean over trials)": trial_df.groupby(["order_type", "trial"])["MAP"].mean().groupby("order_type").mean(),
        "SD across trials": per_trial.groupby("order_type").std(),
        "τ with presented order": trial_df.groupby("order_type")["tau_presented"].mean(),
        "τ with popularity order": trial_df.groupby("order_type")["tau_popularity"].mean(),
        "distinct rankings per category (of 10)": trial_df.groupby(["order_type", "topic"])["ranking"].nunique().groupby("order_type").mean(),
    })
    base_stats.to_csv(out / "baseline_trial_stats.csv")
    sections += ["## Non-personalized baseline: 10 original-order and 10 shuffled calls per category\n",
                 markdown_table(base_stats, floatfmt="{:.4g}"), ""]

    # ---- band
    band = pd.read_csv(out / "band_per_user.csv.gz")
    t_band = by_category(band, ["random", "popularity", "socialvec"] + LLM_COLUMNS)
    t_band.to_csv(out / "band_by_category.csv")
    band_tests = {c: p_value(band[c], band["popularity"]) for c in ["socialvec"] + LLM_COLUMNS}
    sections += [f"## Ranks {BAND_START}-{BAND_END} (per category MAP)\n", markdown_table(labelled(t_band)), "",
                 "Paired Wilcoxon p vs popularity: "
                 + ", ".join(f"{k} {v:.3g}" for k, v in band_tests.items()), ""]
    pd.Series(band_tests).to_csv(out / "band_tests.csv")

    (out / "tables.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))


# ---------------------------------------------------------------- entry point
def cache_only_rankers():
    """Rankers for rescoring saved responses: any response missing from the cache aborts the run."""
    def missing(*args):
        raise BudgetExceeded("response not in cache (rescoring makes no model calls)")
    return missing, missing


def fake_rankers():
    """Offline stand-ins: echo the presented candidate order."""
    def personal(followed, candidates):
        return json.dumps([c["twitter_user_id"] for c in json.loads(candidates)])

    def default(candidates):
        return json.dumps([c["twitter_user_id"] for c in json.loads(candidates)])
    return personal, default


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", nargs="+", default=["main", "order", "band"], choices=["main", "order", "band"])
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="low", help="reasoning effort: none, low, medium, high, xhigh")
    p.add_argument("--budget", type=float, default=1.5, help="stop calling the model at this cost (USD)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--rescore", action="store_true",
                   help="rebuild the per-user results from cached responses only (no model calls)")
    p.add_argument("--dry-run", action="store_true", help="use a fake model that echoes the candidate order")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "llm")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    categories = load_categories()
    users = load_users(categories)
    main_users = main_sample(users)

    if not args.analyze_only:
        metadata = load_metadata()
        names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))
        cap = None
        info_path = args.out / "run_info.json"
        runs = json.loads(info_path.read_text()) if info_path.exists() else []
        if args.dry_run:
            rankers = fake_rankers()
        elif args.rescore:
            rankers = cache_only_rankers()
        else:
            import dspy
            spent_before = sum(r["cost_usd"] for r in runs)  # the budget covers earlier (resumed) runs too
            lm = dspy.LM(f"openai/{args.model}", api_key=load_openai_key(), reasoning_effort=args.effort,
                         cache=False, num_retries=8)
            cap = SpendingCap(lm, args.budget - spent_before)
            rankers = tuple(cap.guard(f) for f in make_dspy_rankers(lm=lm))
        try:
            for part in args.parts:
                {"main": run_main, "order": run_order, "band": run_band}[part](
                    users, categories, names, rankers, args.out, args.workers)
                if cap is not None:
                    print(f"after {part}: {cap.calls} calls, ${cap.update():.3f} spent in this run")
        finally:
            if cap is not None:
                cap.update()
                runs.append({"model": args.model, "reasoning_effort": args.effort, "parts": args.parts,
                             "calls": cap.calls, "cost_usd": round(cap.spent, 4), "budget_usd": args.budget})
                info_path.write_text(json.dumps(runs, indent=2))
    analyze(args.out, categories, main_users)


if __name__ == "__main__":
    main()
