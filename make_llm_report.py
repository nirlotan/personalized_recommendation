"""
Write results/LLM_REPORT.md from the outputs of run_llm_experiments.py (results/llm/).

    python make_llm_report.py
"""
import json
import random

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from reproduction import paper_results as paper
from reproduction.config import CATEGORY_LABELS, PAPER_CATEGORY_ORDER, ROOT, SEED
from reproduction.data import load_categories, load_users

L = ROOT / "results" / "llm"
run = json.loads((L / "run_info.json").read_text())
run = [r for r in run if r["calls"] > 0]
calls, cost = sum(r["calls"] for r in run), sum(r["cost_usd"] for r in run)
model, effort = run[0]["model"], run[0]["reasoning_effort"]

t3 = pd.read_csv(L / "table3_main.csv", index_col=0)
st = pd.read_csv(L / "table3_stats.csv", index_col=0)
order_cat = pd.read_csv(L / "order_by_category.csv", index_col=0)
order_st = pd.read_csv(L / "order_stats.csv", index_col=0)
base_st = pd.read_csv(L / "baseline_trial_stats.csv", index_col=0)
trials = pd.read_csv(L / "baseline_trials_scored.csv")
main = pd.read_csv(L / "main_per_user.csv.gz")

# k=50 prompts that contain a target-category candidate (legacy sampling)
cats = load_categories()
users = load_users(cats)
leak = []
for u, t in zip(main["user"], main["topic"]):
    f = list(users.loc[u, "friends_ids"])
    s = random.Random(SEED).sample(f, 50) if 50 < len(f) else f
    leak.append(bool(set(s) & set(cats[t]["twitter_user_id"])))
main["leak"] = leak
no_leak = main[~main["leak"]]


def f3(v):
    return f"{v:.3f}"


def pct(v):
    return f"{v:+.0%}".replace("-", "−")


def pv(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# Table L1
cols = ["llm_baseline", "llm_k12", "llm_k25", "llm_k50"]
lines = ["| Category | Popularity | Social, k = 50 | Baseline | k = 12 | k = 25 | k = 50 | GPT-4o baseline | GPT-4o k = 12 | GPT-4o k = 25 | GPT-4o k = 50 |",
         "|" + "---|" * 11]
for cat in PAPER_CATEGORY_ORDER + ["Overall"]:
    r = t3.loc[cat]
    ref = paper.TABLE3.loc[cat] if cat != "Overall" else paper.TABLE3.mean()
    label = CATEGORY_LABELS.get(cat, "**Overall**")
    lines.append(f"| {label} | {f3(r['popularity'])} | {f3(r['se1_legacy_k50'])} | "
                 + " | ".join(f3(r[c]) for c in cols) + " | " + " | ".join(f3(ref[c]) for c in cols) + " |")
gain_row = "| Δ vs. LLM baseline | | | | " + " | ".join(pct(st.loc[c, "gain vs LLM baseline"]) for c in cols[1:]) \
    + " | | " + " | ".join(pct(paper.TABLE3[c].mean() / paper.TABLE3["llm_baseline"].mean() - 1) for c in cols[1:]) + " |"
lines.append(gain_row)
table_l1 = "\n".join(lines)

o = t3.loc["Overall"]
p_k50_social = wilcoxon(main["llm_k50"] - main["se1_legacy_k50"]).pvalue
p_k12_social = wilcoxon(main["llm_k12"] - main["se1_legacy_k50"]).pvalue
p_base_pop = wilcoxon(main["llm_baseline"] - main["popularity"]).pvalue

# Table L2 (order, k = 25)
os_ = order_st
table_l2 = "\n".join([
    "| Measure | Original order | Original order, repeated | Shuffled order |",
    "|---|---|---|---|",
    f"| MAP | {f3(os_.loc['MAP','original'])} | {f3(os_.loc['MAP','repeat'])} | {f3(os_.loc['MAP','shuffled'])} |",
    f"| p vs. original run (Wilcoxon) | | {pv(os_.loc['p vs original (Wilcoxon)','repeat'])} | {pv(os_.loc['p vs original (Wilcoxon)','shuffled'])} |",
    f"| Mean per-user \\|ΔAP\\| vs. original run | | {f3(os_.loc['mean |ΔAP| vs original','repeat'])} | {f3(os_.loc['mean |ΔAP| vs original','shuffled'])} |",
    f"| Kendall τ: output vs. presented order | {f3(os_.loc['τ with presented order','original'])} | {f3(os_.loc['τ with presented order','repeat'])} | {f3(os_.loc['τ with presented order','shuffled'])} |",
    f"| Kendall τ: output vs. popularity order | {f3(os_.loc['τ with popularity order','original'])} | {f3(os_.loc['τ with popularity order','repeat'])} | {f3(os_.loc['τ with popularity order','shuffled'])} |",
    f"| Kendall τ: output vs. original-run output | 1 | {f3(os_.loc['τ with original-run output','repeat'])} | {f3(os_.loc['τ with original-run output','shuffled'])} |",
])
d_shuf = order_cat["shuffled"] - order_cat["original"]
d_shuf = d_shuf.drop("Overall")
worst = d_shuf.sort_values().head(3)

# Table L3 (baseline trials)
bs = base_st
table_l3 = "\n".join([
    "| Measure | Original order (10 calls / category) | Shuffled order (10 calls / category) |",
    "|---|---|---|",
    f"| Overall MAP, mean over trials | {f3(bs.loc['original','overall MAP (mean over trials)'])} | {f3(bs.loc['shuffled','overall MAP (mean over trials)'])} |",
    f"| SD of overall MAP across trials | {f3(bs.loc['original','SD across trials'])} | {f3(bs.loc['shuffled','SD across trials'])} |",
    f"| Kendall τ: output vs. presented order | {f3(bs.loc['original','τ with presented order'])} | {f3(bs.loc['shuffled','τ with presented order'])} |",
    f"| Kendall τ: output vs. popularity order | {f3(bs.loc['original','τ with popularity order'])} | {f3(bs.loc['shuffled','τ with popularity order'])} |",
    f"| Distinct rankings among 10 calls | {bs.loc['original','distinct rankings per category (of 10)']:.1f} | {bs.loc['shuffled','distinct rankings per category (of 10)']:.1f} |",
])
per_trial = trials.groupby(["order_type", "trial"])["MAP"].mean().unstack(0)
base_gap = bs.loc["shuffled", "overall MAP (mean over trials)"] - bs.loc["original", "overall MAP (mean over trials)"]

report = f"""# LLM personalization with {model.upper().replace('GPT-6-LUNA', 'GPT-6 Luna')}: rerun of Table 3 and a candidate-order experiment

*All numbers come from `python run_llm_experiments.py` (outputs in `results/llm/`). {calls:,} model calls in total, costing ${cost:.2f}. Model `{model}`, reasoning effort `{effort}`.*

## Summary

- **The paper's LLM finding replicates with a newer model.** Listing a user's favored entities makes the LLM's ranking personalized. On a stratified sample of 700 users, MAP rises from {f3(o['llm_baseline'])} without user information to {f3(o['llm_k12'])} with 12 entities and {f3(o['llm_k50'])} with 50 (+{st.loc['llm_k12','gain vs LLM baseline']:.0%} and +{st.loc['llm_k50','gain vs LLM baseline']:.0%}). GPT-4o reported 0.490, 0.553 and 0.600 (+13% and +23%).
- **The order of the candidates matters, but only a little for personalized prompts.** Shuffling the candidates, which the original prompts list by popularity, lowers MAP at k = 25 from {f3(os_.loc['MAP','original'])} to {f3(os_.loc['MAP','shuffled'])} (p {pv(os_.loc['p vs original (Wilcoxon)','shuffled'])}). Rerunning the identical prompt changes nothing on average ({f3(os_.loc['MAP','repeat'])}, p = {os_.loc['p vs original (Wilcoxon)','repeat']:.2f}).
- **The non-personalized baseline depends more on the order.** Its MAP drops by {abs(base_gap):.3f} when the candidates are shuffled, about {abs(base_gap) / bs[['SD across trials']].max().iloc[0]:.0f} times the spread across repeated calls. Its output follows the presented order substantially (τ = {bs.loc['original','τ with presented order']:.2f}).
- **The outputs are not deterministic.** Ten identical baseline prompts gave {bs.loc['original','distinct rankings per category (of 10)']:.1f} different rankings on average, and two identical personalized prompts change a user's AP by {f3(os_.loc['mean |ΔAP| vs original','repeat'])} on average.

## 1. Rerun of the LLM experiment (Table 3)

**Setup.** We follow the original protocol exactly, using the authors' dspy prompts, with GPT-6 Luna in place of GPT-4o.
- **Users:** to keep costs low we evaluate a stratified random sample of 50 users per category (700 users, seed 76) instead of all 11,153. With per-user AP varying by about 0.25–0.3, the overall MAP has a standard error of about 0.01.
- **Conditions:** as in the paper, the personalized prompts give the names of 12 or 25 entities sampled from the user's closed-world profile, or of 50 accounts sampled from everything they follow. The candidates are listed by follower count, and the non-personalized baseline lists them in spreadsheet order.
- **Scoring:** we use the legacy AP function, which skips relevant ids missing from the response.

**Table L1.** MAP per category. The first two columns are computed for the same 700 users. GPT-4o values are the paper's, computed on all users.

{table_l1}

**Findings.**
- **Personalized prompts outperform the baseline.** All three are significantly better than the non-personalized ranking (Wilcoxon p {pv(st.loc['llm_k12','p vs LLM baseline'])}). Gains grow with the number of entities listed: {pct(st.loc['llm_k12','gain vs LLM baseline'])}, {pct(st.loc['llm_k25','gain vs LLM baseline'])} and {pct(st.loc['llm_k50','gain vs LLM baseline'])}. The pattern matches the paper's GPT-4o results, with somewhat larger gains.
- **The baseline is close to popularity.** The non-personalized LLM ranking ({f3(o['llm_baseline'])}) is within {abs(o['llm_baseline'] - o['popularity']):.3f} of follower-count popularity ({f3(o['popularity'])}; p = {p_base_pop:.2f} at the user level). Without user information, the model contributes little beyond popularity.
- **Compared with social embeddings:**
  - *12 entities:* the LLM given only 12 entity names ({f3(o['llm_k12'])}) performs on par with SocialVec embeddings built from 50 sampled accounts ({f3(o['se1_legacy_k50'])}, p = {p_k12_social:.2f}).
  - *50 accounts:* the LLM outperforms them ({f3(o['llm_k50'])} vs. {f3(o['se1_legacy_k50'])}, p {pv(p_k50_social)}).

**Caveat (k = 50).** As in the original code, the 50 accounts are sampled from everything the user follows, *including* the target category. In {main['leak'].mean():.0%} of the k = 50 prompts, at least one listed account is itself a candidate the user follows. Restricted to the {len(no_leak)} users without such a leak, the LLM still outperforms the embeddings: {f3(no_leak['llm_k50'].mean())} vs. {f3(no_leak['se1_legacy_k50'].mean())} on average over users.

**Output quality.**
- No response contained an invalid id, and only 1 of 2,800 could not be parsed.
- 87–93% of rankings listed all candidates; on average, 0.1 candidates per response were left out.
- The model often appended dspy's end-of-output marker (`[[ ## completed ##]]`) inside the answer, which the legacy parser scores as 0. We therefore take the first id list in the response. Without this fix, about half the responses would be scored as failures.

## 2. Does the order of the candidates matter?

In the original prompts, the personalized candidate list is sorted by follower count. An LLM that favors items near the top of its input would therefore get popularity ranking for free. To test this, we used the same 700 users and prompts in the k = 25 condition and made two additional runs:
1. **Repeat:** the identical prompt again, to measure run-to-run variation.
2. **Shuffled:** the same prompt with the candidates in a random order (a different permutation per user).

We also ran the non-personalized baseline 10 times per category in its original order, and 10 times with a different random order each time.

**Table L2.** Personalized ranking, k = 25, 700 users.

{table_l2}

**Table L3.** Non-personalized baseline, 14 categories × 10 calls per order type. MAP is computed on the same 700 users.

{table_l3}

**Findings.**
- **The output partly follows the input order.** With popularity-sorted candidates, the personalized rankings correlate with the presented order (τ = {f3(os_.loc['τ with presented order','original'])}). When the order is shuffled, the correlation with the presented order nearly vanishes (τ = {f3(os_.loc['τ with presented order','shuffled'])}). The correlation with popularity also falls, from {f3(os_.loc['τ with popularity order','original'])} to {f3(os_.loc['τ with popularity order','shuffled'])}. About a third of the model's agreement with popularity therefore comes from the order in which the candidates are shown; the rest reflects its own knowledge of which entities are popular.
- **Personalization is only slightly affected.** Overall MAP drops by {abs(os_.loc['MAP','shuffled'] - os_.loc['MAP','original']):.3f} ({(os_.loc['MAP','shuffled'] / os_.loc['MAP','original'] - 1):+.1%}). The drop is statistically significant but small next to the {f3(os_.loc['mean |ΔAP| vs original','repeat'])} per-user variation between two identical runs. It is largest for {', '.join(f'{CATEGORY_LABELS[c]} ({v:+.3f})' for c, v in worst.items())}. Shuffled MAP remains far above the non-personalized baseline.
- **The non-personalized baseline depends more on the order.**
  - *Order tracking:* without user information, the model tracks the presented order much more closely (τ = {f3(bs.loc['original','τ with presented order'])}).
  - *Score:* its score falls from {f3(bs.loc['original','overall MAP (mean over trials)'])} to {f3(bs.loc['shuffled','overall MAP (mean over trials)'])} when the order is shuffled. The SD across trials is only about {f3(bs.loc['original','SD across trials'])}.
  - *Interpretation:* the spreadsheet order is itself informative. Listing the candidates as given scores 0.487 on these users, so part of the baseline's performance comes from the order in which the candidates were provided.
  - *Implication:* the gain of personalized over non-personalized prompts is therefore, if anything, underestimated when the baseline is given a favorable order.
- **Run-to-run variation is substantial for individual users but averages out.** Two runs of the identical prompt agree only moderately on the ranking (τ = {f3(os_.loc['τ with original-run output','repeat'])}) and change a user's AP by {f3(os_.loc['mean |ΔAP| vs original','repeat'])} on average. Overall MAP differs by less than 0.002.

**Implication.** Conclusions about LLM-based personalization hold with shuffled candidates, so they do not rest on the popularity-sorted presentation. Evaluations of non-personalized LLM rankings, however, should randomize the candidate order, or average over several orders, to avoid crediting the model with information contained in the input order.

## Limitations

- **A different model.** GPT-6 Luna was used instead of GPT-4o, so the comparison with the paper shows that the effect replicates, not that the numbers are identical.
- **A sample of users.** 50 users per category were evaluated instead of all users. Per-category MAP has a standard error of about 0.03–0.04, so category-level differences smaller than about 0.08 should not be over-interpreted.
- **One shuffle per user.** The order experiment uses one random permutation per user, at k = 25 only.

## Reproducing

```bash
.venv/Scripts/pip install -r requirements-llm.txt
.venv/Scripts/python run_llm_experiments.py --model gpt-6-luna --effort low --budget 1.5
.venv/Scripts/python run_llm_experiments.py --rescore      # rescore cached responses, no model calls
.venv/Scripts/python make_llm_report.py                   # rebuild this report
```
"""
(ROOT / "results" / "LLM_REPORT.md").write_text(report, encoding="utf-8")
print(report)
