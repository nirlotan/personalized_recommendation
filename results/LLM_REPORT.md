# LLM personalization with GPT-6 Luna: rerun of Table 3 and a candidate-order experiment

*All numbers come from `python run_llm_experiments.py` (outputs in `results/llm/`). 4,613 model calls in total, costing $0.86. Model `gpt-6-luna`, reasoning effort `low`.*

## Summary

- **The paper's LLM finding replicates with a newer model.** Listing a user's favored entities makes the LLM's ranking personalized. On a stratified sample of 700 users, MAP rises from 0.474 without user information to 0.549 with 12 entities and 0.611 with 50 (+16% and +29%). GPT-4o reported 0.490, 0.553 and 0.600 (+13% and +23%).
- **The order of the candidates matters, but only a little for personalized prompts.** Shuffling the candidates, which the original prompts list by popularity, lowers MAP at k = 25 from 0.554 to 0.541 (p < 0.001). Rerunning the identical prompt changes nothing on average (0.553, p = 0.97).
- **The non-personalized baseline depends more on the order.** Its MAP drops by 0.022 when the candidates are shuffled, about 4 times the spread across repeated calls. Its output follows the presented order substantially (τ = 0.51).
- **The outputs are not deterministic.** Ten identical baseline prompts gave 9.9 different rankings on average, and two identical personalized prompts change a user's AP by 0.058 on average.

## 1. Rerun of the LLM experiment (Table 3)

**Setup.** We follow the original protocol exactly, using the authors' dspy prompts, with GPT-6 Luna in place of GPT-4o.
- **Users:** to keep costs low we evaluate a stratified random sample of 50 users per category (700 users, seed 76) instead of all 11,153. With per-user AP varying by about 0.25–0.3, the overall MAP has a standard error of about 0.01.
- **Conditions:** as in the paper, the personalized prompts give the names of 12 or 25 entities sampled from the user's closed-world profile, or of 50 accounts sampled from everything they follow. The candidates are listed by follower count, and the non-personalized baseline lists them in spreadsheet order.
- **Scoring:** we use the legacy AP function, which skips relevant ids missing from the response.

**Table L1.** MAP per category. The first two columns are computed for the same 700 users. GPT-4o values are the paper's, computed on all users.

| Category | Popularity | Social, k = 50 | Baseline | k = 12 | k = 25 | k = 50 | GPT-4o baseline | GPT-4o k = 12 | GPT-4o k = 25 | GPT-4o k = 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Musical artists | 0.505 | 0.639 | 0.521 | 0.599 | 0.594 | 0.699 | 0.508 | 0.582 | 0.611 | 0.667 |
| News outlets | 0.645 | 0.710 | 0.677 | 0.744 | 0.756 | 0.753 | 0.648 | 0.695 | 0.698 | 0.716 |
| Comedians | 0.545 | 0.600 | 0.578 | 0.640 | 0.642 | 0.662 | 0.576 | 0.638 | 0.643 | 0.655 |
| Politicians | 0.737 | 0.736 | 0.802 | 0.823 | 0.830 | 0.832 | 0.785 | 0.797 | 0.801 | 0.811 |
| TV stations | 0.450 | 0.615 | 0.474 | 0.604 | 0.627 | 0.667 | 0.507 | 0.636 | 0.640 | 0.664 |
| Actors | 0.550 | 0.619 | 0.496 | 0.628 | 0.623 | 0.678 | 0.511 | 0.605 | 0.603 | 0.662 |
| TV shows | 0.364 | 0.603 | 0.433 | 0.610 | 0.633 | 0.624 | 0.427 | 0.595 | 0.612 | 0.637 |
| Sports teams | 0.348 | 0.478 | 0.322 | 0.371 | 0.362 | 0.554 | 0.347 | 0.393 | 0.391 | 0.571 |
| Fashion | 0.332 | 0.435 | 0.335 | 0.380 | 0.369 | 0.471 | 0.377 | 0.409 | 0.409 | 0.493 |
| Journalists | 0.628 | 0.656 | 0.684 | 0.692 | 0.695 | 0.712 | 0.680 | 0.673 | 0.675 | 0.704 |
| TV hosts | 0.336 | 0.412 | 0.299 | 0.424 | 0.453 | 0.538 | 0.313 | 0.436 | 0.451 | 0.522 |
| Films | 0.245 | 0.413 | 0.175 | 0.273 | 0.254 | 0.361 | 0.216 | 0.306 | 0.308 | 0.399 |
| Food chains | 0.441 | 0.498 | 0.426 | 0.444 | 0.434 | 0.449 | 0.492 | 0.496 | 0.504 | 0.496 |
| Car makers | 0.432 | 0.409 | 0.411 | 0.450 | 0.485 | 0.551 | 0.472 | 0.480 | 0.487 | 0.409 |
| **Overall** | 0.468 | 0.559 | 0.474 | 0.549 | 0.554 | 0.611 | 0.490 | 0.553 | 0.559 | 0.600 |
| Δ vs. LLM baseline | | | | +16% | +17% | +29% | | +13% | +14% | +23% |

**Findings.**
- **Personalized prompts outperform the baseline.** All three are significantly better than the non-personalized ranking (Wilcoxon p < 0.001). Gains grow with the number of entities listed: +16%, +17% and +29%. The pattern matches the paper's GPT-4o results, with somewhat larger gains.
- **The baseline is close to popularity.** The non-personalized LLM ranking (0.474) is within 0.005 of follower-count popularity (0.468; p = 0.03 at the user level). Without user information, the model contributes little beyond popularity.
- **Compared with social embeddings:**
  - *12 entities:* the LLM given only 12 entity names (0.549) performs on par with SocialVec embeddings built from 50 sampled accounts (0.559, p = 0.75).
  - *50 accounts:* the LLM outperforms them (0.611 vs. 0.559, p < 0.001).

**Caveat (k = 50).** As in the original code, the 50 accounts are sampled from everything the user follows, *including* the target category. In 20% of the k = 50 prompts, at least one listed account is itself a candidate the user follows. Restricted to the 562 users without such a leak, the LLM still outperforms the embeddings: 0.589 vs. 0.534 on average over users.

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

| Measure | Original order | Original order, repeated | Shuffled order |
|---|---|---|---|
| MAP | 0.554 | 0.553 | 0.541 |
| p vs. original run (Wilcoxon) | | 0.974 | < 0.001 |
| Mean per-user \|ΔAP\| vs. original run | | 0.058 | 0.085 |
| Kendall τ: output vs. presented order | 0.315 | 0.315 | 0.091 |
| Kendall τ: output vs. popularity order | 0.315 | 0.315 | 0.200 |
| Kendall τ: output vs. original-run output | 1 | 0.776 | 0.667 |

**Table L3.** Non-personalized baseline, 14 categories × 10 calls per order type. MAP is computed on the same 700 users.

| Measure | Original order (10 calls / category) | Shuffled order (10 calls / category) |
|---|---|---|
| Overall MAP, mean over trials | 0.475 | 0.452 |
| SD of overall MAP across trials | 0.004 | 0.005 |
| Kendall τ: output vs. presented order | 0.513 | 0.133 |
| Kendall τ: output vs. popularity order | 0.426 | 0.324 |
| Distinct rankings among 10 calls | 9.9 | 10.0 |

**Findings.**
- **The output partly follows the input order.** With popularity-sorted candidates, the personalized rankings correlate with the presented order (τ = 0.315). When the order is shuffled, the correlation with the presented order nearly vanishes (τ = 0.091). The correlation with popularity also falls, from 0.315 to 0.200. About a third of the model's agreement with popularity therefore comes from the order in which the candidates are shown; the rest reflects its own knowledge of which entities are popular.
- **Personalization is only slightly affected.** Overall MAP drops by 0.013 (-2.3%). The drop is statistically significant but small next to the 0.058 per-user variation between two identical runs. It is largest for TV shows (-0.058), Car makers (-0.046), Musical artists (-0.044). Shuffled MAP remains far above the non-personalized baseline.
- **The non-personalized baseline depends more on the order.**
  - *Order tracking:* without user information, the model tracks the presented order much more closely (τ = 0.513).
  - *Score:* its score falls from 0.475 to 0.452 when the order is shuffled. The SD across trials is only about 0.004.
  - *Interpretation:* the spreadsheet order is itself informative. Listing the candidates as given scores 0.487 on these users, so part of the baseline's performance comes from the order in which the candidates were provided.
  - *Implication:* the gain of personalized over non-personalized prompts is therefore, if anything, underestimated when the baseline is given a favorable order.
- **Run-to-run variation is substantial for individual users but averages out.** Two runs of the identical prompt agree only moderately on the ranking (τ = 0.776) and change a user's AP by 0.058 on average. Overall MAP differs by less than 0.002.

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
