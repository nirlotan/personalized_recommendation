# LLM runs: what was run, what it cost, and what was not run

Costs are as logged in `results/llm/run_info.json`, in USD. Table numbers refer to `results/results_v2.html`.

## Totals

| Model | Role | Calls | Cost |
|---|---|---|---|
| GPT-6 Luna (OpenAI, effort *low*; *medium* for the 200-candidate runs) | Ranks the candidates | 39,938 | $9.89 |
| GPT-4o (OpenAI) | Writes the demographic descriptions | 2,260 | $3.41 |
| Claude Haiku 5.5 (Anthropic, adaptive thinking, effort *low*, Batch API) | Ranks the candidates (second model) | 17,745 | $2.03 |
| **All** | | **59,943** | **$15.33** |

## What was run

### GPT-6 Luna

| Experiment | Users | Calls | Cost |
|---|---|---|---|
| Table 3 (paper's LLM table), Section 4 (candidate order) and a first 30-per-category Table 9 | 700 (50 per category, 14 categories) | 4,613 | $0.86 |
| Table 3 and Tables 9–11 again, with shuffled candidates in the personalized prompts | 700; 1,798 and 1,751 | 9,155 | $1.77 |
| Tables 9–11 extended to 200 users per category | 1,798 (ranks 81–100), 1,751 (ranks 1–20) | 12,865 | $2.39 |
| Section 5, ranking from demographics (all versions: 270 users, then 437, then all 1,798) | 270 → 1,798 | 11,433 | $2.03 |
| Backup page: ranks 181–200 and 200-candidate experiments (extended lists, not in results_v2) | 270 | 1,872 | $2.84 |

### GPT-4o (descriptions only)

| Experiment | Descriptions | Cost |
|---|---|---|
| Section 5, first version (results.html) | 270 | $0.39 |
| Section 5, one-value prompt, filtered 437 users (superseded; 255 reused) | 456 | $0.71 |
| Section 5, all Table 9 users | 1,534 new | $2.31 |

### Claude Haiku 5.5

| Experiment | Users | Calls | Cost |
|---|---|---|---|
| Section 5 rankings, all five conditions (no information, demographics, 12/25/50 names), both bands; Figure 4, Table 16, Supplement (Tables 20–22) | 1,798 (ranks 81–100), 1,751 (ranks 1–20) | 17,745 (50 pilot + 17,695 batch) | $2.03 |

Because Haiku used the same users, candidate orders and prompts as Tables 9–11, its no-information and account-name rankings also cover the LLM columns of Tables 9–11.

## What was not run (estimated cost)

Estimates use the measured cost per call: GPT-6 Luna $0.00026, GPT-4o $0.0015 per description, Claude Haiku 5.5 $0.00011 (Batch API).

| Not run | GPT (Luna rankings, GPT-4o descriptions) | Claude Haiku 5.5 |
|---|---|---|
| Table 3 on all 11,153 users (now 700) | ~31K calls, ~$8 | ~34K calls, ~$4 |
| Table 3 with Claude at the current 700 users (included in the row above) | — (done with GPT) | ~2.2K calls, ~$0.25 |
| Tables 9–11 on all 3,798 eligible users (now 1,798) | ~16K calls, ~$4 | ~16K calls, ~$1.8 |
| Section 5 on all 3,798 eligible users | ~2K descriptions + ~4K rankings, ~$4 | ~4K calls (demographics condition), ~$0.5 |
| Section 4 (candidate order test) with Claude | — (done with GPT) | ~2.4K calls, ~$0.3 |
| Everything above together | **~$16** | **~$6.5** |
| Section 5 with Claude Sonnet 5.5 instead of Haiku (all five conditions, current users, thinking on) | — | ~17.7K calls, ~$75 |
