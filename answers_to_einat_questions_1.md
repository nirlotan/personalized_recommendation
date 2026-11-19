# Answers to Einat's questions (round 1)

Answers checked against the code and the results of `results/results.html`. Nothing has been changed in the report yet;
each answer ends with the action it would need.

## Table 2 in the paper: is the delta with respect to popularity?

In our results page, yes. Our Table 1 reproduces the paper's Table 2, and its Δ is the relative gain over popularity:
social ÷ popularity − 1. For example, the paper's overall figures (0.571 vs. 0.468) give +22%. I could not confirm from the
paper PDF that the paper defines Δ the same way (the numbers stored from the paper's Table 2 have no Δ column).

**Action:** none, unless the paper defines Δ differently.

## Figure 3: legend

Figure 3 has a color legend for the two ranges (ranks 1–20 and 81–100). It does not explain the marks: each dot is one
account and the vertical bar is the range's median. That is only in the caption.

**Action:** add "dot = one account" and "bar = median" to the legend.

## Table 6: add Random for ranks 1–20

The value is already computed but not shown for ranks 1–20.

**Action:** add the column. No new runs.

## Figure 1: fix the popularity baseline, remove "target removed", adapt to 9 categories, fix the numbers in the text

Figure 1 and its text average over all 14 categories, so the popularity line (0.468) is the 14-category mean. The
per-category values are stored, so everything can be recomputed for the 9 categories without new runs. The paper's markers
(diamonds) are 14-category values and would no longer be comparable.

**Action:** recompute Figure 1, its data table (Table 2), the popularity line and the text numbers for the 9 categories,
and remove the "SE1 · target removed" line. Drop the paper's markers, or keep them with a note that they are 14-category values.

## Section 5: show an example of the followed entities, separated by commas

Not shown yet; the prompt section only has a placeholder.

**Action:** add one real example list for one user, identified only by internal row number.

## How were the entity names obtained (instead of the handles)?

- **In the LLM prompts:** each account's Twitter display name, from the SocialVec account metadata. Accounts without a
  known name are dropped, which is why the k = 12 prompts average 11.7 names and the 50-account samples give about 29
  named accounts.
- **In the entity tables (7 and B4):** the display names from the category spreadsheets, plus hand-mapped real names
  (`reproduction/entity_names.csv`).

**Action:** state the prompt-name source in the text.

## Were they selected randomly? Were 12/25 sampled from the closed sets? Are they sorted randomly?

Yes to all three.

- **k = 12 and k = 25:** random samples from the closed set: the user's followed entities in the other categories' lists,
  with the target category's entities removed.
- **k = 50:** a random sample from everything the user follows, with every listed entity of the target category removed.
  In Table 3 (the paper's setup) this removal is not done, so the target's entities can appear there.
- **Order:** the names appear in sampled order, which is random, not sorted.

**Action:** say this explicitly in the text.

## Statistics on users' follow patterns (median, minimum, maximum)? Did we filter bots, and why?

| Users | Accounts followed (min / median / max) | Followed category entities (min / median / max) |
|---|---|---|
| All 11,153 experiment users | 59 / 1,554 / 4,999 | 11 / 35 / 170 |
| Table 9 (1,798) | 68 / 1,950 / 4,999 | 11 / 44 / 170 |
| Section 5 (270) | 82 / 1,623 / 4,998 | 11 / 46 / 147 |

The maximum of 4,999 suggests the follow lists were collected with a 5,000-account cap. The only filter is the paper's
rule: a user must follow more than 10 of the category entities. There is no bot filter, because the paper did not use one
and we reproduced its setup.

**Action:** add this table. Optionally, check whether results change when likely bots are excluded (e.g. users following
an extreme number of accounts); no model calls needed.

## "Since LLMs often exhibit position bias, we list the candidate entities in random orders": add details

Shuffling is not applied the same way everywhere:

| Where | Baseline (no user information) | Personalized prompts |
|---|---|---|
| Table 3 (paper protocol) | 10 calls per category, each with a different random order | Sorted by follower count |
| Section 4 (order test) | Original vs. shuffled | k = 25: original vs. shuffled, same users |
| Tables 9–11 | One random order per user | Sorted by follower count |
| Section 5 / Figure 4 | One random order per user, shared by all of that user's conditions | Same shared random order |

The orders come from fixed random seeds, so the runs can be reproduced.

**Action:** add this explanation. The sentence only fully holds after the Table 9 fix below.

## Demographics: how many accounts were given? What was the prompt? Did it also see their names?

- **Accounts:** GPT-4o saw the user's k = 50 sample: 50 random followed accounts with the target category's entities
  removed, keeping only the named ones. The median is 30 names (range 3–50).
- **Prompt:** shown in full in Section 5 (Stage 1).
- **Names:** it saw only the display names of the followed accounts, never the user's own name or handle.

**Action:** state the account counts in the text.

## Table 9: "the personalized prompts list them by follower count" is problematic

Agreed. Section 4 found the order matters little for personalized prompts, but it should still be fixed: rerun the
k = 12/25/50 prompts with shuffled candidates, using the same random order per user as the baseline.

- **Covers:** Tables 9 and 10 (both columns) and Table 11.
- **Reuse:** the same orders and cache as Section 5, so its 270 and 264 users come from the cache for free.
- **Cost:** about 10,000 new calls, ~$2.

Table 3 also lists them by follower count (the paper's protocol); shuffling it too would cost ~$0.40.

**Action:** rerun Tables 9–11 shuffled (~$2). Table 3 is optional.

## Stage 2 instructions: include an example

There is no filled-in example yet.

**Action:** show one complete Stage 2 call for one user (internal row only): GPT-4o's description, the candidate list and
the model's answer. Note that the description, a stereotype-based guess about a real user, would appear in the page.

## Figure 4 is for 270 users: why a different number? The SocialVec results differ quite a lot

Figure 4 can only use users with a GPT-4o description. Those exist for the original 30 per category: 270 users, or 264 at
ranks 1–20. Table 9 now has 1,798 users. The SocialVec gap is a sampling difference:

| SocialVec, all entities | Figure 4 sample | Table 9 / 10 sample |
|---|---|---|
| Ranks 81–100 | 0.451 (270) | 0.418 (1,798) |
| Ranks 1–20 | 0.671 (264) | 0.660 (1,751) |

The account-name numbers also differ for now, because Figure 4 is shuffled and Table 9 is sorted.

**Action, either:**

- **Clarify the text:** say Figure 4 uses the first 30 per category and why. Free.
- **Extend the demographics to all 1,798 users:** GPT-4o descriptions plus rankings in both ranges, about $5. All numbers
  would then line up.

## Table 15: why were some users given multiple values? Can we get rid of them?

The multiple values are GPT-4o hedging, e.g. "in his 30s or 40s" or "Caucasian or mixed". The prompt asks for the most
likely value but does not forbid ranges. 135 of 270 descriptions hedge on age, 33 on ethnicity and 6 on education; none on
gender. Each currently counts under its first value.

**Options:**

- **Change the prompt to require exactly one value per attribute, and rerun.** Removes the problem at its source, about
  $0.50. Changes the Section 5 numbers.
- **Leave those users out of the statistics.** Not workable: it would drop half the users for age.
- **Keep the current rule (first value) and state it.** The current setup.

## Summary of actions

| Action | Cost |
|---|---|
| Figure 3 legend, Table 6 Random column, Figure 1 for 9 categories, Section 5 example and explanations, follow statistics, shuffling details, Stage 2 example | Free (page edits only) |
| Rerun Tables 9–11 with shuffled personalized prompts | ~$2 (+$0.40 to include Table 3) |
| Extend the demographics to all 1,798 users | ~$5 |
| Single-value demographics prompt | ~$0.50 (included in the extension if both are chosen) |
| Bot-sensitivity check | Free |
