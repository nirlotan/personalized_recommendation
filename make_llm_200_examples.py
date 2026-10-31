"""
Worked examples of the 200-candidate LLM ranking (run_llm_200.py), with the model's explanations.

Picks six users by rule from results/llm/candidates_200_per_user.csv.gz, all following at least
MIN_RELEVANT of the 200 candidates and at most one per category:

    good - the largest AP lift over a random ranking, among answers ranking at least 95% of the candidates
    bad  - the lowest AP (any completeness)

and asks GPT-6 Luna for the same ranking again (same names, same shuffled candidates, same prompt and
reasoning effort), now also asking what it inferred about the user and why it ranked its top 10 high.
The explanations are the model's own account, written next to the ranking; the hidden reasoning
tokens are not available.

    python make_llm_200_examples.py

Writes results/LLM_200_EXAMPLES.html. Responses are cached in results/llm/cache/candidates_200_explain.jsonl.
"""

import html
import json

import pandas as pd

from reproduction.config import CATEGORY_LABELS, RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.embeddings import load_metadata
from reproduction.llm import ResponseCache, SpendingCap, candidates_json, load_openai_key, make_dspy_rankers, names_of
from reproduction.page_style import CSS, JS
from run_llm_200 import BAND, COMPLETE_LIST, PROMPT_VERSION, average_precision, followed_sample, model_ranking
from run_llm_experiments import band_sample, band_setup

OUT = RESULTS_DIR / "llm"
MIN_RELEVANT = 10
PER_GROUP = 3
TOP = 10
esc = html.escape


def pick(res):
    res = res[res["n_relevant"] >= MIN_RELEVANT].assign(lift=res["llm_k200"] - res["random"],
                                                       frac=res["returned_k200"] / res["n_candidates"])
    good = res[res["frac"] >= 0.95].sort_values("lift", ascending=False).drop_duplicates("topic").head(PER_GROUP)
    bad = res.sort_values("llm_k200").drop_duplicates("topic").head(PER_GROUP)
    return good, bad


def make_explainer(lm):
    import dspy

    class ExplainedRecommendation(dspy.Signature):
        """
        given the below Twitter user, to which we provide a sampled list of accounts the user already follows, and a list of possible accounts to recommend for the user to follow, create a prioritized recommendation list.
        You must rank the entire list provided to you.
        Try to deduce what you think the user might like based on the list of accounts they already follow.
        Return ONLY a valid list object with the ranked account twitter_user_id sorted by relevance (highest to lowest). Do not include name or reasoning.
        Example format: [71026122, 141341662, 567562934, ...]
        """
        followed_accounts = dspy.InputField(desc="Sampled list of other accounts the user already follows")
        potential_accounts = dspy.InputField(desc="JSON list of potential accounts to recommend with twitter_user_id and names")
        user_interests = dspy.OutputField(desc="Two or three sentences: what the followed accounts suggest about the user's interests")
        ranked_ids = dspy.OutputField(desc="list with the account twitter_user_id sorted by priority as a ranked key")
        top_reasons = dspy.OutputField(desc=f"JSON list of {TOP} objects, one per account you ranked highest, in your order: "
                                            '{"twitter_user_id": ..., "reason": "one sentence on why it is ranked this high, '
                                            'naming the followed accounts that support it"}')

    signature = ExplainedRecommendation.with_instructions(
        ExplainedRecommendation.instructions + "\n" + COMPLETE_LIST
        + f"\nIn addition to the list, describe the user's interests and explain why you ranked your top {TOP} accounts highest.")
    predict = dspy.Predict(signature)

    def explain(followed, candidates):
        with dspy.context(lm=lm):
            p = predict(followed_accounts=followed, potential_accounts=candidates)
        return json.dumps({"user_interests": p.user_interests, "ranked_ids": str(p.ranked_ids), "top_reasons": str(p.top_reasons)})

    return explain


def parse_reasons(text):
    """The reasons as {id: reason}; text around the JSON list (e.g. dspy's completion marker) is ignored."""
    text = str(text).split("[[ ##")[0]
    try:
        items = json.loads(text[text.find("["):text.rfind("]") + 1])
    except ValueError:
        return {}
    return {int(i["twitter_user_id"]): i.get("reason", "") for i in items if isinstance(i, dict) and str(i.get("twitter_user_id", "")).isdigit()}


EXTRA_CSS = """
.example h4{margin:0 0 4px}.example .meta{color:var(--muted);font-size:13px;margin:0 0 12px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}.chip{background:var(--accent-wash);border-radius:999px;padding:3px 10px;font-size:13px}
.chip b{font-variant-numeric:tabular-nums}
.names{font-size:13px;color:var(--ink-2);line-height:1.5}
.quote{margin:0 0 12px;padding:10px 14px;border-left:3px solid var(--s1);background:var(--accent-wash);border-radius:0 10px 10px 0}
table.data td.reason{white-space:normal;text-align:left;min-width:260px;font-size:13px}
table.data td.acct{white-space:normal;text-align:left}
.hit{color:var(--s3);font-weight:650}.miss{color:var(--muted)}
.positions{font-size:13px;color:var(--ink-2);font-variant-numeric:tabular-nums}
"""


def mark_followed(reason, followed_names):
    """The reason with every followed account name it mentions in bold; also how many it mentions."""
    found = sorted({n for n in followed_names if len(n) >= 4 and n in reason}, key=len, reverse=True)
    text = esc(reason)
    for n in found:
        text = text.replace(esc(n), f"<b>{esc(n)}</b>")
    return text, len(found)


def example_html(kind, i, ex):
    r = ex["row"]
    rel = set(ex["relevant"])
    cname = ex["cand_names"]
    rerun = ex["rerun_ranking"]
    reasons = ex["reasons"]
    rows = "".join(
        f"<tr><td>{k}</td><td class='acct'>{esc(cname.get(a, str(a)))}</td>"
        f"<td>{'<span class=hit>✓ follows</span>' if a in rel else '<span class=miss>–</span>'}</td>"
        f"<td class='reason'>{mark_followed(reasons.get(a, ''), ex['followed_names'])[0]}</td></tr>"
        for k, a in enumerate(rerun[:TOP], 1))
    orig = ex["orig_ranking"]
    orig_top = ", ".join(f"<span class='{'hit' if a in rel else ''}'>{esc(cname.get(a, str(a)))}</span>" for a in orig[:TOP])
    pos = sorted(orig.index(a) + 1 for a in rel if a in orig)
    missing = len(rel) - len(pos)
    rerun_pos = sorted(rerun.index(a) + 1 for a in rel if a in rerun)
    shown = ", ".join(esc(n) for n in ex["followed_names"][:40])
    relevant_names = ", ".join(esc(cname.get(a, str(a))) for a in ex["relevant_sorted"])
    return f"""
    <div class="example card">
      <h4>{kind} {i} · user row {r['user']} · {esc(CATEGORY_LABELS[r['topic']])}</h4>
      <p class="meta">Follows {r['n_relevant']} of the {r['n_candidates']} candidates. {r['n_followed_listed']} followed account names in the prompt.</p>
      <div class="chips">
        <span class="chip">LLM AP <b>{r['llm_k200']:.3f}</b></span>
        <span class="chip">SocialVec, same accounts <b>{r['socialvec_k200']:.3f}</b></span>
        <span class="chip">LLM, no user information <b>{r['llm_baseline']:.3f}</b></span>
        <span class="chip">Random (expected) <b>{r['random']:.3f}</b></span>
        <span class="chip">Candidates ranked <b>{r['returned_k200']} / {r['n_candidates']}</b></span>
        <span class="chip">Repeat, same prompt <b>{ex['repeat_ap']:.3f}</b> · ranked {ex['repeat_n']}</span>
        <span class="chip">Rerun with explanations <b>{ex['rerun_ap']:.3f}</b> · ranked {len(rerun)}</span>
      </div>
      <p class="step"><b>Followed accounts given to the model</b> (first 40 of {r['n_followed_listed']}, in prompt order)</p>
      <p class="names">{shown}{' …' if r['n_followed_listed'] > 40 else ''}</p>
      <p class="step"><b>Candidates the user follows</b> (the relevant ones, in category-list order)</p>
      <p class="names">{relevant_names}</p>
      <p class="step"><b>What the model inferred</b> (rerun)</p>
      <p class="quote">{esc(ex['interests'])}</p>
      <p class="step"><b>Rerun: top {TOP} and the model's reasons</b> (followed accounts that appear in the prompt are in bold)</p>
      <div class="scroll"><table class="data compact"><thead><tr><th>#</th><th>Account</th><th>User follows it?</th><th>Model's reason</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
      <p class="positions">Rerun: relevant candidates at ranks {', '.join(map(str, rerun_pos)) or '–'}; {len(rel) - len(rerun_pos)} left out.</p>
      <p class="step"><b>Original run (scored in Section 7): top {TOP}</b> (followed ones in green)</p>
      <p class="names">{orig_top}</p>
      <p class="positions">Original run: relevant candidates at ranks {', '.join(map(str, pos)) or '–'}; {missing} left out (counted as misses).</p>
    </div>"""


def main():
    res = pd.read_csv(OUT / "candidates_200_per_user.csv.gz")
    good, bad = pick(res)

    categories = load_categories()
    users = load_users(categories)
    sample, _, _, _ = band_sample(users, categories)
    candidates, exclude, _ = band_setup(users, categories, BAND)
    metadata = load_metadata()
    names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))

    import dspy
    lm = dspy.LM("openai/gpt-6-luna", api_key=load_openai_key(), reasoning_effort="medium", cache=False,
                 num_retries=8, max_tokens=32000)
    cap = SpendingCap(lm, 0.5)
    explain = cap.guard(make_explainer(lm))
    rank_personal = cap.guard(make_dspy_rankers(lm=lm, extra_instructions=COMPLETE_LIST)[0])
    cache = ResponseCache(OUT / "cache" / "candidates_200_explain.jsonl")

    examples = {}
    for kind, picked in (("Good", good), ("Bad", bad)):
        examples[kind] = []
        for _, r in picked.iterrows():
            row, topic = int(r["user"]), r["topic"]
            user = sample.loc[row]
            order = json.loads(r["order"])
            followed = followed_sample(user, exclude[topic], names)
            relevant = [int(f) for f in user["friends_ids"] if int(f) in set(order)]
            assert len(relevant) == r["n_relevant"] and len(followed) == r["n_followed_listed"]
            followed_text, cand_text = names_of(followed, names), candidates_json(candidates[topic], order=order)
            out = json.loads(cache.call(f"{PROMPT_VERSION}|explain|{row}", lambda: explain(followed_text, cand_text)))
            # the original prompt once more, without explanations: how much does a plain repeat vary?
            repeat = model_ranking(cache.call(f"{PROMPT_VERSION}|repeat|{row}", lambda: rank_personal(followed_text, cand_text)), order)
            rerun = model_ranking(out["ranked_ids"], order)
            cand = candidates[topic]
            cand_names = dict(zip(cand["twitter_user_id"].astype(int), cand["twitter_name"]))
            examples[kind].append({
                "row": r, "relevant": relevant, "followed_names": [names[str(a)] for a in followed],
                "relevant_sorted": [int(a) for a in cand["twitter_user_id"] if int(a) in set(relevant)],
                "cand_names": cand_names, "interests": out["user_interests"], "reasons": parse_reasons(out["top_reasons"]),
                "rerun_ranking": rerun, "rerun_ap": average_precision(rerun, relevant),
                "repeat_ap": average_precision(repeat, relevant), "repeat_n": len(repeat),
                "orig_ranking": model_ranking(r["resp_k200"], order),
            })
    cap.update()
    print(f"{cap.calls} model calls, ${cap.spent:.4f}")
    if cap.calls:
        info_path = OUT / "run_info.json"
        runs = json.loads(info_path.read_text())
        runs.append({"model": "gpt-6-luna", "reasoning_effort": "medium", "parts": ["candidates-200-explain"],
                     "calls": cap.calls, "cost_usd": round(cap.spent, 4), "budget_usd": 0.5})
        info_path.write_text(json.dumps(runs, indent=2))

    everyone = [e for v in examples.values() for e in v]
    reasons_all = [(r, e["followed_names"]) for e in everyone for r in e["reasons"].values()]
    grounded = sum(mark_followed(r, f)[1] > 0 for r, f in reasons_all)
    repeat_diff = sum(abs(e["row"]["llm_k200"] - e["repeat_ap"]) for e in everyone) / len(everyone)
    g, b = examples["Good"], examples["Bad"]
    rerun_hits = {k: sum(len(set(e["rerun_ranking"][:TOP]) & set(e["relevant"])) for e in v) for k, v in examples.items()}
    orig_hits = {k: sum(len(set(e["orig_ranking"][:TOP]) & set(e["relevant"])) for e in v) for k, v in examples.items()}
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>200-Candidate LLM Examples</title><style>{CSS}{EXTRA_CSS}</style></head>
<body><div id="tip" role="tooltip"></div>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">Extension · Social Knowledge for Cross-Domain User Preference Modeling</p>
  <h1>Ranking 200 candidates: worked examples</h1>
  <p class="lede">Section 7 of <a href="results.html#all200">the results page</a> reports that GPT-6 Luna, given up to 200 followed
  account names, ranks 200 candidates better than SocialVec. These six users show what that looks like: three where it went well and
  three where it went badly, each rerun with the model asked to explain its choices.</p>
</header>

<section>
  <h2><span class="num">1</span>How the examples were chosen</h2>
  <ul>
    <li><i>Eligible users:</i> those following at least {MIN_RELEVANT} of the 200 candidates, so that a high AP cannot come from one lucky
    guess; at most one user per category.</li>
    <li><i>Good:</i> the largest AP above a random ranking, among answers that ranked at least 95% of the candidates.</li>
    <li><i>Bad:</i> the lowest AP, whatever the answer's completeness.</li>
    <li><i>Rerun:</i> the same prompt, followed accounts, shuffled candidates and reasoning effort (medium), with three output fields
    instead of one: the user's interests, the ranked list, and a one-sentence reason for each of the top {TOP}. The chips show the AP of
    the original run (the one in Section 7) and of the rerun, which is a new call and can rank differently.</li>
    <li><i>Caveat:</i> the reasons are written by the model alongside its answer. They are its account of the ranking, not a record of
    the hidden reasoning, which the API does not return.</li>
    <li><i>Privacy:</i> users are identified by internal row only. Followed and candidate accounts are public accounts in SocialVec.</li>
  </ul>
  <p>Across the three good examples, {orig_hits['Good']} of the 30 top-{TOP} places in the original runs went to accounts the user
  follows ({rerun_hits['Good']} in the reruns); across the bad ones, {orig_hits['Bad']} ({rerun_hits['Bad']} in the reruns).</p>
</section>

<section>
  <h2><span class="num">2</span>Where it went well</h2>
  {''.join(example_html('Good example', i, e) for i, e in enumerate(examples['Good'], 1))}
</section>

<section>
  <h2><span class="num">3</span>Where it went badly</h2>
  {''.join(example_html('Bad example', i, e) for i, e in enumerate(examples['Bad'], 1))}
</section>

<section>
  <h2><span class="num">4</span>What the examples show</h2>
  <ul class="highlights">
    <li><b>When it works, the ranking follows a clear reading of the user.</b> The model names an interest that the followed accounts
    support, such as political news (good example 1), esports and NFL (good example 2) or politically outspoken entertainers (good
    example 3), and puts the matching candidates first. {grounded} of the {len(reasons_all)} reasons cite at least one account that is
    actually in the prompt (in bold).</li>
    <li><b>The failures have different causes.</b>
    <ul>
      <li><i>A plausible but wrong reading</i> (bad example 1): following the University of Alabama leads the model to rank Alabama and
      college football first, while the user follows mostly professional teams. SocialVec scores {b[0]['row']['socialvec_k200']:.3f} here.</li>
      <li><i>A cut-off answer</i> (bad example 2): the original answer ranked only {b[1]['row']['returned_k200']} of
      {b[1]['row']['n_candidates']} candidates. Its top picks are reasonable, but the relevant candidates it never listed count as misses.</li>
      <li><i>Interests outside the category</i> (bad example 3): the model favours actors known for activism, matching the user's
      political follows, over the mainstream actors the user follows. SocialVec does even worse here ({b[2]['row']['socialvec_k200']:.3f}).</li>
    </ul></li>
    <li class="warn"><b>Single users vary a lot between runs.</b> Repeating the identical prompt changed AP by {repeat_diff:.3f} on average
    across these six users (from {g[2]['row']['llm_k200']:.3f} to {g[2]['repeat_ap']:.3f} for good example 3). The examples were chosen for
    extreme AP, so some move back toward the middle on a repeat. Section 7's comparison averages over 270 users and is the better guide
    to how well the method works; these examples show how it works.</li>
  </ul>
</section>

<footer><p>Generated by <code>python make_llm_200_examples.py</code> from <code>results/llm/candidates_200_per_user.csv.gz</code>.</p></footer>
</div>
<script>{JS}</script>
</body></html>
"""
    (RESULTS_DIR / "LLM_200_EXAMPLES.html").write_text(page, encoding="utf-8")
    print("wrote", RESULTS_DIR / "LLM_200_EXAMPLES.html")
    for kind, v in examples.items():
        for e in v:
            print(kind, e["row"]["user"], e["row"]["topic"], f"orig {e['row']['llm_k200']:.3f} repeat {e['repeat_ap']:.3f}",
                  f"rerun {e['rerun_ap']:.3f}",
                  f"ranked {len(e['rerun_ranking'])}", f"reasons {len(e['reasons'])}")


if __name__ == "__main__":
    main()
