"""
Build results/LLM_DEMOGRAPHICS_REPORT.html: the demographics-only LLM ranking experiment
(run_llm_demographics.py --describe-model gpt-4o) compared with every other method on the same users.

GPT-4o writes the demographic descriptions and GPT-6 Luna ranks from them. The report covers the
ranks 81-100 candidates, first in follower-count order and then shuffled (run_llm_demographics_bands.py).

    python make_demographics_report.py

Reads the generated descriptions from data/llm_demographics/ (not tracked by git). The page
identifies users by internal row only, never by username. Also writes results/llm/demographics_summary.json
for the summaries in results.html and results_backup.html.
"""

import html
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from reproduction.config import CATEGORY_LABELS, DATA_DIR, PAPER_CATEGORY_ORDER, RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.demographics import classify_users
from reproduction.embeddings import SocialVec, load_metadata
from reproduction.llm import names_of, ranked_ids, user_samples
from reproduction.page_style import CSS, JS
from run_llm_experiments import band_sample

LLM = RESULTS_DIR / "llm"
PRIVATE = DATA_DIR / "llm_demographics"

# ---------------------------------------------------------------- data
band = pd.read_csv(LLM / "band_per_user.csv.gz").set_index("user")
demo = pd.read_csv(LLM / "demographics_per_user_gpt-4o.csv.gz").set_index("user")
desc = pd.read_csv(PRIVATE / "descriptions_gpt-4o.csv").set_index("row")
m = band.join(demo[["llm_demographics", "n_accounts_described"]], how="inner").join(desc[["description", "ranking_response"]])
# the no-information baseline always uses shuffled candidates (one random order per user, run_llm_demographics_bands.py);
# with candidates in follower-count order it partly copies the presented order
_shuffled = pd.read_csv(LLM / "demographics_bands_per_user.csv.gz")
_shuffled = _shuffled[_shuffled["band"] == "81-100"].set_index("user")
m["llm_baseline_sorted"] = m["llm_baseline"]
m["llm_baseline"] = _shuffled["ap_baseline"].reindex(m.index)
runs = json.loads((LLM / "run_info.json").read_text())
run_4o = [r for r in runs if r.get("describe_model") == "gpt-4o"][-1]

categories = load_categories()
users = load_users(categories)
sample, candidates, exclude, _ = band_sample(users, categories)
metadata = load_metadata()
names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))
prompt_ids = user_samples(sample, categories, names, exclude=exclude)["llm_k50"]
cand_name = {int(i): n for c in candidates.values() for i, n in zip(c["twitter_user_id"], c["twitter_name"])}

# ---------------------------------------------------------------- description content and parsed attributes
ATTR = {
    "gender": r"\b(?:woman|man|female|male|women|men)\b",
    "age": r"(?:\b\d0s\b|\bteen|\byoung\b|middle[- ]aged|\bolder\b|\bsenior|\badult\b|\b\d{2}\s*(?:-|to|–)\s*\d{2}\b)",
    "ethnicity": r"\b(?:white|black|caucasian|african[- ]american|hispanic|latin[oax]\w*|asian)\b",
    "education": r"(?:college|university|degree|high[- ]school|higher[- ]education|graduate|bachelor|primary school|educated)",
}
# Topics or interests named in a description, which the prompt forbids (it leaks more than demographics)
TOPICS = (r"\b(?:politic\w*|sport\w*|music\w*|entertainment|news|pop culture|conservative|liberal|progressive|republican|"
          r"democrat\w*|business|military|gaming|comedy|celebrit\w*|interest\w*|fan|fans|enthusiast\w*|hip[- ]hop|media)\b")

n_attr = sum(m["description"].str.contains(p, case=False, regex=True).astype(int) for p in ATTR.values())
m["status"] = np.select([n_attr == 4, n_attr == 0], ["complete", "declined"], "partial")
m["leak"] = m["description"].str.contains(TOPICS, case=False, regex=True)


def parse_gender(t):
    if re.search(r"\b(woman|female|women)\b", t, re.I):
        return "Female"
    if re.search(r"\b(man|male|men)\b", t, re.I):
        return "Male"


def parse_ethnicity(t):
    if re.search(r"\b(white|caucasian)\b", t, re.I):
        return "Caucasian"
    if re.search(r"\b(black|african[- ]american)\b", t, re.I):
        return "AfrAmerican"
    if re.search(r"\b(hispanic|latin\w*|asian)\b", t, re.I):
        return "Other"


def parse_education(t):
    if re.search(r"(college|university|degree|higher[- ]education|graduate|bachelor)", t, re.I):
        return "Degree"
    if re.search(r"high[- ]school", t, re.I):
        return "HighSchool"


def parse_age(t):
    if re.search(r"(\bteen|early 20s|under 25|college[- ]age|\b1[89]\s*(?:-|to|–)\s*2[0-4]\b)", t, re.I):
        return "Under25"
    if re.search(r"(\b[3-7]0s\b|middle[- ]aged|\bolder\b|\bsenior|late 20s|mid[- ]20s)", t, re.I):
        return "Over25"


texts = m.loc[m["status"] != "declined", "description"]
parsed = pd.DataFrame({"gender": texts.map(parse_gender), "ethnicity": texts.map(parse_ethnicity),
                       "education": texts.map(parse_education), "age": texts.map(parse_age)})
traits = classify_users(sample.loc[m.index], SocialVec("2020c"))

# Descriptive statistics of the stated attributes (results.html shows these counts, never the descriptions).
# Each description gets the first value it states for an attribute; hedges such as "30s or 40s" also count as giving more than one.
_RANGE = r"\s*(?:-|to|–)\s*\d{2}"
ATTRIBUTE_VALUES = {
    "Gender": [("Female", r"\b(?:female|woman|women)\b"), ("Male", r"\b(?:male|man|men)\b")],
    "Age": [("Under 30", rf"\b(?:teen\w*|young adults?|20s|twenties|[12]\d{_RANGE})\b"),
            ("30s", rf"\b(?:30s|thirties|3\d{_RANGE})\b"),
            ("40s", rf"\b(?:40s|forties|4\d{_RANGE})\b"),
            ("Middle-aged", r"\bmiddle[- ]aged\b"),
            ("50 or older", rf"\b(?:[5-9]0s|older|seniors?|[5-9]\d{_RANGE})\b")],
    "Ethnicity": [("Caucasian / White", r"\b(?:caucasian|white)\b"), ("African American / Black", r"\b(?:african[- ]american|black)\b"),
                  ("Hispanic / Latino", r"\b(?:hispanic|latin[oax]\w*)\b"), ("Asian", r"\basian\b"),
                  ("Mixed / diverse", r"\b(?:mixed|diverse|multi[- ]?ethnic|multiracial)\b")],
    "Education": [("High school", r"\bhigh[- ]school\b"),
                  ("Some higher education", r"\b(?:some college|some higher education|at least some (?:higher education|college))\b"),
                  ("College / university degree", r"\b(?:bachelor\w*|college degree|university degree|undergraduate)\b"),
                  ("Graduate degree", r"\b(?:graduate degree|postgraduate|master\w*|advanced degree|doctora\w*|phd)\b"),
                  ("Higher education, level not stated", r"\b(?:higher education|college|university|degree)\b")],
}
UNSPECIFIC = "Higher education, level not stated"  # used only when no specific level is stated


def stated_values(text, attr):
    """The values a description states for an attribute, in order of first mention."""
    hits = sorted((mt.start(), value) for value, pat in ATTRIBUTE_VALUES[attr] if (mt := re.search(pat, text, re.I)))
    values = [v for _, v in hits]
    specific = [v for v in values if v != UNSPECIFIC]
    return specific or values


attributes = pd.DataFrame({attr: m["description"].map(lambda t, a=attr: (stated_values(t, a) or ["Not stated"])[0])
                           for attr in ATTRIBUTE_VALUES}, index=m.index)
attributes_multiple = {attr: int(m["description"].map(lambda t, a=attr: len(stated_values(t, a)) > 1).sum()) for attr in ATTRIBUTE_VALUES}

# ---------------------------------------------------------------- numbers, candidates in follower-count order
METHODS = [
    ("random", "Random ranking (expected)", "Non-personalized"),
    ("popularity", "Popularity (follower count)", "Non-personalized"),
    ("llm_baseline", "LLM, no user information (shuffled)", "Non-personalized"),
    ("llm_demographics", "LLM, demographics by GPT-4o", "This experiment"),
    ("socialvec_k12", "SocialVec, 12 accounts", "Same 12 accounts"),
    ("llm_k12", "LLM, 12 account names", "Same 12 accounts"),
    ("socialvec_k25", "SocialVec, 25 accounts", "Same 25 accounts"),
    ("llm_k25", "LLM, 25 account names", "Same 25 accounts"),
    ("llm_k50", "LLM, 50 accounts", "Full or large profile"),
    ("socialvec", "SocialVec, all followed entities", "Full or large profile"),
]
cols = [c for c, _, _ in METHODS]
by_cat = m.groupby("topic")[cols].mean().reindex([c for c in PAPER_CATEGORY_ORDER if c in set(m["topic"])])
overall = by_cat.mean()
sorted_baseline = m.groupby("topic")["llm_baseline_sorted"].mean().mean()  # for the text only
by_cat.loc["Overall"] = overall


def pv(a, b):
    d = np.asarray(a) - np.asarray(b)
    return wilcoxon(d).pvalue if (d != 0).any() else np.nan


def ptxt(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p):
    """A p-value for running text: '< 0.001' or '= 0.006'."""
    return ptxt(p) if p < 0.001 else f"= {p:.3f}"


tests = [(label, overall["llm_demographics"] - overall[c], pv(m["llm_demographics"], m[c]))
         for c, label, _ in METHODS if c not in ("llm_demographics", "random")]

leak, clean = m[m["leak"]], m[~m["leak"]]
subset_rows = [(label, len(g), g["llm_demographics"].mean(), g["llm_baseline"].mean(), pv(g["llm_demographics"], g["llm_baseline"]))
               for label, g in [("All users", m), ("Description also names interests", leak), ("Description is demographics only", clean)]]
_, leak_n, leak_d, leak_b, leak_p = subset_rows[1]
_, clean_n, clean_d, clean_b, clean_p = subset_rows[2]


def agreement_with_socialvec(parsed_attrs):
    """Per attribute: users with a parsed binary guess, agreement with SocialVec, majority-class rate, guess counts."""
    rows = []
    for attr, label in [("gender", "Gender (female / male)"), ("ethnicity", "Ethnicity (Caucasian / African American)"),
                        ("education", "Education (degree / high school)"), ("age", "Age (over / under 25)")]:
        both = parsed_attrs[attr].isin(["Female", "Male", "Caucasian", "AfrAmerican", "Degree", "HighSchool", "Over25", "Under25"])
        a, b = parsed_attrs.loc[both, attr], traits.loc[both[both].index, attr]
        majority = b.value_counts(normalize=True).max() if len(b) else np.nan
        rows.append((label, int(both.sum()), (a == b).mean() if len(b) else np.nan, majority, a.value_counts().to_dict()))
    return rows


agreement = agreement_with_socialvec(parsed)
above = [(label.split(" (")[0].lower(), a, maj) for label, _, a, maj, _ in agreement if a - maj >= 0.05]
dominant = {label.split(" (")[0].lower(): max(dist.items(), key=lambda kv: kv[1]) for label, _, _, _, dist in agreement}
agreement_text = (
    "GPT-4o's guesses lean heavily toward the most common class: "
    + ", ".join(f"{dominant[n][1]} of its {n} guesses are {dominant[n][0]}" for n in ("education", "age", "ethnicity"))
    + ". Agreement with SocialVec is therefore close to the majority-class rate, except for "
    + " and ".join(f"{n} ({a:.0%} vs. {maj:.0%})" for n, a, maj in above) + "."
) if above else ""
cost = run_4o["cost_by_model"]

# ---------------------------------------------------------------- shuffled-order rankings (run_llm_demographics_bands.py)
sh_all = pd.read_csv(LLM / "demographics_bands_per_user.csv.gz")
sh_all["leak"] = sh_all["user"].map(m["leak"]).astype(bool)
SH_METHODS = [("random", "Random ranking (expected)"), ("popularity", "Popularity (follower count)"),
              ("ap_baseline", "LLM, no user information"), ("ap_gpt-4o", "LLM, demographics by GPT-4o"),
              ("ap_k12", "LLM, 12 followed account names"), ("ap_k25", "LLM, 25 followed account names"),
              ("ap_k50", "LLM, 50 followed account names"), ("socialvec", "SocialVec, all followed entities")]
SH_COLORS = {"ap_gpt-4o": "s2", "ap_k12": "s3", "ap_k25": "s3", "ap_k50": "s3"}  # highlighted LLM rankings
SH_BANDS = ["1-20", "81-100", "181-200"]  # the report shows 81-100; 1-20 feeds results.html, 181-200 (extended lists) results_backup.html
sh_cat = {b: sh_all[sh_all["band"] == b].groupby("topic")[[c for c, _ in SH_METHODS]].mean() for b in SH_BANDS}
sh_overall = {b: t.mean() for b, t in sh_cat.items()}  # unweighted mean over categories
sh_n = sh_all.groupby("band").size().to_dict()
sh_tests = {b: {name: pv(sh_all.loc[sh_all["band"] == b, x], sh_all.loc[sh_all["band"] == b, y]) for name, x, y in [
    ("4o_vs_baseline", "ap_gpt-4o", "ap_baseline"), ("baseline_vs_popularity", "ap_baseline", "popularity"),
    ("baseline_vs_random", "ap_baseline", "random"), ("4o_vs_socialvec", "ap_gpt-4o", "socialvec"),
    ("k12_vs_baseline", "ap_k12", "ap_baseline"), ("k12_vs_4o", "ap_k12", "ap_gpt-4o"), ("k25_vs_k12", "ap_k25", "ap_k12"),
    ("k50_vs_k12", "ap_k50", "ap_k12"), ("k50_vs_socialvec", "ap_k50", "socialvec")]} for b in SH_BANDS}
sh_groups = {b: [(label, len(g), g["ap_gpt-4o"].mean(), g["ap_baseline"].mean(), pv(g["ap_gpt-4o"], g["ap_baseline"]))
                 for label, g in [("Demographics only", sh_all[(sh_all["band"] == b) & ~sh_all["leak"]]),
                                  ("Also names interests", sh_all[(sh_all["band"] == b) & sh_all["leak"]])]] for b in SH_BANDS}
sh = sh_all[sh_all["band"] == "81-100"]
so, st = sh_overall["81-100"], sh_tests["81-100"]
order_effect = {k: (sh[a].mean(), sh[b].mean(), pv(sh[a], sh[b])) for k, a, b in [
    ("baseline", "ap_baseline", "ordered_llm_baseline"), ("4o", "ap_gpt-4o", "ordered_llm_demographics_4o"),
    ("k12", "ap_k12", "ordered_llm_k12"), ("k25", "ap_k25", "ordered_llm_k25"), ("k50", "ap_k50", "ordered_llm_k50")]}
# SocialVec on the same 12 / 25 account names (Table 9 of results.html); it does not depend on the candidate order
names_vs_sv = {k: (sh[f"ap_k{k}"].mean(), sh[f"ordered_socialvec_k{k}"].mean(), pv(sh[f"ap_k{k}"], sh[f"ordered_socialvec_k{k}"]))
               for k in (12, 25)}
sh_complete = float(sh[[f"returned_{k}" for k in ("baseline", "gpt-4o", "k12", "k25", "k50")]].eq(sh["n_candidates"], axis=0).mean().min())

# ---------------------------------------------------------------- examples, chosen by rule
multi = m[(m["status"] == "complete") & ~m["leak"] & (m["n_relevant"] >= 2)]
ex_high = multi.sort_values(["llm_demographics", "n_relevant"], ascending=False).index[0]
ex_low = multi.assign(gap=multi["llm_baseline"] - multi["llm_demographics"]).sort_values(["llm_demographics", "gap"], ascending=[True, False]).index[0]
ex_leak = (leak["llm_demographics"] - leak["llm_demographics"].median()).abs().sort_values().index[0]
examples = [(ex_high, "A high-AP case"), (ex_low, "A low-AP case"), (ex_leak, "A description that also names interests")]

# ---------------------------------------------------------------- html helpers
esc = html.escape


def f3(v):
    return "" if pd.isna(v) else f"{v:.3f}"


def signed(v):
    return f"{v:+.3f}".replace("-", "−")


def bar_chart(values, methods, colors, context, W=760, L=250, x1=0.55):
    """Horizontal bars of overall MAP per method; methods in `colors` are highlighted."""
    Rm, T, rowh = 60, 10, 30
    H = T + rowh * len(methods) + 30
    px = lambda v: L + v / x1 * (W - L - Rm)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Overall MAP by method, {context}">']
    ticks = np.arange(0, x1 + 1e-9, 0.1)
    for v in ticks:
        parts.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{T}" y2="{H-24}"/>'
                     f'<text class="tick" x="{px(v):.1f}" y="{H-8}" text-anchor="middle">{v:.1f}</text>')
    for i, (c, label) in enumerate(methods):
        y, v = T + i * rowh, values[c]
        w = px(v) - L
        weight = ' font-weight="650"' if c in colors else ""
        parts.append(f'<text class="cat" x="{L-10}" y="{y+17}" text-anchor="end"{weight}>{esc(label)}</text>'
                     f'<path class="bar {colors.get(c, "muted-bar")}" d="M{L},{y+6} h{w-4:.1f} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 h-{w-4:.1f} z"/>'
                     f'<text class="delta" x="{px(v)+8:.1f}" y="{y+17}"{weight}>{v:.3f}</text>'
                     f'<rect class="hit" x="{L}" y="{y+4}" width="{w:.1f}" height="18" data-tip="{esc(label)}, {context}<br>MAP <b>{v:.3f}</b>"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def category_table(table, methods, highlight, users=None):
    rows = []
    for cat, r in table.iterrows():
        cls = " class='total'" if cat == "Overall" else ""
        best = r[[c for c, _ in methods]].max()
        cells = "".join(f"<td class='{'hl' if c in highlight else ''}'>{'<b>' if r[c] == best else ''}{f3(r[c])}{'</b>' if r[c] == best else ''}</td>"
                        for c, _ in methods)
        n = f"<td>{users.get(cat, '')}</td>" if users is not None else ""
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th>{n}{cells}</tr>")
    head = ("<th>Users</th>" if users is not None else "") + "".join(f"<th>{esc(label)}</th>" for _, label in methods)
    return f"<div class='scroll'><table class='data compact'><thead><tr><th>Category</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def example_block(row, title):
    r = m.loc[row]
    topic = r["topic"]
    followed = names_of(prompt_ids[row], names)
    relevant = {f for f in sample.at[row, "friends_ids"] if f in set(candidates[topic]["twitter_user_id"])}
    order = ranked_ids(r["ranking_response"])
    missing = [i for i in candidates[topic]["twitter_user_id"] if int(i) not in set(order)]
    items = "".join(f"<li class='{'hit-item' if a in relevant else ''}'>{esc(cand_name.get(a, str(a)))}{' ✓' if a in relevant else ''}</li>" for a in order)
    found = sorted(order.index(a) + 1 for a in relevant if a in order)
    calc = " + ".join(f"{i + 1}/{p}" for i, p in enumerate(found))
    comp = [("Random (expected)", "random"), ("Popularity", "popularity"), ("LLM, no information", "llm_baseline"),
            ("LLM, demographics by GPT-4o", "llm_demographics"), ("LLM, 12 names", "llm_k12"),
            ("SocialVec, same 12", "socialvec_k12"), ("SocialVec, all entities", "socialvec")]
    comp_rows = "".join(f"<tr{' class=hl-row' if c == 'llm_demographics' else ''}><th>{n}</th><td>{f3(r[c])}</td></tr>" for n, c in comp)
    label = {"complete": "all four attributes guessed", "partial": "some attributes guessed", "declined": "no attribute guessed"}[r["status"]]
    if r["leak"]:
        label += "; also names interests"
    return f"""
    <div class="example card">
      <h4>{esc(title)} · user row {row} · {esc(CATEGORY_LABELS[topic])}</h4>
      <div class="ex-grid">
        <div>
          <p class="step"><span>1</span>Accounts given to GPT-4o ({len(prompt_ids[row])} names)</p>
          <p class="quote small">{esc(followed)}</p>
          <p class="step"><span>2</span>Description by GPT-4o ({label})</p>
          <p class="quote">{esc(r['description'])}</p>
          <p class="step"><span>3</span>GPT-6 Luna's ranking from the description only (✓ = followed by the user)</p>
          <ol class="ranking">{items}</ol>
          {f"<p class='caption'>Not returned by the model: {', '.join(esc(cand_name.get(int(i), str(i))) for i in missing)}.</p>" if missing else ""}
        </div>
        <div>
          <p class="step"><span>4</span>Score</p>
          <p class="ap">AP = ({calc}) / {len(found)} = <b>{r['llm_demographics']:.3f}</b></p>
          <table class="data compact mini"><tbody>{comp_rows}</tbody></table>
        </div>
      </div>
    </div>"""


EXTRA_CSS = """
table.data td.hl{background:var(--accent-wash)}
table.data tr.hl-row th,table.data tr.hl-row td{background:var(--accent-wash);font-weight:650}
.example h4{margin:0 0 12px;font-size:16px}
.ex-grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:24px}
@media (max-width:820px){.ex-grid{grid-template-columns:1fr}}
.step{font-size:13px;font-weight:600;color:var(--ink-2);margin:14px 0 6px;display:flex;gap:8px;align-items:center}
.step span{display:inline-flex;width:20px;height:20px;border-radius:50%;background:var(--chip);align-items:center;justify-content:center;font-size:12px}
.quote{margin:0;padding:10px 14px;border-left:3px solid var(--s1);background:var(--accent-wash);border-radius:0 10px 10px 0;max-width:none}
.quote.small{font-size:13.5px;color:var(--ink-2);border-left-color:var(--axis);background:transparent}
ol.ranking{columns:2;column-gap:24px;margin:0;padding-left:26px;font-size:14px}
ol.ranking li{padding:1px 0;break-inside:avoid}
ol.ranking li.hit-item{font-weight:650;color:var(--good)}
.ap{font-size:15px;margin:0 0 10px}
table.mini{width:auto;min-width:240px}
pre.prompt{white-space:pre-wrap;font-size:12.5px}
.method p{max-width:800px}
"""

# ---------------------------------------------------------------- page
o = overall
counts = m["status"].value_counts()
test_rows = "".join(f"<tr><th>{esc(label)}</th><td>{signed(diff)}</td><td>{ptxt(p)}</td></tr>" for label, diff, p in tests)
subset_table = "".join(f"<tr><th>{esc(label)}</th><td>{n}</td><td class='hl'>{f3(d)}</td><td>{f3(b)}</td><td>{ptxt(p)}</td></tr>"
                       for label, n, d, b, p in subset_rows)
agree_rows = "".join(f"<tr><th>{esc(label)}</th><td>{n}</td><td>{a:.0%}</td><td>{maj:.0%}</td><td class='muted'>{esc(', '.join(f'{k} {v}' for k, v in dist.items()))}</td></tr>"
                     for label, n, a, maj, dist in agreement)
group_rows = "".join(f"<tr><th>{esc(label)}</th><td>{n}</td><td class='hl'>{f3(d)}</td><td>{f3(b)}</td><td>{signed(d - b)}</td><td>{ptxt(p)}</td></tr>"
                     for label, n, d, b, p in sh_groups["81-100"])
SORTED_LABELS = [(c, label) for c, label, _ in METHODS]
g_clean = sh_groups["81-100"][0]

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demographic LLM Ranking</title><style>{CSS}{EXTRA_CSS}</style></head>
<body><div id="tip" role="tooltip"></div>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">Extension · Social Knowledge for Cross-Domain User Preference Modeling</p>
  <h1>Can demographics alone personalize an LLM's recommendations?</h1>
  <p class="lede">GPT-4o infers a user's gender, age, ethnicity and education from the names of accounts they follow, and
  GPT-6 Luna then ranks less popular entities given only that inferred description. I compare this with ranking from the
  account names themselves, with social embeddings, and with non-personalized baselines, on the same 270 users, first with
  the candidates in follower-count order (Section 2) and then shuffled (Section 4).</p>
  <div class="tiles">
    <div class="tile"><div class="label">Shuffled candidates: demographics vs. no information</div><div class="value">{so['ap_gpt-4o']:.3f} vs. {so['ap_baseline']:.3f}</div><div class="sub">MAP, p {p_eq(st['4o_vs_baseline'])}</div></div>
    <div class="tile"><div class="label">Shuffled, demographics-only descriptions</div><div class="value">{g_clean[2]:.3f} vs. {g_clean[3]:.3f}</div><div class="sub">{g_clean[1]} users whose description names no interests</div></div>
    <div class="tile"><div class="label">Descriptions that also name interests</div><div class="value">{leak_n} / {len(m)}</div><div class="sub">against the instruction to describe demographics only</div></div>
    <div class="tile"><div class="label">Shuffled: 12 account names</div><div class="value">{so['ap_k12']:.3f}</div><div class="sub">above the demographic description (p {p_eq(st['k12_vs_4o'])})</div></div>
  </div>
</header>

<section class="method">
  <h2><span class="num">1</span>Method</h2>
  <p><b>Task and data.</b> We use the less-popular-entity setting of the popularity-band extension. In each of nine categories
  (musical artists, news outlets, comedians, politicians, TV stations, actors, TV shows, sports teams and journalists),
  the candidates are the entities ranked 81–100 by Twitter follower count among the category's typed entities. Only
  entities with a SocialVec embedding are kept, leaving 18–20 per category. The users are the same stratified random sample used
  for the LLM experiments on this band: 30 users per category (270 in total), drawn with a fixed seed from the users
  sampled for that category who follow at least one candidate. The candidates a user follows are the relevant answers.</p>

  <p><b>Stage 1: demographic inference (GPT-4o).</b> For each user, GPT-4o (default sampling settings) receives the names of the
  accounts listed in that user's k = 50 prompt. These are 50 accounts drawn at random from everything the user follows, after
  removing every listed entity of the target category, keeping only accounts with a known display name
  ({m['n_accounts_described'].mean():.1f} names on average; minimum {int(m['n_accounts_described'].min())}). The model is asked for a
  one- or two-sentence description stating the user's most likely gender, age, ethnicity and level of education. It is
  instructed to always guess from stereotypes about the typical audience of the accounts, never to state that the attributes
  cannot be determined, and not to mention interests or accounts, so that the description carries demographic information
  only. The prompt, in the dspy signature format of the original experiments:</p>
  <pre class="prompt">Given the names of accounts that a Twitter user follows, write a short, concise description of the user that states their
most likely gender, age, ethnicity, and level of education (primary school, high school, higher education, etc.).
Deduce these attributes only from the names of the accounts they follow.
You must always guess the most likely value of all four attributes, based on stereotypes about the typical audience of
these accounts, even when the evidence is weak. Never answer that the attributes cannot be determined or that the
evidence is insufficient.
Describe only these four attributes: do not mention the user's interests or hobbies, and do not name any of the accounts.

Input  followed_accounts: Names of accounts the user follows
Output description: One or two sentences stating the user's most likely gender, age, ethnicity, and level of education</pre>

  <p><b>Stage 2: ranking from the description (GPT-6 Luna).</b> The ranking prompt is the authors' personalized ranking prompt,
  with the list of followed accounts replaced by the Stage 1 description. The candidates are given as a JSON list of ids and
  display names, and the model must return a full ranking of the ids. GPT-6 Luna runs at reasoning effort <i>low</i>, as in the
  other LLM experiments:</p>
  <pre class="prompt">given the below Twitter user, for whom we provide a short demographic description, and a list of possible accounts to
recommend for the user to follow, create a prioritized recommendation list.
You must rank the entire list provided to you.
Try to deduce what you think the user might like based on their demographic description.
Return ONLY a valid list object with the ranked account twitter_user_id sorted by relevance (highest to lowest). Do not
include name or reasoning.

Inputs user_description, potential_accounts (JSON list of twitter_user_id and names)
Output ranked_ids</pre>

  <p><b>Candidate order.</b> Section 2 lists the candidates sorted by follower count, as in the original experiments. Because an
  LLM partly copies the order it is given, Section 4 repeats Stage 2 with the candidates in one random order per user, shared by
  every condition of that user.</p>

  <p><b>Evaluation.</b> Each ranking is scored by average precision over the candidates the user follows, using the legacy
  scoring rule (relevant candidates missing from the response are skipped). MAP is the mean over users within a category,
  and overall MAP is the unweighted mean over the nine categories. All other methods are evaluated on the same users and
  candidates:
  <ul>
    <li><i>Non-personalized baselines:</i> the expected AP of a random ranking; follower-count popularity; and the LLM with no user information.</li>
    <li><i>LLM with account names:</i> the LLM given the names of 12, 25 or 50 followed accounts.</li>
    <li><i>SocialVec:</i> cosine similarity to the average embedding of the same 12 or 25 accounts, or of all the user's followed entities.</li>
  </ul>
  Differences are tested with two-sided paired Wilcoxon signed-rank tests on per-user AP.</p>

  <p><b>Description content.</b> A description is <i>complete</i> if it states a value for all four attributes, <i>declined</i> if
  it states none, and <i>partial</i> otherwise. Since the prompt forbids mentioning interests, descriptions that name topics or
  interests (for example politics, sports, music or news) are flagged as <i>naming interests</i>: they carry information beyond
  demographics. Stated values are also parsed into the binary classes of SocialVec's trait classifiers and compared with those
  classifiers' predictions from the user's full embedding.</p>
</section>

<section>
  <h2><span class="num">2</span>Results with candidates in follower-count order</h2>
  <p class="intro">GPT-4o's descriptions give {o['llm_demographics']:.3f} MAP, compared with {o['llm_baseline']:.3f} for the LLM
  without user information and {o['popularity']:.3f} for popularity. The gain is largest for descriptions that break the prompt by
  naming the user's interests (Section 3). Naming 12 followed accounts gives {o['llm_k12']:.3f}. In every table, the LLM without user
  information gets the candidates in a random order per user; with the candidates in follower-count order it scores
  {sorted_baseline:.3f}, because it partly copies the
  presented order (Section 4). The rankings that use user information barely depend on the order, so they are compared with the
  shuffled baseline.</p>
  <div class="card">{bar_chart(o, SORTED_LABELS, {"llm_demographics": "s2"}, "candidates in follower-count order")}
  <p class="caption">Overall MAP (mean over nine categories) on the same 270 users. The highlighted bar is this experiment.</p></div>

  <h3>Per category</h3>
  <div class="card">{category_table(by_cat, SORTED_LABELS, {"llm_demographics"})}
  <p class="caption">MAP per category; bold marks the best method in each row. With 30 users per category, per-category MAP has a
  standard error of about 0.05.</p></div>

  <h3>Statistical tests</h3>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Demographics by GPT-4o vs.</th><th>Δ MAP</th><th>p (Wilcoxon)</th></tr></thead>
  <tbody>{test_rows}</tbody></table></div>
  <p class="caption">Δ is the overall MAP of the demographic ranking minus that of the other method.</p></div>
</section>

<section>
  <h2><span class="num">3</span>What the descriptions contain</h2>
  <p class="intro">GPT-4o described every user: {counts.get('complete', 0)} descriptions state all four attributes,
  {counts.get('partial', 0)} only some (in wording the parser misses, such as "twenties" or "diverse ethnicity") and
  {counts.get('declined', 0)} none. But {leak_n} of the {len(m)} ({leak_n / len(m):.0%}) also name the user's interests, despite the
  instruction, for example "…with a higher education level, likely due to the interest in diverse media and sports analysts,
  personalities, and NFL-related accounts".</p>

  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Users</th><th>n</th><th>LLM, demographics by GPT-4o</th>
  <th>LLM, no information</th><th>p</th></tr></thead><tbody>{subset_table}</tbody></table></div>
  <p class="caption">MAP over the users in each row, pooled across categories. The demographic ranking lists the candidates by
  follower count; the baseline gets them shuffled.</p></div>

  <p><b>Leaked interests give most of the gain, but demographics alone also help.</b> For the {leak_n} users whose description names
  interests, the ranking reaches {leak_d:.3f}, far above the no-information baseline ({leak_b:.3f}, p {p_eq(leak_p)}). For the remaining
  {clean_n} users, whose descriptions are demographics only, the score is {clean_d:.3f} against {clean_b:.3f} (p {p_eq(clean_p)}): a smaller
  but significant gain. Against the baseline with sorted candidates, the demographics-only descriptions had looked no better
  ({clean_d:.3f} vs. {clean['llm_baseline_sorted'].mean():.3f}, p {p_eq(pv(clean['llm_demographics'], clean['llm_baseline_sorted']))}).</p>

  <h3>Do the guesses agree with SocialVec's trait classifiers?</h3>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Attribute</th><th>Users with a parsed guess</th>
  <th>Agreement with SocialVec</th><th>Majority-class rate</th><th>GPT-4o guesses</th></tr></thead><tbody>{agree_rows}</tbody></table></div>
  <p class="caption">The stated value is mapped to the classifier's binary classes (descriptions stating another value, or an
  ambiguous one such as "20s" for age, are left out). The majority-class rate is the agreement a constant guess of SocialVec's
  most frequent class would achieve. Treat this as a rough check: the parsing is rule-based, and SocialVec's classifiers are
  themselves only estimates.</p></div>
  <p>{agreement_text}</p>
</section>

<section>
  <h2><span class="num">4</span>Shuffled candidates</h2>
  <p class="intro">With the candidates sorted by follower count, an LLM that partly follows the order it is given gets a
  popularity ranking for free, which can hide what a description adds. Here every user's candidates are put in one random order,
  used for all of that user's rankings: with no user information, from GPT-4o's description, and from the names of 12, 25 or 50
  followed accounts (the prompts of Table 9 in <code>results/results.html</code>). Differences between them therefore come from the
  user information alone. All {sh_n['81-100']} users are included.</p>

  <div class="card">{bar_chart(so, SH_METHODS, SH_COLORS, "shuffled candidates", x1=0.6)}
  <p class="caption">Overall MAP (mean over categories) with the candidates in random order. Highlighted bars: LLM rankings from
  GPT-4o's demographic description (orange) and from the names of followed accounts (green).</p></div>

  <p><b>Findings.</b>
  <ul>
    <li><b>Shuffling removes the order advantage of the no-information ranking.</b> It drops from {order_effect['baseline'][1]:.3f}
    (popularity-sorted candidates) to {order_effect['baseline'][0]:.3f} (p {p_eq(order_effect['baseline'][2])}). The ranking from GPT-4o's
    descriptions barely changes ({order_effect['4o'][1]:.3f} → {order_effect['4o'][0]:.3f}, p = {order_effect['4o'][2]:.2f}), and neither
    do the name-based rankings ({order_effect['k12'][1]:.3f} → {order_effect['k12'][0]:.3f}, {order_effect['k25'][1]:.3f} → {order_effect['k25'][0]:.3f}
    and {order_effect['k50'][1]:.3f} → {order_effect['k50'][0]:.3f} for 12, 25 and 50 names, p ≥ {min(order_effect[k][2] for k in ('k12', 'k25', 'k50')):.2f}).</li>
    <li><b>With a fair baseline, demographic descriptions help.</b> GPT-4o's descriptions score {so['ap_gpt-4o']:.3f} against
    {so['ap_baseline']:.3f} without user information (p {p_eq(st['4o_vs_baseline'])}).</li>
    <li><b>The gain does not depend on leaked interests.</b> Demographics-only descriptions also beat the no-information ranking
    ({g_clean[2]:.3f} vs. {g_clean[3]:.3f}, p {p_eq(g_clean[4])}), as they do in Section 3.</li>
    <li><b>Account names give larger gains than demographics.</b> Ranking from 12 followed account names scores {so['ap_k12']:.3f},
    above GPT-4o's descriptions (p {p_eq(st['k12_vs_4o'])}). 25 names add nothing over 12 (p = {st['k25_vs_k12']:.2f}); 50 names do
    ({so['ap_k50']:.3f}, p {p_eq(st['k50_vs_k12'])}).</li>
    <li><b>Social embeddings vs. the LLM.</b> SocialVec, using all of the user's followed entities, scores {so['socialvec']:.3f}.
    <ul>
      <li><i>Demographics:</i> GPT-4o's descriptions stay below it (p {p_eq(st['4o_vs_socialvec'])}).</li>
      <li><i>50 names:</i> the LLM with 50 account names scores above it (p {p_eq(st['k50_vs_socialvec'])}).</li>
      <li><i>Same accounts:</i> SocialVec using only the same 12 or 25 accounts scores {names_vs_sv[12][1]:.3f} and
      {names_vs_sv[25][1]:.3f}, below the LLM's {names_vs_sv[12][0]:.3f} and {names_vs_sv[25][0]:.3f}
      (p {p_eq(names_vs_sv[12][2])} and p {p_eq(names_vs_sv[25][2])}; means over users).</li>
    </ul></li>
  </ul></p>

  <h3>By description type</h3>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>GPT-4o description</th><th>Users</th>
  <th>Ranking from the description</th><th>LLM, no user information</th><th>Δ</th><th>p (Wilcoxon)</th></tr></thead>
  <tbody>{group_rows}</tbody></table></div>
  <p class="caption">MAP over the users of each row, pooled across categories. Both rankings in a row used the same shuffled
  candidates.</p></div>

  <h3>Per category</h3>
  <div class="card">{category_table(sh_cat['81-100'], SH_METHODS, SH_COLORS, sh.groupby('topic').size().to_dict())}
  <p class="caption">MAP per category with shuffled candidates; bold marks the best method in each row. Ranking completeness: at least
  {sh_complete:.0%} of the responses of every condition listed all candidates (legacy scoring skips missing ones).</p></div>
</section>

<section>
  <h2><span class="num">5</span>Examples</h2>
  <p class="intro">The cases are chosen by rule rather than by hand: the highest and the lowest AP among complete,
  demographics-only descriptions for users who follow at least two candidates, and a description that names interests with
  typical AP. Rankings are those with candidates in follower-count order. Users are identified by internal row only.</p>
  {''.join(example_block(r, t) for r, t in examples)}
</section>

<section>
  <h2><span class="num">6</span>Discussion and limitations</h2>
  <ul class="highlights">
    <li><b>The model does not always follow the instruction.</b> GPT-4o described every user, but in {leak_n / len(m):.0%} of
    cases it also named the user's interests. The clean condition is the demographics-only subset. A stricter test would filter or
    rewrite descriptions to a fixed format (e.g. four labelled fields), or supply the attributes from SocialVec's classifiers
    directly.</li>
    <li><b>The candidate order matters for this comparison.</b> With popularity-sorted candidates, the no-information ranking
    partly copies the presented order, and demographics-only descriptions looked no better than it. With shuffled candidates, they
    give a significant gain. Comparisons with a non-personalized LLM ranking should therefore use randomized candidate orders.</li>
    <li><b>Demographics are a coarse summary of taste.</b> Ranking from four attributes scores below ranking from 12 account names
    and below SocialVec's full-profile embedding. The account names carry information beyond the demographic groups they imply.</li>
    <li><b>Stereotype-based inference.</b> The method asks a model to apply stereotypes about gender, age, ethnicity and
    education. The resulting descriptions are guesses about real people. They are stored outside the repository, and this page
    identifies users by internal row only. Conclusions about which groups like which entities should not be drawn from them.</li>
    <li><b>Scale and noise.</b> 30 users per category, one description and one ranking per user. LLM outputs are not
    deterministic: repeated identical prompts change a user's AP by about 0.06 on average (see the candidate-order experiment).
    Stage 1 with GPT-4o cost ${cost.get('openai/gpt-4o', 0):.2f} for {len(m)} descriptions.</li>
  </ul>
</section>

<footer><p>Regenerate with <code>python run_llm_demographics.py --describe-model gpt-4o</code> and
<code>python run_llm_demographics_bands.py</code> (model calls; cached responses are reused), then
<code>python make_demographics_report.py</code>. Per-user scores are in <code>results/llm/demographics_per_user_gpt-4o.csv.gz</code>
and <code>results/llm/demographics_bands_per_user.csv.gz</code>; descriptions and usernames are in <code>data/llm_demographics/</code>,
which is not tracked by git.</p></footer>
</div>
<script>{JS}</script>
</body></html>
"""
(RESULTS_DIR / "LLM_DEMOGRAPHICS_REPORT.html").write_text(page, encoding="utf-8")
print("wrote", RESULTS_DIR / "LLM_DEMOGRAPHICS_REPORT.html")

# headline numbers for the summaries in results.html and results_backup.html (make_results_page.py)
summary = {
    "users": len(m), "leak_4o": int(leak_n), "sorted_map": overall.round(4).to_dict(),
    "shuffled": {b: {"users": int(sh_n[b]), "map": sh_overall[b].round(4).to_dict(), "tests": sh_tests[b],
                     "clean_4o": dict(zip(["label", "users", "map", "baseline", "p"], sh_groups[b][0]))} for b in SH_BANDS},
    "order_effect": {k: dict(zip(["shuffled", "sorted", "p"], v)) for k, v in order_effect.items()},
    "names_vs_socialvec_81_100": {k: dict(zip(["llm", "socialvec", "p"], v)) for k, v in names_vs_sv.items()},
    "attributes": {attr: {"values": [v for v, _ in ATTRIBUTE_VALUES[attr]] + ["Not stated"],
                          "counts": attributes[attr].value_counts().to_dict(), "multiple": attributes_multiple[attr],
                          "by_category": {t: g.value_counts().to_dict() for t, g in attributes[attr].groupby(m["topic"])}}
                   for attr in ATTRIBUTE_VALUES},
}
(LLM / "demographics_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print("examples:", examples)
print("status:", m["status"].value_counts().to_dict(), "leak:", leak_n)
