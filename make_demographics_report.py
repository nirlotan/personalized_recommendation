"""
Build results/LLM_DEMOGRAPHICS_REPORT.html: the demographics-only LLM ranking experiment
(run_llm_demographics.py) compared with every other method on the same users.

    python make_demographics_report.py

Reads the generated descriptions from data/llm_demographics/ (not tracked by git). The page
identifies users by internal row only, never by username.
"""

import html
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

from reproduction.config import CATEGORY_LABELS, DATA_DIR, PAPER_CATEGORY_ORDER, RESULTS_DIR
from reproduction.data import load_categories, load_users
from reproduction.demographics import classify_users
from reproduction.embeddings import SocialVec, load_metadata
from reproduction.llm import names_of, ranked_ids, user_samples
from reproduction.llm_demographics import PROMPT_VERSION
from reproduction.page_style import CSS, JS
from run_llm_experiments import band_sample

LLM = RESULTS_DIR / "llm"
PRIVATE = DATA_DIR / "llm_demographics"

# ---------------------------------------------------------------- data
desc = pd.read_csv(PRIVATE / "descriptions.csv").set_index("row")
demo = pd.read_csv(LLM / "demographics_per_user.csv.gz").set_index("user")
band = pd.read_csv(LLM / "band_per_user.csv.gz").set_index("user")
m = band.join(demo[["llm_demographics", "n_accounts_described"]], how="inner").join(desc[["description", "ranking_response"]])
# Stage 1 repeated with GPT-4o (Stage 2 still GPT-6 Luna)
desc4 = pd.read_csv(PRIVATE / "descriptions_gpt-4o.csv").set_index("row")
demo4 = pd.read_csv(LLM / "demographics_per_user_gpt-4o.csv.gz").set_index("user")
m = m.join(demo4["llm_demographics"].rename("llm_demographics_4o")).join(
    desc4[["description", "ranking_response"]].rename(columns=lambda c: f"{c}_4o"))
runs = json.loads((LLM / "run_info.json").read_text())
run_4o = [r for r in runs if r.get("describe_model") == "gpt-4o"][-1]

categories = load_categories()
users = load_users(categories)
sample, candidates, exclude, _ = band_sample(users, categories)
metadata = load_metadata()
names = dict(zip(metadata["twitter_id"].astype(str), metadata["name"]))
prompt_ids = user_samples(sample, categories, names, exclude=exclude)["llm_k50"]
cand_name = {int(i): n for c in candidates.values() for i, n in zip(c["twitter_user_id"], c["twitter_name"])}

# ---------------------------------------------------------------- description status and parsed attributes
ATTR = {
    "gender": r"\b(woman|man|female|male|women|men)\b",
    "age": r"(\b\d0s\b|\bteen|\byoung\b|middle[- ]aged|\bolder\b|\bsenior|\badult\b|\b\d{2}\s*(?:-|to|–)\s*\d{2}\b)",
    "ethnicity": r"\b(white|black|caucasian|african[- ]american|hispanic|latin[oax]\w*|asian)\b",
    "education": r"(college|university|degree|high[- ]school|higher[- ]education|graduate|bachelor|primary school|educated)",
}
# Topics or interests named in a description, which the prompt forbids (it leaks more than demographics)
TOPICS = (r"\b(?:politic\w*|sport\w*|music\w*|entertainment|news|pop culture|conservative|liberal|progressive|republican|"
          r"democrat\w*|business|military|gaming|comedy|celebrit\w*|interest\w*|fan|fans|enthusiast\w*|hip[- ]hop|media)\b")


def description_status(texts: pd.Series) -> pd.Series:
    n = sum(texts.str.contains(p, case=False, regex=True).astype(int) for p in ATTR.values())
    return pd.Series(np.select([n == 4, n == 0], ["complete", "declined"], "partial"), index=texts.index)


for attr, pattern in ATTR.items():
    m[attr] = m["description"].str.contains(pattern, case=False, regex=True)
m["n_attr"] = m[list(ATTR)].sum(axis=1)
m["status"] = description_status(m["description"])
m["status_4o"] = description_status(m["description_4o"])
m["leak"] = m["description"].str.contains(TOPICS, case=False, regex=True)
m["leak_4o"] = m["description_4o"].str.contains(TOPICS, case=False, regex=True)


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


def parse_attributes(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"gender": texts.map(parse_gender), "ethnicity": texts.map(parse_ethnicity),
                         "education": texts.map(parse_education), "age": texts.map(parse_age)})


parsed = parse_attributes(m.loc[m["status"] != "declined", "description"])
parsed_4o = parse_attributes(m.loc[m["status_4o"] != "declined", "description_4o"])
traits = classify_users(sample.loc[m.index], SocialVec("2020c"))

# ---------------------------------------------------------------- numbers
METHODS = [
    ("random", "Random ranking (expected)", "Non-personalized"),
    ("popularity", "Popularity (follower count)", "Non-personalized"),
    ("llm_baseline", "LLM, no user information", "Non-personalized"),
    ("llm_demographics", "LLM, demographics by GPT-6 Luna", "This experiment"),
    ("llm_demographics_4o", "LLM, demographics by GPT-4o", "This experiment"),
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
by_cat.loc["Overall"] = overall


def pv(a, b):
    d = np.asarray(a) - np.asarray(b)
    return wilcoxon(d).pvalue if (d != 0).any() else np.nan


def ptxt(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


tests = [(label, overall["llm_demographics"] - overall[c], pv(m["llm_demographics"], m[c]))
         for c, label, _ in METHODS if c not in ("llm_demographics", "llm_demographics_4o")]
tests_4o = [(label, overall["llm_demographics_4o"] - overall[c], pv(m["llm_demographics_4o"], m[c]))
            for c, label, _ in METHODS if c not in ("llm_demographics_4o", "random")]

status_rows = []
for s, label in [("complete", "All four attributes guessed"), ("partial", "Some attributes guessed"), ("declined", "Declined to guess")]:
    g = m[m["status"] == s]
    status_rows.append((label, len(g), g["llm_demographics"].mean(), g["llm_baseline"].mean(), g["popularity"].mean(),
                        g["llm_k12"].mean(), g["socialvec_k12"].mean(), pv(g["llm_demographics"], g["llm_baseline"])))
by_status_cat = m.groupby(["topic", "status"]).size().unstack(fill_value=0).reindex(by_cat.index[:-1])


def tau(response, reference):
    ids = [a for a in ranked_ids(response) if a in set(reference)]
    if len(ids) < 3:
        return np.nan
    pos = {a: i for i, a in enumerate(reference)}
    return kendalltau(range(len(ids)), [pos[a] for a in ids]).statistic


pop_order = {t: [int(i) for i in c.sort_values("followers_count", ascending=False)["twitter_user_id"]] for t, c in candidates.items()}
m["tau_baseline"] = [tau(r, ranked_ids(b)) for r, b in zip(m["ranking_response"], m["resp_llm_baseline"])]
m["tau_popularity"] = [tau(r, pop_order[t]) for r, t in zip(m["ranking_response"], m["topic"])]
tau_by_status = m.groupby("status")[["tau_baseline", "tau_popularity"]].mean()

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
agreement_4o = agreement_with_socialvec(parsed_4o)

# GPT-4o descriptions: who complied, who leaked interests, and what each subset scores
clean_4o = m[(m["status_4o"] == "complete") & ~m["leak_4o"]]
leak_4o = m[m["leak_4o"]]
luna_declined = m[m["status"] == "declined"]
luna_complete = m[m["status"] == "complete"]
subsets_4o = [
    ("All users", m),
    ("GPT-4o description names interests or topics", leak_4o),
    ("GPT-4o description is demographics only", m[~m["leak_4o"]]),
    ("GPT-6 Luna declined; GPT-4o described", luna_declined),
    ("Both models gave complete descriptions", luna_complete[luna_complete["status_4o"] == "complete"]),
]
subset_rows_4o = [(label, len(g), g["llm_demographics_4o"].mean(), g["llm_demographics"].mean(), g["llm_baseline"].mean(),
                   pv(g["llm_demographics_4o"], g["llm_baseline"]))
                  for label, g in subsets_4o]

# ---------------------------------------------------------------- examples
complete = m[m["status"] == "complete"]
multi = complete[complete["n_relevant"] >= 2]
ex_high = multi.sort_values(["llm_demographics", "n_relevant"], ascending=False).index[0]
ex_low = multi.assign(gap=multi["llm_baseline"] - multi["llm_demographics"]).sort_values(["llm_demographics", "gap"], ascending=[True, False]).index[0]
declined = m[m["status"] == "declined"]
ex_declined = (declined["llm_demographics"] - declined["llm_demographics"].median()).abs().sort_values().index[0]
examples = [(ex_high, "A high-AP case"), (ex_low, "A low-AP case"), (ex_declined, "A case where the model declined")]
if 1864 in m.index and 1864 not in {ex_high, ex_low, ex_declined}:
    examples.append((1864, "The debugging user"))
# GPT-4o: the user GPT-6 Luna declined in both debugging rounds, and a leaking description with typical AP
leaks = m[m["leak_4o"]]
ex_leak = (leaks["llm_demographics_4o"] - leaks["llm_demographics_4o"].median()).abs().sort_values().index[0]
examples_4o = [(1808, "GPT-6 Luna declined; GPT-4o described"), (ex_leak, "A GPT-4o description that names interests")]

# ---------------------------------------------------------------- html helpers
esc = html.escape


def f3(v):
    return "" if pd.isna(v) else f"{v:.3f}"


def signed(v):
    return f"{v:+.3f}".replace("-", "−")


def methods_table():
    head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in METHODS)
    rows = []
    for cat, r in by_cat.iterrows():
        cls = " class='total'" if cat == "Overall" else ""
        best = r[cols].max()
        cells = "".join(f"<td class='{'hl' if c.startswith('llm_demographics') else ''}'>{'<b>' if r[c] == best else ''}{f3(r[c])}{'</b>' if r[c] == best else ''}</td>" for c in cols)
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th>{cells}</tr>")
    return f"<div class='scroll'><table class='data compact'><thead><tr><th>Category</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def overall_chart():
    W, L, Rm, T, rowh = 760, 250, 60, 10, 30
    H = T + rowh * len(METHODS) + 30
    x1 = 0.55
    px = lambda v: L + v / x1 * (W - L - Rm)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Overall MAP by method">']
    for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        parts.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{T}" y2="{H-24}"/>'
                     f'<text class="tick" x="{px(v):.1f}" y="{H-8}" text-anchor="middle">{v:.1f}</text>')
    for i, (c, label, group) in enumerate(METHODS):
        y = T + i * rowh
        v = overall[c]
        cls = {"llm_demographics": "s1", "llm_demographics_4o": "s2"}.get(c, "muted-bar")
        w = px(v) - L
        weight = ' font-weight="650"' if c.startswith("llm_demographics") else ""
        parts.append(f'<text class="cat" x="{L-10}" y="{y+17}" text-anchor="end"{weight}>{esc(label)}</text>'
                     f'<path class="bar {cls}" d="M{L},{y+6} h{w-4:.1f} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 h-{w-4:.1f} z"/>'
                     f'<text class="delta" x="{px(v)+8:.1f}" y="{y+17}"{weight}>{v:.3f}</text>'
                     f'<rect class="hit" x="{L}" y="{y+4}" width="{w:.1f}" height="18" data-tip="{esc(label)}<br>MAP <b>{v:.3f}</b> · {esc(group)}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def example_block(row, title, variant=""):
    """One worked example; variant "" shows the GPT-6 Luna description, "_4o" the GPT-4o one."""
    r = m.loc[row]
    topic = r["topic"]
    followed = names_of(prompt_ids[row], names)
    relevant = {f for f in sample.at[row, "friends_ids"] if f in set(candidates[topic]["twitter_user_id"])}
    order = ranked_ids(r[f"ranking_response{variant}"])
    missing = [i for i in candidates[topic]["twitter_user_id"] if int(i) not in set(order)]
    items = "".join(f"<li class='{'hit-item' if a in relevant else ''}'>{esc(cand_name.get(a, str(a)))}{' ✓' if a in relevant else ''}</li>" for a in order)
    found = sorted(order.index(a) + 1 for a in relevant if a in order)
    calc = " + ".join(f"{i + 1}/{p}" for i, p in enumerate(found))
    model = "GPT-4o" if variant else "GPT-6 Luna"
    this = f"llm_demographics{variant}"
    comp = [("Random (expected)", "random"), ("Popularity", "popularity"), ("LLM, no information", "llm_baseline"),
            ("LLM, demographics by GPT-6 Luna", "llm_demographics"), ("LLM, demographics by GPT-4o", "llm_demographics_4o"),
            ("LLM, 12 names", "llm_k12"), ("SocialVec, same 12", "socialvec_k12"), ("SocialVec, all entities", "socialvec")]
    comp_rows = "".join(f"<tr{' class=hl-row' if c == this else ''}><th>{n}</th><td>{f3(r[c])}</td></tr>" for n, c in comp)
    status_label = {"complete": "all four attributes guessed", "partial": "some attributes guessed",
                    "declined": "declined to guess"}[r[f"status{variant}"]]
    if variant and r["leak_4o"]:
        status_label += "; also names interests"
    other = ""
    if variant:
        other = (f"<p class='step'><span>·</span>For comparison, GPT-6 Luna's description</p>"
                 f"<p class='quote small'>{esc(r['description'])}</p>")
    return f"""
    <div class="example card">
      <h4>{esc(title)} · user row {row} · {esc(CATEGORY_LABELS[topic])}</h4>
      <div class="ex-grid">
        <div>
          <p class="step"><span>1</span>Accounts given to the model ({len(prompt_ids[row])} names)</p>
          <p class="quote small">{esc(followed)}</p>
          <p class="step"><span>2</span>Description by {model} ({status_label})</p>
          <p class="quote">{esc(r[f'description{variant}'])}</p>
          {other}
          <p class="step"><span>3</span>GPT-6 Luna's ranking from the description only (✓ = followed by the user)</p>
          <ol class="ranking">{items}</ol>
          {f"<p class='caption'>Not returned by the model: {', '.join(esc(cand_name.get(int(i), str(i))) for i in missing)}.</p>" if missing else ""}
        </div>
        <div>
          <p class="step"><span>4</span>Score</p>
          <p class="ap">AP = ({calc}) / {len(found)} = <b>{r[this]:.3f}</b></p>
          <table class="data compact mini"><tbody>{comp_rows}</tbody></table>
        </div>
      </div>
    </div>"""


EXTRA_CSS = """
.chart .bar.muted-bar{fill:var(--axis)}
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
comp_label, comp_n, comp_demo, comp_base, comp_pop, comp_k12, comp_sv12, comp_p = status_rows[0]
dec_label, dec_n, dec_demo, dec_base, dec_pop, dec_k12, dec_sv12, dec_p = status_rows[2]
part_n = status_rows[1][1]

status_table = "".join(
    f"<tr><th>{label}</th><td>{n}</td><td class='hl'>{f3(d)}</td><td>{f3(b)}</td><td>{f3(p)}</td><td>{f3(k)}</td><td>{f3(s12)}</td><td>{ptxt(pp) if not pd.isna(pp) else ''}</td></tr>"
    for label, n, d, b, p, k, s12, pp in status_rows)
cat_status = "".join(f"<tr><th>{esc(CATEGORY_LABELS[c])}</th><td>{r.get('complete', 0)}</td><td>{r.get('partial', 0)}</td><td>{r.get('declined', 0)}</td></tr>"
                     for c, r in by_status_cat.iterrows())
test_rows = "".join(f"<tr><th>{esc(label)}</th><td>{signed(diff)}</td><td>{ptxt(p)}</td></tr>" for label, diff, p in tests)
agree_rows = "".join(f"<tr><th>{esc(label)}</th><td>{n}</td><td>{a:.0%}</td><td>{maj:.0%}</td><td class='muted'>{esc(', '.join(f'{k} {v}' for k, v in dist.items()))}</td></tr>"
                     for label, n, a, maj, dist in agreement)
tau_rows = "".join(f"<tr><th>{s.capitalize()}</th><td>{f3(r['tau_baseline'])}</td><td>{f3(r['tau_popularity'])}</td></tr>"
                   for s, r in tau_by_status.reindex(["complete", "partial", "declined"]).iterrows())

above = [(label.split(" (")[0].lower(), a, maj) for label, _, a, maj, _ in agreement if a - maj >= 0.1]
near = [(label.split(" (")[0].lower(), a, maj) for label, _, a, maj, _ in agreement if a - maj < 0.1]
agreement_text = (
    "Where the model did state an attribute, its guesses agree with SocialVec's classifiers well beyond a constant guess for "
    + " and ".join(f"{n} ({a:.0%} vs. {maj:.0%})" for n, a, maj in above)
    + ". For " + " and ".join(f"{n} ({a:.0%} vs. {maj:.0%})" for n, a, maj in near)
    + ", agreement is close to what always guessing the most common class would give, so the guesses add little there."
) if above and near else ""

# GPT-4o section tables
test_rows_4o = "".join(f"<tr><th>{esc(label)}</th><td>{signed(diff)}</td><td>{ptxt(p)}</td></tr>" for label, diff, p in tests_4o)
subset_table_4o = "".join(
    f"<tr><th>{esc(label)}</th><td>{n}</td><td class='hl'>{f3(g4)}</td><td>{f3(gl)}</td><td>{f3(b)}</td><td>{ptxt(p) if not pd.isna(p) else ''}</td></tr>"
    for label, n, g4, gl, b, p in subset_rows_4o)
agree_rows_4o = "".join(f"<tr><th>{esc(label)}</th><td>{n}</td><td>{a:.0%}</td><td>{maj:.0%}</td><td class='muted'>{esc(', '.join(f'{k} {v}' for k, v in dist.items()))}</td></tr>"
                        for label, n, a, maj, dist in agreement_4o)
outcomes = "".join(
    f"<tr><th>{model}</th><td>{(m[s] == 'complete').sum()}</td><td>{(m[s] == 'partial').sum()}</td><td>{(m[s] == 'declined').sum()}</td><td>{int(m[lk].sum())}</td></tr>"
    for model, s, lk in [("GPT-6 Luna", "status", "leak"), ("GPT-4o", "status_4o", "leak_4o")])
s4 = {label: (n, g4, gl, b, p) for label, n, g4, gl, b, p in subset_rows_4o}
leak_n, leak_g4, _, leak_b, leak_p = s4["GPT-4o description names interests or topics"]
clean_n, clean_g4, _, clean_b, clean_p = s4["GPT-4o description is demographics only"]
switch_n, switch_g4, switch_gl, switch_b, _ = s4["GPT-6 Luna declined; GPT-4o described"]
both_n, both_g4, both_gl, both_b, _ = s4["Both models gave complete descriptions"]
p_4o_luna = pv(m["llm_demographics_4o"], m["llm_demographics"])
p_switch = pv(luna_declined["llm_demographics_4o"], luna_declined["llm_demographics"])
p_both = pv(*[luna_complete.loc[luna_complete["status_4o"] == "complete", c] for c in ("llm_demographics_4o", "llm_demographics")])
switch_leak, switch_clean = luna_declined[luna_declined["leak_4o"]], luna_declined[~luna_declined["leak_4o"]]
p_switch_clean = pv(switch_clean["llm_demographics_4o"], switch_clean["llm_demographics"])
cost_4o = run_4o["cost_by_model"]


def p_eq(p):
    """A p-value for running text: '< 0.001' or '= 0.006'."""
    return ptxt(p) if p < 0.001 else f"= {p:.3f}"


above_4o = [(label.split(" (")[0].lower(), a, maj) for label, _, a, maj, _ in agreement_4o if a - maj >= 0.05]
dominant = {label.split(" (")[0].lower(): max(dist.items(), key=lambda kv: kv[1]) for label, _, _, _, dist in agreement_4o}
agreement_text_4o = (
    "GPT-4o's guesses lean heavily toward the most common class: "
    + ", ".join(f"{dominant[n][1]} of its {n} guesses are {dominant[n][0]}" for n in ("education", "age", "ethnicity"))
    + ". Agreement with SocialVec is therefore close to the majority-class rate, except for "
    + " and ".join(f"{n} ({a:.0%} vs. {maj:.0%})" for n, a, maj in above_4o) + "."
) if above_4o else ""

o = overall
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demographic LLM Ranking</title><style>{CSS}{EXTRA_CSS}</style></head>
<body><div id="tip" role="tooltip"></div>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">Extension · Social Knowledge for Cross-Domain User Preference Modeling</p>
  <h1>Can demographics alone personalize an LLM's recommendations?</h1>
  <p class="lede">An LLM first infers a user's gender, age, ethnicity and education from the names of accounts they follow,
  then ranks less popular entities given only that inferred description. I compare this with ranking from the account
  names themselves, with social embeddings, and with non-personalized baselines, on the same 270 users.</p>
  <div class="tiles">
    <div class="tile"><div class="label">MAP, demographic description only</div><div class="value">{o['llm_demographics']:.3f} · {o['llm_demographics_4o']:.3f}</div><div class="sub">descriptions by GPT-6 Luna · by GPT-4o; {o['llm_baseline']:.3f} with no user information</div></div>
    <div class="tile"><div class="label">Users the model declined to describe</div><div class="value">{dec_n} · {int((m['status_4o'] == 'declined').sum())}</div><div class="sub">of {len(m)}: GPT-6 Luna · GPT-4o, despite being told to always guess</div></div>
    <div class="tile"><div class="label">GPT-4o descriptions that also name interests</div><div class="value">{leak_n} / {len(m)}</div><div class="sub">MAP {leak_g4:.3f} for those users; {clean_g4:.3f} for demographics-only descriptions</div></div>
    <div class="tile"><div class="label">MAP when GPT-6 Luna guessed all four attributes</div><div class="value">{comp_demo:.3f}</div><div class="sub">vs. {comp_base:.3f} without user information; {comp_k12:.3f} from 12 account names ({comp_n} users)</div></div>
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

  <p><b>Stage 1: demographic inference.</b> For each user, the LLM receives the names of the accounts listed in that user's
  k = 50 prompt. These are 50 accounts drawn at random from everything the user follows, after removing every listed entity of the
  target category, keeping only accounts with a known display name ({m['n_accounts_described'].mean():.1f} names on average;
  minimum {int(m['n_accounts_described'].min())}). The model is asked for a one- or two-sentence description stating the user's
  most likely gender, age, ethnicity and level of education. It is instructed to always guess from stereotypes about the
  typical audience of the accounts, never to state that the attributes cannot be determined, and not to mention interests or
  accounts, so that the description carries demographic information only. The prompt, in the dspy signature format of the
  original experiments:</p>
  <pre class="prompt">Given the names of accounts that a Twitter user follows, write a short, concise description of the user that states their
most likely gender, age, ethnicity, and level of education (primary school, high school, higher education, etc.).
Deduce these attributes only from the names of the accounts they follow.
You must always guess the most likely value of all four attributes, based on stereotypes about the typical audience of
these accounts, even when the evidence is weak. Never answer that the attributes cannot be determined or that the
evidence is insufficient.
Describe only these four attributes: do not mention the user's interests or hobbies, and do not name any of the accounts.

Input  followed_accounts: Names of accounts the user follows
Output description: One or two sentences stating the user's most likely gender, age, ethnicity, and level of education</pre>

  <p><b>Stage 2: ranking from the description.</b> The ranking prompt is the authors' personalized ranking prompt, with the
  list of followed accounts replaced by the Stage 1 description. As in the original experiments, the candidates are given as a
  JSON list of ids and display names, sorted by follower count, and the model must return a full ranking of the ids:</p>
  <pre class="prompt">given the below Twitter user, for whom we provide a short demographic description, and a list of possible accounts to
recommend for the user to follow, create a prioritized recommendation list.
You must rank the entire list provided to you.
Try to deduce what you think the user might like based on their demographic description.
Return ONLY a valid list object with the ranked account twitter_user_id sorted by relevance (highest to lowest). Do not
include name or reasoning.

Inputs user_description, potential_accounts (JSON list of twitter_user_id and names)
Output ranked_ids</pre>

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

  <p><b>Repeating Stage 1 with GPT-4o.</b> Because GPT-6 Luna often refused to infer the attributes (Section 3), we
  repeated Stage 1 with GPT-4o (default sampling settings), using the identical prompt and inputs. Stage 2 was left
  unchanged: GPT-6 Luna ranks the candidates from GPT-4o's description, so the two runs differ only in who wrote the
  description. Since the prompt forbids mentioning interests, we also flag descriptions that name topics or interests (for
  example politics, sports, music or news) as <i>leaking</i> information beyond demographics.</p>

  <p><b>Implementation.</b> GPT-6 Luna with reasoning effort <i>low</i>, called through dspy without caching, under a
  spending cap. Responses are cached locally per prompt version (this run uses version {PROMPT_VERSION}).
  dspy's end-of-output marker, which the model sometimes leaves inside the description, is removed before Stage 2.
  To measure how much demographic information each description carries, we label it <i>complete</i> if it states a value
  for all four attributes, <i>declined</i> if it states none, and <i>partial</i> otherwise. For complete and partial
  descriptions, we parse the stated values into the binary classes of SocialVec's trait classifiers, and compare them with
  those classifiers' predictions from the user's full embedding.</p>
</section>

<section>
  <h2><span class="num">2</span>Results</h2>
  <p class="intro">With GPT-6 Luna's descriptions, demographics add almost nothing: {o['llm_demographics']:.3f} MAP, compared with
  {o['llm_baseline']:.3f} for the LLM without user information and {o['popularity']:.3f} for popularity. GPT-4o's descriptions
  give {o['llm_demographics_4o']:.3f}, but most of that gain comes from descriptions that break the prompt by naming the user's
  interests (Section 4). Naming 12 followed accounts gives {o['llm_k12']:.3f}.</p>
  <div class="card">{overall_chart()}
  <p class="caption">Overall MAP (mean over nine categories) on the same 270 users. The highlighted bar is this experiment.</p></div>

  <h3>Per category</h3>
  <div class="card">{methods_table()}
  <p class="caption">MAP per category; bold marks the best method in each row. With 30 users per category, per-category MAP has a
  standard error of about 0.05. The demographic ranking helps most for musical artists ({by_cat.loc['MusicalArtist','llm_demographics']:.3f} vs.
  {by_cat.loc['MusicalArtist','llm_baseline']:.3f} without user information) and news outlets ({by_cat.loc['news','llm_demographics']:.3f} vs.
  {by_cat.loc['news','llm_baseline']:.3f}), and hurts for politicians ({by_cat.loc['Politician','llm_demographics']:.3f} vs.
  {by_cat.loc['Politician','llm_baseline']:.3f}).</p></div>

  <h3>Statistical tests</h3>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Demographics by GPT-6 Luna vs.</th><th>Δ MAP</th><th>p (Wilcoxon)</th></tr></thead>
  <tbody>{test_rows}</tbody></table></div>
  <p class="caption">Δ is the overall MAP of the demographic ranking minus that of the other method. The same tests for
  GPT-4o's descriptions are in Section 4.</p></div>
</section>

<section>
  <h2><span class="num">3</span>The model often refused to infer demographics</h2>
  <p class="intro">Despite the instruction to always guess, the model declined to state any attribute for {dec_n} of the {len(m)}
  users ({dec_n / len(m):.0%}). Typical answers were "I can’t infer the user’s gender, age, ethnicity, or education level
  from the accounts they follow." It gave all four attributes for {comp_n} users, and only some for {part_n}, typically
  refusing ethnicity alone. Refusals occur in every category, from {by_status_cat['declined'].min()} to {by_status_cat['declined'].max()}
  of the 30 users, and are most frequent for {' and '.join(CATEGORY_LABELS[c].lower() for c in by_status_cat.index[by_status_cat['declined'] == by_status_cat['declined'].max()])}.
  The debugging runs did not reveal how common refusals are. Row 1864 complied, while row 1808, which had declined under the
  first prompt, declined again in the full run despite the stricter instruction.</p>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Description</th><th>Users</th><th>LLM, demographics</th>
  <th>LLM, no information</th><th>Popularity</th><th>LLM, 12 names</th><th>SocialVec, same 12</th><th>p (demographics vs. no information)</th></tr></thead>
  <tbody>{status_table}</tbody></table></div>
  <p class="caption">MAP by description type (users pooled across categories, so values are not directly comparable to the
  per-category means above).</p></div>
  <p><b>When the model does describe the user, demographics carry real signal.</b> For the {comp_n} complete descriptions, the
  demographic ranking reaches {comp_demo:.3f}, compared with {comp_base:.3f} without user information (p = {comp_p:.2f}) and
  {comp_pop:.3f} for popularity. It is on par with the LLM given 12 account names ({comp_k12:.3f}) and with SocialVec on those
  same 12 accounts ({comp_sv12:.3f}). For declined descriptions, the model ranks from an uninformative sentence, and the result
  is slightly but significantly worse than the no-information baseline ({dec_demo:.3f} vs. {dec_base:.3f}, p = {dec_p:.2f}).
  The overall null result is therefore mostly a consequence of the refusals.</p>
  <p><b>Caveat.</b> With {comp_n} users, the complete-description estimate is uncertain. These users were also not selected at
  random: the model may comply more readily when the accounts point clearly to a demographic group, which may also make the
  ranking easier.</p>

  <div class="panels">
    <div class="panel"><h4>How much the description changes the ranking</h4><div class="scroll"><table class="data compact">
    <thead><tr><th>Description</th><th>τ with no-information ranking</th><th>τ with popularity order</th></tr></thead>
    <tbody>{tau_rows}</tbody></table></div>
    <p class="caption">Mean Kendall τ between the demographic ranking and (a) the LLM's ranking without user information for
    the same category, (b) follower-count order. A complete description moves the ranking further away from both.</p></div>
    <div class="panel"><h4>Descriptions by category</h4><div class="scroll"><table class="data compact">
    <thead><tr><th>Category</th><th>Complete</th><th>Partial</th><th>Declined</th></tr></thead><tbody>{cat_status}</tbody></table></div></div>
  </div>

  <h3>Do the guesses agree with SocialVec's trait classifiers?</h3>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Attribute</th><th>Users with a parsed guess</th>
  <th>Agreement with SocialVec</th><th>Majority-class rate</th><th>LLM guesses</th></tr></thead><tbody>{agree_rows}</tbody></table></div>
  <p class="caption">For complete and partial descriptions, the stated value is mapped to the classifier's binary classes
  (descriptions stating another value, or an ambiguous one such as "20s" for age, are left out). The majority-class rate is
  the agreement a constant guess of SocialVec's most frequent class would achieve. Treat this as a rough check: the parsing is
  rule-based, and SocialVec's classifiers are themselves only estimates.</p></div>
  <p>{agreement_text}</p>
</section>

<section>
  <h2><span class="num">4</span>Repeating the inference with GPT-4o</h2>
  <p class="intro">GPT-4o's policy is looser: it described every user. But it also often ignored the instruction to describe
  demographics only, and the extra information it added explains most of its advantage.</p>

  <div class="panels">
    <div class="panel"><h4>Description outcomes</h4><div class="scroll"><table class="data compact">
    <thead><tr><th>Stage 1 model</th><th>Complete</th><th>Partial</th><th>Declined</th><th>Names interests</th></tr></thead>
    <tbody>{outcomes}</tbody></table></div>
    <p class="caption">Users out of {len(m)}. GPT-4o's two "partial" descriptions do guess all four attributes, but in wording the
    parser misses ("twenties", "diverse ethnicity"). The last column counts descriptions that name topics or interests.</p></div>
    <div class="panel"><h4>Cost</h4><div class="scroll"><table class="data compact"><thead><tr><th>Model</th><th>Role</th><th>Cost</th></tr></thead><tbody>
    <tr><th>GPT-4o</th><td>Stage 1, {len(m)} descriptions</td><td>${cost_4o.get('openai/gpt-4o', 0):.2f}</td></tr>
    <tr><th>GPT-6 Luna</th><td>Stage 2, {len(m)} rankings</td><td>${cost_4o.get('openai/gpt-6-luna', 0):.2f}</td></tr>
    <tr class="total"><th>Total</th><td></td><td>${run_4o['cost_usd']:.2f}</td></tr></tbody></table></div></div>
  </div>

  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Users</th><th>n</th><th>Demographics by GPT-4o</th>
  <th>Demographics by GPT-6 Luna</th><th>LLM, no information</th><th>p (GPT-4o vs. no information)</th></tr></thead>
  <tbody>{subset_table_4o}</tbody></table></div>
  <p class="caption">MAP over the users in each row, pooled across categories. In every row, Stage 2 is GPT-6 Luna.</p></div>

  <p><b>More descriptions, mostly from leaked interests.</b>
  <ul>
    <li><i>Overall:</i> GPT-4o's descriptions improve on GPT-6 Luna's ({o['llm_demographics_4o']:.3f} vs. {o['llm_demographics']:.3f}
    overall MAP, p = {p_4o_luna:.3f}).</li>
    <li><i>Leaking descriptions:</i> {leak_n} of them ({leak_n / len(m):.0%}) also name the user's interests, despite the instruction,
    for example "…with a higher education level, likely due to the interest in diverse media and sports analysts,
    personalities, and NFL-related accounts". For these users the ranking reaches {leak_g4:.3f}, far above the no-information
    baseline ({leak_b:.3f}, p {p_eq(leak_p)}).</li>
    <li><i>Demographics-only descriptions:</i> for the remaining {clean_n} users the score is {clean_g4:.3f}, indistinguishable from
    the baseline ({clean_b:.3f}, p = {clean_p:.2f}).</li>
  </ul>
  So GPT-4o's advantage reflects interest information, not demographics.</p>

  <p><b>Same users, both models.</b>
  <ul>
    <li><i>Both models complete:</i> where GPT-6 Luna had given a complete description, GPT-4o's descriptions perform the same
    ({both_g4:.3f} vs. {both_gl:.3f}, {both_n} users, p = {p_both:.2f}).</li>
    <li><i>GPT-6 Luna declined:</i> for the {switch_n} users GPT-6 Luna refused, GPT-4o's descriptions raise MAP from
    {switch_gl:.3f} to {switch_g4:.3f} (p {p_eq(p_switch)}). Most of this gain comes from the {len(switch_leak)} of them whose
    GPT-4o description leaked interests ({switch_leak['llm_demographics'].mean():.3f} → {switch_leak['llm_demographics_4o'].mean():.3f}).
    For the other {len(switch_clean)}, MAP rises only from {switch_clean['llm_demographics'].mean():.3f} to
    {switch_clean['llm_demographics_4o'].mean():.3f} (p {p_eq(p_switch_clean)}).</li>
  </ul>
  Taken together, explicit demographic guesses help most for the users whose followed accounts point clearly to a
  demographic group, which are also the users GPT-6 Luna was willing to describe.</p>

  <h3>GPT-4o's guesses vs. SocialVec's trait classifiers</h3>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Attribute</th><th>Users with a parsed guess</th>
  <th>Agreement with SocialVec</th><th>Majority-class rate</th><th>GPT-4o guesses</th></tr></thead><tbody>{agree_rows_4o}</tbody></table></div>
  <p class="caption">As in Section 3, using the same parsing rules, on all users GPT-4o described.</p></div>
  <p>{agreement_text_4o}</p>

  <h3>Statistical tests for GPT-4o's descriptions</h3>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Demographics by GPT-4o vs.</th><th>Δ MAP</th><th>p (Wilcoxon)</th></tr></thead>
  <tbody>{test_rows_4o}</tbody></table></div></div>

  <h3>GPT-4o examples</h3>
  {''.join(example_block(r, t, "_4o") for r, t in examples_4o)}
</section>

<section>
  <h2><span class="num">5</span>Examples</h2>
  <p class="intro">The cases are chosen by rule rather than by hand: the highest and the lowest AP among complete descriptions
  for users who follow at least two candidates; a declined description with typical AP; and the user used for debugging.
  Users are identified by internal row only.</p>
  {''.join(example_block(r, t) for r, t in examples)}
</section>

<section>
  <h2><span class="num">6</span>Discussion and limitations</h2>
  <ul class="highlights">
    <li><b>Model policies shape the result.</b> GPT-6 Luna declined to infer demographics for {dec_n / len(m):.0%} of users, even
    when told to rely on stereotypes. GPT-4o described every user, but in {leak_n / len(m):.0%} of cases it also named the user's
    interests, against the instruction. Neither model gives a clean demographics-only condition for all users.
    <ul>
      <li><i>Cleaner subsets:</i> the clean conditions are GPT-6 Luna's complete descriptions and GPT-4o's demographics-only
      descriptions. They show a benefit for some users, but none on average for the others.</li>
      <li><i>A stricter test:</i> filter or rewrite descriptions to a fixed format (e.g. four labelled fields), or supply the
      attributes from SocialVec's classifiers directly.</li>
    </ul></li>
    <li><b>Demographics are a coarse summary of taste.</b> Even when the description was complete, ranking from four
    attributes did not beat ranking from 12 account names, and it was far from SocialVec's full-profile embedding
    ({o['socialvec']:.3f}). The account names carry information beyond the demographic groups they imply.</li>
    <li><b>Stereotype-based inference.</b> The method asks a model to apply stereotypes about gender, age, ethnicity and
    education. The resulting descriptions are guesses about real people. They are stored outside the repository, and this page
    identifies users by internal row only. Conclusions about which groups like which entities should not be drawn from them.</li>
    <li><b>Scale and noise.</b> 30 users per category, one description and one ranking per user, at reasoning effort low.
    LLM outputs are not deterministic: repeated identical prompts change a user's AP by about 0.06 on average (see the
    candidate-order experiment).</li>
  </ul>
</section>

<footer><p>Regenerate with <code>python run_llm_demographics.py</code> and
<code>python run_llm_demographics.py --describe-model gpt-4o</code> (model calls; cached responses are reused), then
<code>python make_demographics_report.py</code>. Per-user scores are in <code>results/llm/demographics_per_user.csv.gz</code>;
descriptions and usernames are in <code>data/llm_demographics/</code>, which is not tracked by git.</p></footer>
</div>
<script>{JS}</script>
</body></html>
"""
(RESULTS_DIR / "LLM_DEMOGRAPHICS_REPORT.html").write_text(page, encoding="utf-8")
print("wrote", RESULTS_DIR / "LLM_DEMOGRAPHICS_REPORT.html")
print("examples:", examples)
print("status:", m["status"].value_counts().to_dict())
print("agreement:", [(a[0], a[1], round(a[2], 3), round(a[3], 3)) for a in agreement])
print("tau:", tau_by_status.round(3).to_dict())
