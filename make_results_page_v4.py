"""
Build results/results_v4.html: results_v3.html without the organizations and likely bots among the users of
Tables 9-11 and Section 5 (make_excluded_users.py), and with a fifth account-name condition: 25 names drawn
from the k = 50 list, which separates the size of the list from where its accounts come from.

    python make_excluded_users.py
    python run_llm_demographics_v2.py --exclude results/excluded_users.csv   # Section 5 summary, cached calls only
    python make_results_page_v4.py

results_v3.html (Figure 1 on the full-scale setup of Table 6, a demographics column in Tables 9 and 11,
significance tests in Figure 4):

    python run_figure1_bands.py      # Figure 1 data (SocialVec only, no LLM calls)

Earlier differences (results_v2.html), from make_results_page.py:

Differences from make_results_page.py: Table 1 and Figure 1 report the 10 categories that have a
ranks 181-200 list (with a random baseline, and without the closed-world setup in Figure 1);
Section 3 removes the category's extended-list entities from the profiles too
(run_popularity_band.py --exclude-extended) and leads with the all-users setup; Section 5 uses
the second demographics experiment (run_llm_demographics_v2.py).
"""
import ast
import base64
import gzip
import html
import json
import pickle
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from reproduction import paper_results as paper
from reproduction.config import CATEGORY_LABELS, MODELS_DIR, PAPER_CATEGORY_ORDER, ROOT
from reproduction.data import load_categories
from reproduction.plots import plot_figure1
from reproduction.popularity_bands import expected_random_ap, load_category_sheets, popularity_band
from reproduction.page_style import CSS, JS
from run_llm_experiments import by_category

R = ROOT / "results"
LLM = R / "llm"
# organizations and likely bots among the users of Tables 9-11 and Section 5, left out of those results (make_excluded_users.py)
EXCLUDED_USERS = pd.read_csv(R / "excluded_users.csv")
EXCLUDED = set(EXCLUDED_USERS["user"])
BAND = R / "popularity_band_81_100_excl_extended"  # profiles also exclude the extended-list entities
# Table 1 and Figure 1 report the 10 categories that have a ranks 181-200 list; *_all keep all 14 for the paper checks
CATS = [c for c in PAPER_CATEGORY_ORDER if c in set(pd.read_csv(R / "popularity_band_181_200_extended" / "band_entities.csv")["category"])]
t2_all = pd.read_csv(R / "table2.csv", index_col=0)
fig1_all = pd.read_csv(R / "figure1.csv", index_col=0)
_t1_users = pd.read_csv(R / "per_user_ap.csv.gz", usecols=["topic", "n_relevant"])
_t1_users = _t1_users[_t1_users["topic"].isin(CATS)]
_n_cand = {t: len(c) for t, c in load_categories().items()}
_t1_users["random"] = [expected_random_ap(_n_cand[t], n) for t, n in zip(_t1_users["topic"], _t1_users["n_relevant"])]
t2 = t2_all.loc[CATS].copy()
t2["random"] = _t1_users.groupby("topic")["random"].mean()
t2.loc["Overall"] = t2.loc[CATS].mean()
for _c in ("se1_all_legacy", "se1_all", "se2_all"):
    t2.loc["Overall", f"{_c}_gain"] = t2.loc["Overall", _c] / t2.loc["Overall", "popularity"] - 1
fig1 = fig1_all.loc[CATS].copy()
fig1.loc["Overall"] = fig1.loc[CATS].mean()
# Figure 1 (v3): the matched user-category pairs of Table 6, both bands, same profiles (run_figure1_bands.py)
F1_KS = ["10", "20", "30", "40", "50", "80", "100", "300", "500", "5000"]
F1_BANDS = [("head", "Ranks 1–20", "s1"), ("band", "Ranks 81–100", "s2")]
_f1 = pd.read_csv(R / "figure1_bands" / "per_pair_ap.csv.gz")
F1_COLS = ["random", "popularity"] + [f"k{k}" for k in F1_KS] + ["all"]
f1b = {}
for _c, _g in _f1.groupby("candidates"):
    _t = _g.groupby("category")[F1_COLS].mean()
    _t = _t.reindex([c for c in PAPER_CATEGORY_ORDER if c in _t.index])
    _t.loc["Overall"] = _t.mean()
    f1b[_c] = _t
f1_pairs = len(_f1[["category", "user"]].drop_duplicates())
f1_users = _f1["user"].nunique()
F1_CATS = [c for c in f1b["head"].index if c != "Overall"]


def plot_figure1_bands(path):
    """Per-category panels of Figure 1: both bands' sampling curves and popularity lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = F1_CATS + ["Overall"]
    xs = F1_KS[:-1] + ["All"]
    colors = {"head": "#2a78d6", "band": "#eb6834"}
    fig, axes = plt.subplots(2, 5, figsize=(21, 6.6), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        for c, label, _ in F1_BANDS:
            row = f1b[c].loc[panel]
            ax.plot(xs, [row[f"k{k}"] for k in F1_KS[:-1]] + [row["all"]], marker="o", ms=3.5, color=colors[c], label=f"SocialVec, {label.lower()}")
            ax.axhline(row["popularity"], color=colors[c], linestyle=":", linewidth=1.4, label=f"Popularity, {label.lower()}")
        ax.set_title(CATEGORY_LABELS.get(panel, panel), fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.4)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0.015, 0.1, 1, 1))
    fig.text(0.5, 0.065, "Accounts sampled per user", ha="center", fontsize=12)
    fig.text(0.004, 0.55, "MAP", va="center", rotation=90, fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


plot_figure1_bands(R / "figure1_v3.png")
grid = pd.read_csv(R / "table4_per_category.csv", index_col=0)
pooled = pd.read_csv(R / "table4_pooled.csv", index_col=0)
mA = {c: pd.read_csv(BAND / f"matched_category_{c}.csv", index_col=0) for c in ("head", "band")}
mB = {c: pd.read_csv(BAND / f"matched_all_{c}.csv", index_col=0) for c in ("head", "band")}
band_entities = pd.read_csv(BAND / "band_entities.csv")
from reproduction.data import load_users
_all_users = load_users(load_categories())
_follows = np.maximum(pd.to_numeric(_all_users["friends_count"]).fillna(0), _all_users["friends_ids"].apply(len))
_entities = _all_users["experiment_accounts"].apply(len)


def follow_stats(rows):
    """Minimum, mean, median and maximum of accounts followed and of category entities followed, for some users."""
    f, e = _follows.loc[rows], _entities.loc[rows]
    return (f"accounts followed: minimum {f.min():,.0f}, mean {f.mean():,.0f}, median {f.median():,.0f}, maximum {f.max():,.0f}; "
            f"candidate entities followed: minimum {e.min():,}, mean {e.mean():.1f}, median {e.median():.0f}, maximum {e.max():,}")


per_user = pd.read_csv(R / "per_user_ap.csv.gz", usecols=["topic", "n_followed", "n_followed_entities", "n_closed_world", "n_relevant"])
users_per_category = per_user.groupby("topic").size()

# Figure 2 agreement with the notebook values
profiles = pd.read_csv(R / "entity_follower_profiles.csv", index_col=0)
sheets = pd.read_excel(ROOT / "data" / "popular_accounts_manually_validated.xlsx", sheet_name=None)
diffs = []
for title, ref in paper.FIGURE2.items():
    sheet = sheets[paper.FIGURE2_SHEETS[title]]
    sheet = sheet[sheet["use"] == 1]
    lookup = {s.lower(): int(i) for s, i in zip(sheet["twitter_screen_name"], sheet["twitter_user_id"])}
    ids = [int(i) for i in sheet["twitter_user_id"]]
    for name, values in ref.items():
        ours = (profiles.loc[profiles.index.intersection(ids), paper.RADAR_TRAITS].mean() if name == "ALL"
                else profiles.loc[lookup[name.lower()], paper.RADAR_TRAITS])
        diffs.extend(np.abs(ours.values.astype(float) - np.array(values)))
radar_mean, radar_max = float(np.mean(diffs)), float(np.max(diffs))
n_entities = int(sum((s["use"] == 1).sum() for s in sheets.values()))

# Users behind Table B, and SocialVec popularity rank of the band candidates
band_users = pd.read_csv(BAND / "per_user_ap.csv.gz")
band_users = band_users[(band_users["population"] == "all") & (band_users["exclusion"] == "exclude_category")]
pairs = {c: set(zip(g["category"], g["user"])) for c, g in band_users.groupby("candidates")}
shared = pairs["head"] & pairs["band"]
n_distinct_users = len({u for _, u in shared})

with gzip.open(MODELS_DIR / "SocialVec2020.pkl.gz", "rb") as f:
    key_to_index = pickle.load(f).wv.key_to_index  # vocabulary is sorted by follow frequency
sv_rank = np.array([key_to_index[str(i)] + 1 for i, ok in zip(band_entities["twitter_user_id"], band_entities["in_socialvec"]) if ok])
n_band_in_sv, n_below_10k = len(sv_rank), int((sv_rank > 10_000).sum())


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def esc(s):
    return html.escape(str(s))


def pct(x):
    s = f"{x:+.0%}"
    return s.replace("-", "−")


def pval(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def signed_short(v):
    return f"{v:+.3f}".replace("-", "−")


def p_text(p):
    """A p-value for running text: '< 0.001' or '= 0.006'."""
    return pval(p) if p < 0.001 else f"= {p:.3f}"


# ---------------------------------------------------------------- table numbering
# results.html numbers its tables 1, 2, ...; results_backup.html (extended category lists) B1, B2, ...
MAIN_TABLES = {
    "ranking": "Entity ranking (MAP), paper Table 2",
    "fig1_data": "Overall MAP by number of sampled accounts, ranks 1–20 and 81–100 (data of Figure 1)",
    "llm": "LLM personalization per category, paper Table 3",
    "grid": "Categories × entities per category, paper Table 4",
    "band_sampled": "Ranks 81–100 vs. top 20, users sampled for each category",
    "band_all": "Ranks 81–100 vs. top 20, all experiment users",
    "band_entities": "Candidate entities, ranks 81–100",
    "popularity": "Global popularity of the candidates",
    "band_llm": "LLM and SocialVec ranking of ranks 81–100",
    "band_compare": "LLM and SocialVec ranking, ranks 1–20 vs. 81–100 (same users)",
    "top_llm": "LLM and SocialVec ranking of ranks 1–20, per category",
    "order_personal": "Candidate order, personalized prompts (k = 25)",
    "order_baseline": "Candidate order, non-personalized baseline",
    "prompt_orders": "Candidate order in the LLM prompts of this page",
    "demo_users": "The users of the demographics experiment",
    "demo_bands": "Overall MAP of demographics-only and name-based ranking, ranks 1–20 and 81–100",
    "demo_attrs": "Demographic attributes stated in GPT-4o's descriptions",
    "demo_attrs_cat": "Demographic attributes stated in GPT-4o's descriptions, by category",
    "demo_examples": "Examples: known entities a user follows and GPT-4o's description",
    "haiku_1_20": "Supplementary · Claude Haiku 5.5 ranking per category, ranks 1–20",
    "haiku_81_100": "Supplementary · Claude Haiku 5.5 ranking per category, ranks 81–100",
    "haiku_complete": "Supplementary · Completeness of the rankings, Claude Haiku 5.5 and GPT-6 Luna",
    "k25_decomp": "Account names by source and number",
}
BACKUP_TABLES = {
    "bands_overview": "Overall MAP by popularity band",
    "band2_sampled": "Ranks 181–200 vs. top 20, users sampled for each category",
    "band2_all": "Ranks 181–200 vs. top 20, all experiment users",
    "band2_entities": "Candidate entities, ranks 181–200 of the extended lists",
    "band2_llm": "LLM and SocialVec ranking of ranks 181–200",
    "popularity_ext": "Global popularity of the candidate bands, including ranks 181–200",
    "c200_stats": "Candidates ranked per answer, 200-candidate prompts",
    "c200_cat_median": "Median candidates ranked per answer, by category",
    "c200_subset": "20 vs. 200 candidates, on the ranks 181–200 candidates",
    "c200_category": "Ranking all 200 candidates, per category",
}


def tnum(key):
    """'Table 5' or 'Table B3'."""
    if key in MAIN_TABLES:
        return f"Table {list(MAIN_TABLES).index(key) + 1}"
    return f"Table B{list(BACKUP_TABLES).index(key) + 1}"


def ttitle(key):
    return f"{tnum(key)} · {(MAIN_TABLES | BACKUP_TABLES)[key]}"


# ---------------------------------------------------------------- charts
def line_chart():
    """Overall MAP vs number of sampled accounts (Figure 1): one curve and one dotted popularity line per band."""
    xs = F1_KS[:-1] + ["All"]
    W, H, L, Rm, T, B = 760, 380, 56, 24, 16, 44
    series = [(cls, label, [f1b[c].loc["Overall", f"k{k}"] for k in F1_KS[:-1]] + [f1b[c].loc["Overall", "all"]],
               f1b[c].loc["Overall", "popularity"]) for c, label, cls in F1_BANDS]
    vals = [v for _, _, vs, p in series for v in vs + [p]]
    y0, y1 = np.floor((min(vals) - 0.02) / 0.05) * 0.05, np.ceil((max(vals) + 0.02) / 0.05) * 0.05
    px = lambda i: L + i * (W - L - Rm) / (len(xs) - 1)
    py = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Overall MAP versus number of sampled entities">']
    for v in np.arange(y0, y1 + 1e-9, 0.05):
        parts.append(f'<line class="grid" x1="{L}" x2="{W-Rm}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="tick" x="{L-8}" y="{py(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    for i, x in enumerate(xs):
        parts.append(f'<text class="tick" x="{px(i):.1f}" y="{H-B+20}" text-anchor="middle">{x}</text>')
    parts.append(f'<text class="axis-title" x="{(L+W-Rm)/2}" y="{H-6}" text-anchor="middle">Accounts sampled per user</text>')
    for cls, label, _, pop in series:
        yp = py(pop)
        parts.append(f'<line x1="{L}" x2="{W-Rm}" y1="{yp:.1f}" y2="{yp:.1f}" style="stroke:var(--{cls});stroke-width:1.6;stroke-dasharray:2 4"/>'
                     f'<text class="ref-label" x="{W-Rm-4}" y="{yp-6:.1f}" text-anchor="end">Popularity, {label.lower()} {pop:.3f}</text>')
    for cls, label, vs, _ in series:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vs))
        parts.append(f'<polyline class="line {cls}" points="{pts}"/>')
    for cls, label, vs, _ in series:
        for i, v in enumerate(vs):
            parts.append(f'<circle class="dot {cls}" cx="{px(i):.1f}" cy="{py(v):.1f}" r="4"/>')
    for cls, label, vs, _ in series:
        for i, v in enumerate(vs):
            parts.append(f'<circle class="hit" cx="{px(i):.1f}" cy="{py(v):.1f}" r="11" data-tip="{esc(f"SocialVec, {label.lower()}<br>k = {xs[i]}: <b>{v:.3f}</b>")}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def bar_panel(table, title):
    """Horizontal grouped bars: popularity vs social per category, random as a tick."""
    rows = [c for c in table.index if c != "Overall"] + ["Overall"]
    W, L, Rm, T, rowh = 520, 118, 56, 30, 34
    H = T + rowh * len(rows) + 28
    x1 = 0.85
    px = lambda v: L + v / x1 * (W - L - Rm)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    for v in [0, 0.2, 0.4, 0.6, 0.8]:
        parts.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{T-6}" y2="{H-26}"/>'
                     f'<text class="tick" x="{px(v):.1f}" y="{H-10}" text-anchor="middle">{v:.1f}</text>')
    parts.append(f'<text class="col-head" x="{W-4}" y="{T-12}" text-anchor="end">Δ</text>')
    for r, cat in enumerate(rows):
        t = table.loc[cat]
        y = T + r * rowh
        label = CATEGORY_LABELS.get(cat, "Overall")
        weight = ' font-weight="600"' if cat == "Overall" else ""
        if cat == "Overall":
            parts.append(f'<line class="sep" x1="8" x2="{W-4}" y1="{y-5}" y2="{y-5}"/>')
        parts.append(f'<text class="cat" x="{L-10}" y="{y+15}" text-anchor="end"{weight}>{esc(label)}</text>')
        for j, (cls, key, name) in enumerate([("s2", "popularity", "Popularity"), ("s1", "socialvec", "Social similarity")]):
            v = t[key]
            yy = y + 2 + j * 12
            w = px(v) - L
            tip = f"{esc(label)} · {name}<br>MAP <b>{v:.3f}</b>"
            parts.append(f'<path class="bar {cls}" d="M{L},{yy} h{w-3:.1f} a3,3 0 0 1 3,3 v4 a3,3 0 0 1 -3,3 h-{w-3:.1f} z" />'
                         f'<rect class="hit" x="{L}" y="{yy-1}" width="{max(w, 6):.1f}" height="12" data-tip="{tip}"/>')
        xr = px(t["random"])
        tip = f"{esc(label)} · Random ranking<br>expected MAP <b>{t['random']:.3f}</b>"
        parts.append(f'<line class="rand" x1="{xr:.1f}" x2="{xr:.1f}" y1="{y-1}" y2="{y+27}"/>'
                     f'<rect class="hit" x="{xr-5:.1f}" y="{y-1}" width="10" height="28" data-tip="{tip}"/>')
        parts.append(f'<text class="delta{" neg" if t["gain"] < 0 else ""}" x="{W-4}" y="{y+15}" text-anchor="end"{weight}>{pct(t["gain"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- tables
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def heatmap(df, caption):
    lo, hi = 0.465, 0.525
    rows = []
    for k, row in df.iterrows():
        cells = []
        for v in row:
            step = int(np.clip((v - lo) / (hi - lo) * (len(RAMP) - 1), 0, len(RAMP) - 1))
            ink = "#ffffff" if step >= 7 else "#0b0b0b"
            cells.append(f'<td style="background:{RAMP[step]};color:{ink}">{v:.3f}</td>')
        rows.append(f"<tr><th>{k}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    return (f'<figure class="heat"><figcaption>{caption}</figcaption><table class="heatmap">'
            f'<thead><tr><th class="corner">k \\ n</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></figure>')


def with_paper(ours, ref, fmt="{:.3f}"):
    return f'<span class="num">{fmt.format(ours)}</span><span class="paper">paper {fmt.format(ref)}</span>'


def table2_html():
    rows = []
    for cat in CATS + ["Overall"]:
        o = t2.loc[cat]
        if cat == "Overall":  # the paper's per-category values averaged over the same categories
            ref = {"n_relevant": np.nan, **paper.TABLE2.loc[CATS].mean().to_dict()}
        else:
            ref = paper.TABLE2.loc[cat]
        ent = "" if cat == "Overall" else with_paper(o["n_relevant"], ref["n_relevant"], "{:.1f}")
        cls = ' class="total"' if cat == "Overall" else ""
        neg = lambda g: ' class="neg"' if g < 0 else ""
        rows.append(
            f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th>"
            f"<td>{len(_t1_users) if cat == 'Overall' else users_per_category[cat]:,}</td><td>{ent}</td>"
            f"<td>{o['random']:.3f}</td><td>{with_paper(o['popularity'], ref['popularity'])}</td>"
            f"<td>{with_paper(o['se1_all_legacy'], ref['all'])}</td><td{neg(o['se1_all_legacy_gain'])}>{pct(o['se1_all_legacy_gain'])}</td>"
            f"<td>{with_paper(o['se2_all'], ref['sampled'])}</td><td{neg(o['se2_all_gain'])}>{pct(o['se2_all_gain'])}</td>"
            f"<td><span class='num'>{o['se1_all']:.3f}</span></td><td{neg(o['se1_all_gain'])}>{pct(o['se1_all_gain'])}</td></tr>")
    return ("<div class='scroll'><table class='data'><thead><tr><th>Category</th><th>Users</th><th>Entities / user</th><th>Random</th><th>Popularity</th>"
            "<th>Social · All</th><th>Δ</th><th>Social · Sampled</th><th>Δ</th><th>All, target removed</th><th>Δ</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def matched_html(tables, band_label="Ranks 81–100"):
    head, band = tables["head"], tables["band"]
    rows = []
    for cat in band.index:
        h, b = head.loc[cat], band.loc[cat]
        rel = lambda v: "" if pd.isna(v) else f"{v:.1f}"
        neg = lambda g: ' class="neg"' if g < 0 else ""
        cls = ' class="total"' if cat == "Overall" else ""
        below = ' class="below"' if b["popularity"] < b["random"] else ""
        rows.append(
            f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th><td>{b['users']:,.0f}</td>"
            f"<td>{rel(h['relevant_per_user'])}</td><td>{h['random']:.3f}</td><td>{h['popularity']:.3f}</td><td>{h['socialvec']:.3f}</td>"
            f"<td{neg(h['gain'])}>{pct(h['gain'])}</td><td class='p'>{pval(h['p_value'])}</td>"
            f"<td class='grp'>{rel(b['relevant_per_user'])}</td><td>{b['random']:.3f}</td><td{below}>{b['popularity']:.3f}</td>"
            f"<td>{b['socialvec']:.3f}</td><td{neg(b['gain'])}>{pct(b['gain'])}</td><td class='p'>{pval(b['p_value'])}</td></tr>")
    return ("<div class='scroll'><table class='data'><thead>"
            f"<tr class='super'><th></th><th></th><th colspan='6'>Ranks 1–20 (original candidates)</th><th colspan='6' class='grp'>{band_label}</th></tr>"
            "<tr><th>Category</th><th>Users</th><th>Rel./user</th><th>Random</th><th>Popularity</th><th>Social</th><th>Δ</th><th>p</th>"
            "<th class='grp'>Rel./user</th><th>Random</th><th>Popularity</th><th>Social</th><th>Δ</th><th>p</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


LLM_COLS = ["llm_baseline", "llm_k12", "llm_k25", "llm_k50"]
LLM_LABELS = ["Baseline", "k = 12", "k = 25", "k = 50"]
SV_COLS = ["socialvec_k12", "socialvec_k25", "se1_legacy_k50"]


def table3_html():
    """GPT-6 Luna on the 700-user sample next to the paper's GPT-4o values."""
    rows = []
    for cat in PAPER_CATEGORY_ORDER + ["Overall"]:
        r = llm_t3.loc[cat]
        ref = paper.TABLE3.loc[cat] if cat != "Overall" else paper.TABLE3.mean()
        cls = " class='total'" if cat == "Overall" else ""
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th><td>{r['popularity']:.3f}</td>"
                    + "".join(f"<td>{r[c]:.3f}</td>" for c in SV_COLS)
                    + "".join(f"<td{' class=grp' if i == 0 else ''}>{r[c]:.3f}</td>" for i, c in enumerate(LLM_COLS))
                    + "".join(f"<td class='muted{' grp' if i == 0 else ''}'>{ref[c]:.3f}</td>" for i, c in enumerate(LLM_COLS)) + "</tr>")
    gains = "".join(f"<td>{pct(llm_gain[c])}</td>" for c in LLM_COLS[1:])
    paper_gains = "".join(f"<td class='muted'>{pct(paper.TABLE3[c].mean() / paper.TABLE3['llm_baseline'].mean() - 1)}</td>" for c in LLM_COLS[1:])
    rows.append(f"<tr><th>Δ vs. LLM baseline</th><td></td><td></td><td></td><td></td><td class='grp'></td>{gains}<td class='grp'></td>{paper_gains}</tr>")
    return ("<div class='scroll'><table class='data'><thead><tr class='super'><th></th><th colspan='4'>Same 700 users</th>"
            f"<th colspan='4' class='grp'>{esc(model_name)} (700 users)</th><th colspan='4' class='grp'>GPT-4o (paper, all users)</th></tr>"
            "<tr><th>Category</th><th>Popularity</th><th>Social, k = 12</th><th>Social, k = 25</th><th>Social, k = 50</th>"
            + "".join(f"<th{' class=grp' if i == 0 else ''}>{l}</th>" for i, l in enumerate(LLM_LABELS)) * 2
            + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def llm_chart():
    """Overall MAP per LLM condition: the rerun vs. the paper, with popularity and social references."""
    W, H, L, Rm, T, B = 760, 300, 56, 150, 26, 40
    y0, y1 = 0.40, 0.65
    py = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    group_w = (W - L - Rm) / len(LLM_COLS)
    bw, gap = 24, 2
    o = llm_t3.loc["Overall"]
    ours = [o[c] for c in LLM_COLS]
    theirs = [paper.TABLE3[c].mean() for c in LLM_COLS]
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Overall MAP per LLM condition">']
    for v in np.arange(y0, y1 + 1e-9, 0.05):
        parts.append(f'<line class="grid" x1="{L}" x2="{W-Rm}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="tick" x="{L-8}" y="{py(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    for name, v in [("Popularity", o["popularity"]), ("Social, k = 50", o["se1_legacy_k50"])]:
        parts.append(f'<line class="ref" x1="{L}" x2="{W-Rm}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="ref-label" x="{W-Rm+8}" y="{py(v)+4:.1f}">{name} {v:.3f}</text>')
    for g, label in enumerate(LLM_LABELS):
        cx = L + group_w * (g + 0.5)
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{H-B+20}" text-anchor="middle">{label}</text>')
        for j, (cls, vals, who) in enumerate([("s1", ours, model_name), ("s2", theirs, "GPT-4o (paper)")]):
            v = vals[g]
            x = cx - bw - gap / 2 if j == 0 else cx + gap / 2
            top, base = py(v), py(y0)
            h = base - top
            parts.append(f'<path class="bar {cls}" d="M{x:.1f},{base:.1f} v-{h-4:.1f} a4,4 0 0 1 4,-4 h{bw-8} a4,4 0 0 1 4,4 v{h-4:.1f} z"/>'
                         f'<rect class="hit" x="{x:.1f}" y="{top:.1f}" width="{bw}" height="{h:.1f}" data-tip="{esc(who)} · {label}<br>MAP <b>{v:.3f}</b>"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def order_html():
    s = order_st
    rows = [
        ("MAP", [f"{s.loc['MAP', c]:.3f}" for c in ("original", "repeat", "shuffled")]),
        ("p vs. original run", ["", pval(s.loc['p vs original (Wilcoxon)', 'repeat']), pval(s.loc['p vs original (Wilcoxon)', 'shuffled'])]),
        ("Mean per-user |ΔAP| vs. original run", ["", f"{s.loc['mean |ΔAP| vs original', 'repeat']:.3f}", f"{s.loc['mean |ΔAP| vs original', 'shuffled']:.3f}"]),
        ("Kendall τ: output vs. presented order", [f"{s.loc['τ with presented order', c]:.3f}" for c in ("original", "repeat", "shuffled")]),
        ("Kendall τ: output vs. popularity order", [f"{s.loc['τ with popularity order', c]:.3f}" for c in ("original", "repeat", "shuffled")]),
        ("Kendall τ: output vs. original-run output", ["1", f"{s.loc['τ with original-run output', 'repeat']:.3f}", f"{s.loc['τ with original-run output', 'shuffled']:.3f}"]),
    ]
    body = "".join(f"<tr><th>{esc(n)}</th>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>" for n, vals in rows)
    b = base_st
    brows = [
        ("Overall MAP, mean of 10 calls", [f"{b.loc[c, 'overall MAP (mean over trials)']:.3f}" for c in ("original", "shuffled")]),
        ("SD across calls", [f"{b.loc[c, 'SD across trials']:.3f}" for c in ("original", "shuffled")]),
        ("Kendall τ: output vs. presented order", [f"{b.loc[c, 'τ with presented order']:.3f}" for c in ("original", "shuffled")]),
        ("Kendall τ: output vs. popularity order", [f"{b.loc[c, 'τ with popularity order']:.3f}" for c in ("original", "shuffled")]),
        ("Distinct rankings among 10 calls", [f"{b.loc[c, 'distinct rankings per category (of 10)']:.1f}" for c in ("original", "shuffled")]),
    ]
    bbody = "".join(f"<tr><th>{esc(n)}</th>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>" for n, vals in brows)
    return (
        f"<div class='panels'><div class='panel'><h4>{ttitle('order_personal')} (700 users)</h4><div class='scroll'><table class='data compact'>"
        "<thead><tr><th>Measure</th><th>Original order</th><th>Repeated</th><th>Shuffled</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div></div>"
        f"<div class='panel'><h4>{ttitle('order_baseline')} (14 categories × 10 calls)</h4><div class='scroll'><table class='data compact'>"
        "<thead><tr><th>Measure</th><th>Original order</th><th>Shuffled</th></tr></thead>"
        f"<tbody>{bbody}</tbody></table></div></div></div>")


def band_llm_html(table=None):
    table = band_llm if table is None else table
    cols = ["random", "popularity", "socialvec", "socialvec_k12", "socialvec_k25"] + BAND_LLM_COLS
    group_start = {"socialvec", "llm_baseline"}
    rows = []
    for cat, r in table.iterrows():
        cls = " class='total'" if cat == "Overall" else ""
        best = max(r[c] for c in cols)
        cells = "".join(f"<td{' class=grp' if c in group_start else ''}>{'<b>' if r[c] == best else ''}{r[c]:.3f}{'</b>' if r[c] == best else ''}</td>" for c in cols)
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th><td>{r['users']:,.0f}</td>{cells}</tr>")
    return ("<div class='scroll'><table class='data'><thead><tr class='super'><th></th><th></th><th colspan='2'>Non-personalized</th>"
            f"<th colspan='3' class='grp'>SocialVec</th><th colspan='7' class='grp'>{esc(model_name)}</th></tr>"
            "<tr><th>Category</th><th>Users</th><th>Random</th><th>Popularity</th>"
            "<th class='grp'>All entities</th><th>k = 12</th><th>k = 25</th><th class='grp'>Baseline</th><th>Demo&shy;graphics</th><th>k = 12</th><th>k = 25</th><th>k = 12 of 50</th><th>k = 25 of 50</th><th>k = 50</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def table1d_html():
    rows = []
    for k in F1_KS:
        rows.append(f"<tr><th>{k}</th>" + "".join(f"<td>{f1b[c].loc['Overall', f'k{k}']:.3f}</td>" for c, _, _ in F1_BANDS) + "</tr>")
    rows.append("<tr class='total'><th>All</th>" + "".join(f"<td>{f1b[c].loc['Overall', 'all']:.3f}</td>" for c, _, _ in F1_BANDS) + "</tr>")
    for col, label in [("popularity", "Popularity"), ("random", "Random (expected)")]:
        rows.append(f"<tr><th>{label}</th>" + "".join(f"<td>{f1b[c].loc['Overall', col]:.3f}</td>" for c, _, _ in F1_BANDS) + "</tr>")
    return (f"<details open><summary>{ttitle('fig1_data')}</summary><div class='scroll'><table class='data compact'><thead><tr><th>k</th>"
            + "".join(f"<th>{label}</th>" for _, label, _ in F1_BANDS) + "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>")


# Real names for accounts whose Twitter display name is a nickname, a joke or oddly cased (keyed by handle)
ENTITY_NAMES = pd.read_csv(ROOT / "reproduction" / "entity_names.csv").set_index("twitter_screen_name")["name"].to_dict()


def entity_name(display, handle):
    """The mapped real name, else the display name without emoji and other symbols, else the handle."""
    if handle in ENTITY_NAMES:
        return ENTITY_NAMES[handle]
    if not isinstance(display, str):
        return handle
    clean = "".join(ch for ch in display if unicodedata.category(ch) not in ("So", "Sk", "Cf") and ch != "️")
    return re.sub(r"\s+", " ", clean).strip() or handle


def entities_html(entities=None, key="band_entities"):
    entities = band_entities if entities is None else entities
    rows = []
    for cat, g in entities.groupby("category", sort=False):
        # entity names, with the handle on hover
        names = ", ".join(f"{esc(entity_name(n, h))} <span class='muted'>(@{esc(h)})</span>" + ("<sup>*</sup>" if not v else "")
                          for n, h, v in zip(g["twitter_name"], g["twitter_screen_name"], g["in_socialvec"]))
        rng = f"{g['followers_count'].min()/1e3:,.0f}K – {g['followers_count'].max()/1e3:,.0f}K"
        rows.append(f"<tr><th>{esc(CATEGORY_LABELS[cat])}</th><td class='nowrap'>{rng}</td><td class='names'>{names}</td></tr>")
    return (f"<details open><summary>{ttitle(key)} (entity name with its Twitter handle; * no SocialVec embedding, excluded)</summary><div class='scroll'><table class='data compact'>"
            "<thead><tr><th>Category</th><th>Followers</th><th>Entities</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>")


# ---------------------------------------------------------------- numbers for prose
ov = t2.loc["Overall"]
max_t2 = np.abs(t2_all.loc[PAPER_CATEGORY_ORDER, ["popularity", "se1_all_legacy", "se2_all"]].values
                - paper.TABLE2.loc[PAPER_CATEGORY_ORDER, ["popularity", "all", "sampled"]].values).max()
d_grid = np.abs(grid.values - paper.TABLE4.values).mean()
d_pool = np.abs(pooled.values - paper.TABLE4.values).mean()
A_h, A_b = mA["head"].loc["Overall"], mA["band"].loc["Overall"]
B_h, B_b = mB["head"].loc["Overall"], mB["band"].loc["Overall"]
below = [CATEGORY_LABELS[c] for c in mB["band"].index if c != "Overall" and mB["band"].loc[c, "popularity"] < mB["band"].loc[c, "random"]]
_bB = mB["band"].drop("Overall")
sv_wins_B = int((_bB["socialvec"] > _bB["popularity"]).sum())
sv_losses_B = [CATEGORY_LABELS[c] for c in _bB.index if _bB.at[c, "socialvec"] <= _bB.at[c, "popularity"]]
sv_above_random_B = int((_bB["socialvec"] > _bB["random"]).sum())

fig1_png, fig2_png = b64(R / "figure1_v3.png"), b64(R / "figure2.png")

# LLM experiments (run_llm_experiments.py)
def use_shuffled_baseline(table, per_category):
    """Replace a table's LLM baseline (candidates in list order) by its shuffled-candidate version, per category."""
    table = table.copy()
    cats = [c for c in table.index if c != "Overall"]
    table.loc[cats, "llm_baseline"] = per_category.reindex(cats).values
    table.loc["Overall", "llm_baseline"] = table.loc[cats, "llm_baseline"].mean()
    return table


# Every LLM baseline in the tables uses shuffled candidates; the list-order version partly copies the presented order.
_trials = pd.read_csv(LLM / "baseline_trials_scored.csv")  # 10 shuffled calls per category, scored on the 700 users
# personalized prompts with the candidates in one random order per user and condition (run_shuffled_personal.py)
_t3_users = pd.read_csv(LLM / "main_per_user.csv.gz", usecols=["user", "topic", "popularity", "socialvec_k12", "socialvec_k25", "se1_legacy_k50"])
_t3_users = _t3_users.merge(pd.read_csv(LLM / "main_shuffled_per_user.csv.gz", usecols=["user", "llm_k12", "llm_k25", "llm_k50"]), on="user")
_t3_users["llm_baseline"] = np.nan
llm_t3 = use_shuffled_baseline(by_category(_t3_users, ["popularity", "socialvec_k12", "socialvec_k25", "se1_legacy_k50"] + LLM_COLS),
                               _trials[_trials["order_type"] == "shuffled"].groupby("topic")["MAP"].mean())
llm_gain = {c: llm_t3.loc["Overall", c] / llm_t3.loc["Overall", "llm_baseline"] - 1 for c in ["llm_k12", "llm_k25", "llm_k50"]}
order_st = pd.read_csv(LLM / "order_stats.csv", index_col=0)
base_st = pd.read_csv(LLM / "baseline_trial_stats.csv", index_col=0)
# 200 users per category (198 for TV stations), the first 30 of which are the users of Section 5 (run_band_more_users.py)
_band_users = pd.read_csv(LLM / "band_per_user_n200.csv.gz")
_band_n_before = len(_band_users)
_band_users = _band_users[~_band_users["user"].isin(EXCLUDED)]
_band_users["llm_baseline"] = _band_users["llm_baseline_shuffled"]  # one random candidate order per user, shared by all conditions
for _k in (12, 25, 50):
    _band_users[f"llm_k{_k}"] = _band_users[f"llm_k{_k}_shuffled"]  # run_shuffled_personal.py
_demo_pu = pd.read_csv(LLM / "demographics_v2_per_user.csv.gz", usecols=["user", "band", "ap_gpt-4o", "ap_k12of50", "ap_k25of50"]).rename(
    columns={"ap_gpt-4o": "llm_demo", "ap_k12of50": "llm_k12of50", "ap_k25of50": "llm_k25of50"})
_band_users = _band_users.merge(_demo_pu[_demo_pu["band"] == "81-100"].drop(columns="band"), on="user", how="left")
assert _band_users["llm_demo"].notna().all()
BAND_LLM_COLS = ["llm_baseline", "llm_demo", "llm_k12", "llm_k25", "llm_k12of50", "llm_k25of50", "llm_k50"]
band_llm = by_category(_band_users, ["random", "popularity", "socialvec", "socialvec_k12", "socialvec_k25"] + BAND_LLM_COLS)
band_tests = {f"llm_k{k} vs socialvec_k{k}": wilcoxon(_band_users[f"llm_k{k}"] - _band_users[f"socialvec_k{k}"]).pvalue for k in (12, 25)}
band_llm_n = {k: _band_users[f"n_k{k}_accounts"].mean() for k in (12, 25)}
demo_n_k25 = band_llm_n[25]
_band_cats = band_llm.drop("Overall")
excl_counts = EXCLUDED_USERS["reason"].value_counts().reindex(
    ["organization", "possible organization", "bot: very active", "bot: 2+ weak signals"], fill_value=0).to_dict()


def word_list(items):
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


band_llm_sv_better = [CATEGORY_LABELS[c].lower().replace("tv ", "TV ") for c in _band_cats.index if _band_cats.at[c, "socialvec"] > _band_cats.at[c, "llm_k12"]]
band_llm_llm_better = [CATEGORY_LABELS[c].lower().replace("tv ", "TV ") for c in _band_cats.index if _band_cats.at[c, "socialvec"] <= _band_cats.at[c, "llm_k12"]]
band_demo_better = [CATEGORY_LABELS[c].lower().replace("tv ", "TV ") for c in _band_cats.index if _band_cats.at[c, "llm_demo"] > _band_cats.at[c, "llm_k12"]]
band_llm_wins = {k: int((band_llm.drop("Overall")[f"llm_k{k}"] > band_llm.drop("Overall")[f"socialvec_k{k}"]).sum()) for k in (12, 25)}

# The same protocol on the top-20 candidates, for the users of Table 9 who follow at least one of them
_top_users = pd.read_csv(LLM / "band_per_user_1_20_original_n200_users_81_100_original.csv.gz")
_top_users = _top_users[~_top_users["user"].isin(EXCLUDED)]
_top_users["llm_baseline"] = _top_users["llm_baseline_shuffled"]
for _k in (12, 25, 50):
    _top_users[f"llm_k{_k}"] = _top_users[f"llm_k{_k}_shuffled"]
_top_users = _top_users.merge(_demo_pu[_demo_pu["band"] == "1-20"].drop(columns="band"), on="user", how="left")
assert _top_users["llm_demo"].notna().all()
top_llm = by_category(_top_users, ["random", "popularity", "socialvec", "socialvec_k12", "socialvec_k25"] + BAND_LLM_COLS)
top_tests = {f"llm_k{k} vs socialvec_k{k}": wilcoxon(_top_users[f"llm_k{k}"] - _top_users[f"socialvec_k{k}"]).pvalue for k in (12, 25)}
top_tests["llm_k50 vs socialvec"] = wilcoxon(_top_users["llm_k50"] - _top_users["socialvec"]).pvalue
band_tests["llm_k50 vs socialvec"] = wilcoxon(_band_users["llm_k50"] - _band_users["socialvec"]).pvalue
top_llm_wins = {k: int((top_llm.drop("Overall")[f"llm_k{k}"] > top_llm.drop("Overall")[f"socialvec_k{k}"]).sum()) for k in (12, 25)}
BAND_METHODS = [("random", "Random ranking (expected)"), ("popularity", "Popularity (follower count)"),
                ("socialvec", "SocialVec, all followed entities"), ("socialvec_k12", "SocialVec, the 12 accounts of the LLM prompt"),
                ("socialvec_k25", "SocialVec, the 25 accounts of the LLM prompt"), ("llm_baseline", "LLM, no user information"), ("llm_demo", "LLM, demographics by GPT-4o"),
                ("llm_k12", "LLM, 12 followed accounts"), ("llm_k25", "LLM, 25 followed accounts"),
                ("llm_k12of50", "LLM, 12 accounts drawn from the 50-account list"),
                ("llm_k25of50", "LLM, 25 accounts drawn from the 50-account list"), ("llm_k50", "LLM, 50 followed accounts")]


def band_compare_html():
    """Overall MAP per method at ranks 1-20 and 81-100, same users and protocol."""
    rows = [f"<tr><th>Users</th><td>{top_llm.loc['Overall', 'users']:,.0f}</td><td>{band_llm.loc['Overall', 'users']:,.0f}</td><td></td></tr>"]
    for c, label in BAND_METHODS:
        a, b = top_llm.loc["Overall", c], band_llm.loc["Overall", c]
        rows.append(f"<tr><th>{esc(label)}</th><td>{a:.3f}</td><td>{b:.3f}</td><td>{b - a:+.3f}</td></tr>")
    for name, label in [("llm_k12 vs socialvec_k12", "p, LLM vs. SocialVec, same 12 accounts"),
                        ("llm_k25 vs socialvec_k25", "p, LLM vs. SocialVec, same 25 accounts"),
                        ("llm_k50 vs socialvec", "p, LLM 50 accounts vs. SocialVec all entities")]:
        rows.append(f"<tr class='muted'><th>{label}</th><td>{p_text(top_tests[name])}</td><td>{p_text(band_tests[name])}</td><td></td></tr>")
    return ("<div class='scroll'><table class='data compact'><thead><tr><th>Method</th><th>Ranks 1–20</th><th>Ranks 81–100</th>"
            f"<th>Difference</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


# Ranks 181-200 of the extended lists (build_extended_categories.py, run_popularity_band.py --source extended)
BAND2 = R / "popularity_band_181_200_extended"
mA2 = {c: pd.read_csv(BAND2 / f"matched_category_{c}.csv", index_col=0) for c in ("head", "band")}
mB2 = {c: pd.read_csv(BAND2 / f"matched_all_{c}.csv", index_col=0) for c in ("head", "band")}
band2_entities = pd.read_csv(BAND2 / "band_entities.csv")
A2_h, A2_b = mA2["head"].loc["Overall"], mA2["band"].loc["Overall"]
B2_h, B2_b = mB2["head"].loc["Overall"], mB2["band"].loc["Overall"]
below2 = [CATEGORY_LABELS[c] for c in mA2["band"].index if c != "Overall" and mA2["band"].loc[c, "popularity"] < mA2["band"].loc[c, "random"]]
_shuffled181 = pd.read_csv(LLM / "band_shuffled_baseline_181_200_extended.csv.gz")  # run_band_shuffled_baseline.py
band2_llm = use_shuffled_baseline(pd.read_csv(LLM / "band_by_category_181_200_extended.csv", index_col=0),
                                  _shuffled181.groupby("topic")["llm_baseline_shuffled"].mean())
band2_tests = pd.read_csv(LLM / "band_tests_181_200_extended.csv", index_col=0).iloc[:, 0]
_band2_users = pd.read_csv(LLM / "band_per_user_181_200_extended.csv.gz", usecols=["n_k12_accounts", "n_k25_accounts"])
band2_llm_n = {k: _band2_users[f"n_k{k}_accounts"].mean() for k in (12, 25)}
band2_llm_wins = {k: int((band2_llm.drop("Overall")[f"llm_k{k}"] > band2_llm.drop("Overall")[f"socialvec_k{k}"]).sum()) for k in (12, 25)}
_extended = pd.read_excel(ROOT / "data" / "popular_accounts_extended.xlsx", sheet_name=None)
_band2_sv = pd.concat([s.loc[(s["rank_in_category"] > 180) & (s["rank_in_category"] <= 200), "socialvec_popularity_rank"]
                       for name, s in _extended.items() if name != "summary" and len(s) >= 200])
band2_sv_median, band2_below_10k = int(_band2_sv.median()), float((_band2_sv > 10_000).mean())

# global popularity of the candidate bands: rank among all SocialVec accounts (vocabulary sorted by follow frequency)
POP_BANDS = [("Original ranks 1–20", "s1"), ("Original ranks 81–100", "s2"), ("Extended ranks 181–200", "s3")]
_head = pd.concat([c.assign(category=t) for t, c in load_categories().items()])
pop = pd.concat([
    _head.assign(band=POP_BANDS[0][0]),
    band_entities.assign(band=POP_BANDS[1][0]),
    band2_entities.assign(band=POP_BANDS[2][0]),
])[["category", "band", "twitter_name", "twitter_user_id", "followers_count"]]
pop["global_rank"] = [key_to_index.get(str(i), np.nan) + 1 for i in pop["twitter_user_id"]]
pop_cats = [c for c in PAPER_CATEGORY_ORDER if c in set(band_entities["category"]) | set(band2_entities["category"])]
pop = pop[pop["category"].isin(pop_cats)]
pop_ranked = pop.dropna(subset=["global_rank"])
pop_n_vocab = len(key_to_index)


def pop_stats(g):
    r = g["global_rank"].dropna()
    return {"n": len(g), "in_sv": len(r), "median": r.median(), "q1": r.quantile(0.25), "q3": r.quantile(0.75),
            "top1k": (r <= 1_000).mean(), "top10k": (r <= 10_000).mean(), "followers": g["followers_count"].median()}


pop_overall = {b: pop_stats(pop[pop["band"] == b]) for b, _ in POP_BANDS}
pop_by_cat = {(c, b): pop_stats(g) for (c, b), g in pop.groupby(["category", "band"])}
_p81, _p181 = (pop_ranked.loc[pop_ranked["band"] == POP_BANDS[i][0], "global_rank"] for i in (1, 2))
pop_mw_p = mannwhitneyu(_p81, _p181).pvalue
pop_more_popular_cats = sum(pop_by_cat[(c, POP_BANDS[2][0])]["median"] < pop_by_cat[(c, POP_BANDS[1][0])]["median"]
                            for c in pop_cats if (c, POP_BANDS[1][0]) in pop_by_cat)
pop_both_cats = sum((c, POP_BANDS[1][0]) in pop_by_cat for c in pop_cats)


def _log_x(L, W, Rm):
    span = np.log10(pop_n_vocab)
    return lambda v: L + np.log10(v) / span * (W - L - Rm)


POP_TICKS = [1, 10, 100, 1_000, 10_000, 100_000]
STRIP_BANDS = POP_BANDS[:2]  # results.html: original lists only
main_pop_cats = [c for c in pop_cats if c in set(band_entities["category"])]  # categories with a ranks 81-100 band
pop_overall_main = {b: pop_stats(pop[(pop["band"] == b) & pop["category"].isin(main_pop_cats)]) for b, _ in STRIP_BANDS}


def pop_strip_chart():
    """Global SocialVec rank of every candidate, one row per category and band (log scale)."""
    W, L, Rm, T, sub, gap = 760, 130, 16, 8, 13, 12
    rowh = sub * len(STRIP_BANDS) + gap
    H = T + rowh * len(main_pop_cats) + 26
    px = _log_x(L, W, Rm)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Global popularity rank of the candidates by category">']
    for v in POP_TICKS:
        parts.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{T}" y2="{H-22}"/>'
                     f'<text class="tick" x="{px(v):.1f}" y="{H-6}" text-anchor="middle">{v:,}</text>')
    for i, cat in enumerate(main_pop_cats):
        y0 = T + i * rowh
        parts.append(f'<text class="cat" x="{L-10}" y="{y0 + sub * len(STRIP_BANDS) / 2 + 4:.1f}" text-anchor="end">{esc(CATEGORY_LABELS[cat])}</text>')
        for j, (band, cls) in enumerate(STRIP_BANDS):
            g = pop_ranked[(pop_ranked["category"] == cat) & (pop_ranked["band"] == band)]
            y = y0 + j * sub + sub / 2
            for _, e in g.iterrows():
                parts.append(f'<circle class="dot {cls}" cx="{px(e["global_rank"]):.1f}" cy="{y:.1f}" r="3.2" fill-opacity="0.55" '
                             f'data-tip="{esc(e["twitter_name"])} · {esc(band)}<br>global rank <b>{int(e["global_rank"]):,}</b> · '
                             f'{e["followers_count"]:,.0f} followers"/>')
            if len(g):
                m = px(g["global_rank"].median())
                parts.append(f'<line class="line {cls}" x1="{m:.1f}" x2="{m:.1f}" y1="{y-6:.1f}" y2="{y+6:.1f}" style="stroke-width:3"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def pop_cdf_chart():
    """Share of each band's candidates within the global top N (log scale)."""
    W, H, L, Rm, T, B = 520, 250, 46, 16, 10, 40
    px = _log_x(L, W, Rm)
    py = lambda v: T + (1 - v) * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Cumulative share of candidates by global popularity rank">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line class="grid" x1="{L}" x2="{W-Rm}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="tick" x="{L-6}" y="{py(v)+4:.1f}" text-anchor="end">{v:.0%}</text>')
    for v in POP_TICKS:
        parts.append(f'<text class="tick" x="{px(v):.1f}" y="{H-B+18}" text-anchor="middle">{v:,}</text>')
    parts.append(f'<text class="tick" x="{(L + W - Rm) / 2:.1f}" y="{H-4}" text-anchor="middle">global popularity rank (log scale)</text>')
    for band, cls in POP_BANDS:
        r = np.sort(pop.loc[pop["band"] == band, "global_rank"].fillna(pop_n_vocab).values)  # not in SocialVec: at the end
        pts = [(px(1), py(0))] + [(px(v), py((k + 1) / len(r))) for k, v in enumerate(r)]
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<path class="line {cls}" d="{path}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


# global rank of the accounts at fixed positions of each original sheet sorted by follower count (the order behind ranks 81-100)
POSITIONS = [1, 20, 81, 100]
_sheets = load_category_sheets()
pos_rank = {}
for _cat in pop_cats:
    if _cat in _sheets and len(_sheets[_cat]) >= max(POSITIONS):
        _ranked = popularity_band(_sheets[_cat], 1, max(POSITIONS))
        for _p in POSITIONS:
            _e = _ranked.iloc[_p - 1]
            pos_rank[(_cat, _p)] = (key_to_index.get(str(_e["twitter_user_id"]), np.nan) + 1, _e["twitter_screen_name"])


def pop_table(bands, overall, positions=()):
    """
    Median global rank, share in the global top 10,000 and median followers per category, for the given bands;
    optionally also the global rank of the accounts at the given positions of the follower-count order.
    """
    labels = [b.split("ranks ")[-1] for b in bands]
    cats = [c for c in pop_cats if any((c, b) in pop_by_cat for b in bands[1:])]  # categories with a lower band
    rows = []
    for cat in cats + ["Overall"]:
        get = (lambda b: overall[b]) if cat == "Overall" else (lambda b: pop_by_cat.get((cat, b)))

        def cells(key, fmt):
            out = [f"<td>{fmt.format(get(b)[key])}</td>" if get(b) else "<td>–</td>" for b in bands]
            out[0] = out[0].replace("<td>", "<td class=grp>", 1)
            return "".join(out)
        label = "Overall" if cat == "Overall" else CATEGORY_LABELS[cat]
        cls = " class='total'" if cat == "Overall" else ""
        pos_cells = ""
        for i, p in enumerate(positions):
            rank, name = pos_rank.get((cat, p), (np.nan, ""))
            value = "" if cat == "Overall" else ("not in SocialVec" if pd.isna(rank) else f"{rank:,.0f}")
            pos_cells += (f"<td{' class=grp' if i == 0 else ''}>{value}"
                          f"{f'<span class=paper>{esc(name)}</span>' if name else ''}</td>")
        rows.append(f"<tr{cls}><th>{esc(label)}</th>" + pos_cells + cells("median", "{:,.0f}") + cells("top10k", "{:.0%}")
                    + cells("followers", "{:,.0f}") + "</tr>")
    n = len(bands)
    head = "".join(f"<th{' class=grp' if i == 0 else ''}>{p}</th>" for i, p in enumerate(positions))
    head += "".join(f"<th{' class=grp' if i == 0 else ''}>{esc(l)}</th>" for i, l in enumerate(labels)) * 3
    pos_group = f"<th colspan='{len(positions)}' class='grp'>Global rank at category position</th>" if positions else ""
    return ("<div class='scroll'><table class='data compact'><thead><tr class='super'><th></th>"
            f"{pos_group}<th colspan='{n}' class='grp'>Median global rank</th><th colspan='{n}' class='grp'>In the global top 10,000</th>"
            f"<th colspan='{n}' class='grp'>Median followers</th></tr><tr><th>Category</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")
band2_cost = sum(r["cost_usd"] for r in json.loads((LLM / "run_info.json").read_text()) if r.get("band") == [181, 200, "extended"])
_b2 = pd.read_csv(LLM / "band_per_user_181_200_extended.csv.gz", usecols=["llm_k50", "socialvec"])
k50_vs_full = wilcoxon(_b2["llm_k50"] - _b2["socialvec"]).pvalue


def bands_overview():
    """Overall MAP for the top 20 and each lower band, on each band's matched users (Table A populations)."""
    rows = [("Ranks 1–20, users of the 81–100 table", A_h), ("Ranks 81–100", A_b),
            ("Ranks 1–20, users of the 181–200 table", A2_h), ("Ranks 181–200", A2_b)]
    body = "".join(
        f"<tr><th>{label if label.startswith('Ranks 1–20') else f'<b>{label}</b>'}</th><td>{r['users']:,.0f}</td><td>{r['random']:.3f}</td>"
        f"<td{' class=below' if r['popularity'] < r['random'] else ''}>{r['popularity']:.3f}</td><td>{r['socialvec']:.3f}</td>"
        f"<td>{pct(r['gain'])}</td></tr>" for label, r in rows)
    return ("<div class='scroll'><table class='data compact'><thead><tr><th>Candidates</th><th>Users</th><th>Random</th>"
            "<th>Popularity</th><th>Social</th><th>Δ</th></tr></thead>" f"<tbody>{body}</tbody></table></div>")
llm_runs = [r for r in json.loads((LLM / "run_info.json").read_text()) if r["calls"] > 0]
llm_calls, llm_cost = sum(r["calls"] for r in llm_runs), sum(r["cost_usd"] for r in llm_runs)
model_name = {"gpt-6-luna": "GPT-6 Luna"}.get(llm_runs[0]["model"], llm_runs[0]["model"])
llm_effort = llm_runs[0]["reasoning_effort"]
lo = llm_t3.loc["Overall"]


def legacy_parses(text):
    try:
        return isinstance(ast.literal_eval(str(text).strip()), list)
    except (ValueError, SyntaxError):
        return False


llm_main = pd.read_csv(LLM / "main_per_user.csv.gz", usecols=[f"resp_{c}" for c in LLM_COLS[1:]])
legacy_fail = 1 - np.mean([legacy_parses(t) for c in llm_main.columns for t in llm_main[c]])

# demographics-only LLM ranking (headline numbers written by make_demographics_report.py)
demo = json.loads((LLM / "demographics_v2_summary_excluded.json").read_text())  # run_llm_demographics_v2.py --exclude
dsh = {b: demo["shuffled"][b] for b in ("1-20", "81-100")}
DEMO_METHODS = [("random", "Random ranking (expected)"), ("popularity", "Popularity (follower count)"),
                ("ap_baseline", "LLM, no user information"),
                ("ap_gpt-4o", "LLM, demographics by GPT-4o"), ("ap_k12", "LLM, 12 followed account names"),
                ("ap_k25", "LLM, 25 followed account names"), ("ap_k12of50", "LLM, 12 names from the 50-name list"),
                ("ap_k25of50", "LLM, 25 names from the 50-name list"),
                ("ap_k50", "LLM, 50 followed account names"),
                ("socialvec", "SocialVec, all followed entities")]
DEMO_COLORS = {"ap_gpt-4o": "s2", "ap_k12": "s3", "ap_k25": "s3", "ap_k12of50": "s3", "ap_k25of50": "s3", "ap_k50": "s3"}


def method_chart(o, methods, colors, context, x1=0.6, marks=None):
    """Horizontal bars of overall MAP per method; methods in `colors` are highlighted; `marks` adds text after a value."""
    marks = marks or {}
    W, L, Rm, T, rowh = (600, 230, 130, 6, 28) if marks else (520, 230, 50, 6, 28)
    H = T + rowh * len(methods) + 28
    px = lambda v: L + v / x1 * (W - L - Rm)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Overall MAP by method, {context}">']
    for v in np.arange(0, x1 + 1e-9, 0.1):
        parts.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{T}" y2="{H-22}"/>'
                     f'<text class="tick" x="{px(v):.1f}" y="{H-6}" text-anchor="middle">{v:.1f}</text>')
    for i, (c, label) in enumerate(methods):
        y, v = T + i * rowh, o[c]
        w = px(v) - L
        weight = ' font-weight="650"' if c in colors else ""
        parts.append(f'<text class="cat" x="{L-10}" y="{y+16}" text-anchor="end"{weight}>{esc(label)}</text>'
                     f'<path class="bar {colors.get(c, "muted-bar")}" d="M{L},{y+5} h{w-4:.1f} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 h-{w-4:.1f} z"/>'
                     f'<text class="delta" x="{px(v)+8:.1f}" y="{y+16}"{weight}>{v:.3f}'
                     + (f'<tspan dx="8" font-weight="400">{esc(marks[c])}</tspan>' if c in marks else "") + '</text>'
                     f'<rect class="hit" x="{L}" y="{y+3}" width="{w:.1f}" height="18" data-tip="{esc(label)}, {context}<br>MAP <b>{v:.3f}</b>"/>')
    parts.append("</svg>")
    return "\n".join(parts)


HAIKU = "Claude Haiku 5.5"
_v2_users = pd.read_csv(LLM / "demographics_v2_per_user.csv.gz", usecols=["user", "band", "topic", "random", "popularity", "socialvec"])
_hk_users = pd.read_csv(LLM / "demographics_v2_claude-haiku-5-5_per_user.csv.gz").merge(
    _v2_users.drop(columns="topic"), on=["user", "band"])  # run_llm_claude.py
_hk_users = _hk_users[~_hk_users["user"].isin(EXCLUDED)]


def haiku_summary(b):
    methods = [c for c, _ in DEMO_METHODS]
    return {"users": len(b), "map": b.groupby("topic")[methods].mean().mean().to_dict(),
            "tests": {name: wilcoxon(b[x] - b[y]).pvalue for name, x, y in [
                ("4o_vs_baseline", "ap_gpt-4o", "ap_baseline"), ("baseline_vs_popularity", "ap_baseline", "popularity"),
                ("4o_vs_socialvec", "ap_gpt-4o", "socialvec"), ("k12_vs_4o", "ap_k12", "ap_gpt-4o"),
                ("k50_vs_socialvec", "ap_k50", "socialvec")]}}


dhk = {b: haiku_summary(_hk_users[_hk_users["band"] == b]) for b in ("1-20", "81-100")}
_luna_users = pd.read_csv(LLM / "demographics_v2_per_user.csv.gz")
_luna_users = _luna_users[~_luna_users["user"].isin(EXCLUDED)]
_hk_runs = [r for r in json.loads((LLM / "run_info.json").read_text()) if r["model"] == "claude-haiku-5-5"]
hk_calls, hk_cost = sum(r["calls"] for r in _hk_runs), sum(r["cost_usd"] for r in _hk_runs)
HK_CONDS = [("ap_baseline", "No user information"), ("ap_gpt-4o", "Demographics by GPT-4o"), ("ap_k12", "12 account names"),
            ("ap_k25", "25 account names"), ("ap_k12of50", "12 names from the 50"), ("ap_k25of50", "25 names from the 50"), ("ap_k50", "50 account names")]


def haiku_category_table(band):
    """Supplementary: Claude Haiku 5.5 MAP per category and condition, with the non-LLM methods and GPT-6 Luna's overall row."""
    b = _hk_users[_hk_users["band"] == band]
    cols = ["random", "popularity", "socialvec"] + [c for c, _ in HK_CONDS]
    t = b.groupby("topic")[cols].mean()
    t = t.reindex([c for c in PAPER_CATEGORY_ORDER if c in t.index])
    t.insert(0, "users", b.groupby("topic").size())
    t.loc["Overall"] = t.mean()
    t.loc["Overall", "users"] = len(b)
    luna = _luna_users[_luna_users["band"] == band].groupby("topic")[[c for c, _ in HK_CONDS]].mean().mean()
    grp = {"socialvec", "ap_baseline"}
    rows = []
    for cat, r in t.iterrows():
        cls = " class='total'" if cat == "Overall" else ""
        best = max(r[c] for c in cols)
        cells = "".join(f"<td{' class=grp' if c in grp else ''}>{'<b>' if r[c] == best else ''}{r[c]:.3f}{'</b>' if r[c] == best else ''}</td>"
                        for c in cols)
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th><td>{r['users']:,.0f}</td>{cells}</tr>")
    rows.append(f"<tr class='muted'><th>{esc(model_name)}, overall</th><td></td><td></td><td></td><td class='grp'></td>"
                + "".join(f"<td{' class=grp' if c == 'ap_baseline' else ''}>{luna[c]:.3f}</td>" for c, _ in HK_CONDS) + "</tr>")
    return ("<div class='scroll'><table class='data'><thead><tr class='super'><th></th><th></th><th colspan='2'>Non-personalized</th>"
            f"<th class='grp'>SocialVec</th><th colspan='5' class='grp'>{HAIKU}</th></tr>"
            "<tr><th>Category</th><th>Users</th><th>Random</th><th>Popularity</th><th class='grp'>All entities</th>"
            + "".join(f"<th{' class=grp' if c == 'ap_baseline' else ''}>{esc(label)}</th>" for c, label in HK_CONDS)
            + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def haiku_completeness_table():
    """Supplementary: share of answers that rank every candidate, at least 95% of them, or none, per model, band and condition."""
    n_cand = _luna_users.set_index(["user", "band"])["n_candidates"]
    rows = []
    for name, users in [(HAIKU, _hk_users), (model_name, _luna_users)]:
        for band in ("1-20", "81-100"):
            u = users[users["band"] == band].set_index(["user", "band"])
            cells = []
            for c, _ in HK_CONDS:
                share = u[c.replace("ap_", "returned_")] / n_cand.reindex(u.index)
                cells.append(f"<td class='grp'>{(share >= 1).mean():.1%}</td><td>{(share >= 0.95).mean():.1%}</td><td>{(share == 0).mean():.1%}</td>")
            rows.append(f"<tr><th>{esc(name)}</th><td>Ranks {band.replace('-', '–')}</td>{''.join(cells)}</tr>")
    head = "".join(f"<th colspan='3' class='grp'>{esc(label)}</th>" for _, label in HK_CONDS)
    sub = "<th class='grp'>All</th><th>≥ 95%</th><th>None</th>" * len(HK_CONDS)
    return ("<div class='scroll'><table class='data compact'><thead>"
            f"<tr class='super'><th></th><th></th>{head}</tr><tr><th>Model</th><th>Band</th>{sub}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# Paired Wilcoxon tests behind the markers of Figure 4: each personalized condition vs. no information (*)
# and each account-name condition vs. demographics (†), per model and band
SIG_PAIRS = [("ap_gpt-4o", "ap_baseline"), ("ap_k12", "ap_baseline"), ("ap_k25", "ap_baseline"), ("ap_k12of50", "ap_baseline"),
             ("ap_k25of50", "ap_baseline"), ("ap_k50", "ap_baseline"), ("ap_k12", "ap_gpt-4o"), ("ap_k25", "ap_gpt-4o"),
             ("ap_k12of50", "ap_gpt-4o"), ("ap_k25of50", "ap_gpt-4o"), ("ap_k50", "ap_gpt-4o"), ("ap_k50", "socialvec"),
             ("ap_k12of50", "ap_k12"), ("ap_k25of50", "ap_k25"), ("ap_k25", "ap_k12"), ("ap_k25of50", "ap_k12of50"),
             ("ap_k50", "ap_k25of50")]
MARKED = [pair for pair in SIG_PAIRS if pair[1] in ("ap_baseline", "ap_gpt-4o")]  # the comparisons marked in Figure 4
_luna_f = _luna_users[(_follows.reindex(_luna_users["user"]).values < demo["filter"]["max_follows"])
                      & (_entities.reindex(_luna_users["user"]).values >= demo["filter"]["min_entities"])]


def sig_tests(users):
    return {b: {(x, y): wilcoxon(g[x] - g[y]).pvalue for x, y in SIG_PAIRS}
            for b, g in users.groupby("band")}


SIG = {"luna": sig_tests(_luna_users), "haiku": sig_tests(_hk_users), "luna_f": sig_tests(_luna_f)}
for _b in ("1-20", "81-100"):  # the filtered users and maps must match the summary
    assert _luna_f[_luna_f["band"] == _b].shape[0] == demo["shuffled_filtered"][_b]["users"], _b
    assert abs(_luna_users[_luna_users["band"] == _b].groupby("topic")["ap_gpt-4o"].mean().mean() - dsh[_b]["map"]["ap_gpt-4o"]) < 1e-4


def stars(p, sym="*"):
    return sym * 3 if p < 0.001 else sym * 2 if p < 0.01 else sym if p < 0.05 else "n.s."


def sig_marks(key, band):
    """Marker text per method: * vs. no information, † vs. demographics."""
    t = SIG[key][band]
    marks = {"ap_gpt-4o": stars(t[("ap_gpt-4o", "ap_baseline")])}
    for k in ("ap_k12", "ap_k25", "ap_k12of50", "ap_k25of50", "ap_k50"):
        marks[k] = f"{stars(t[(k, 'ap_baseline')])} {stars(t[(k, 'ap_gpt-4o')], '†')}"
    return marks


sig_all = all(SIG[key][b][pair] < 0.001 for key in ("luna", "haiku") for b in ("1-20", "81-100") for pair in MARKED)
n_marked = 4 * len(MARKED)


def demographics_chart(band, summary=None, model=None):
    """Overall MAP per method with shuffled candidates, for GPT-6 Luna (default) or another ranking model."""
    summary = dsh if summary is None else summary
    context = f"ranks {band}, shuffled candidates" + (f", {model}" if model else "")
    marks = sig_marks("haiku" if model else "luna", band)
    return method_chart(summary[band]["map"], DEMO_METHODS, DEMO_COLORS, context, x1=0.8, marks=marks)


def demographics_table(bands=("1-20", "81-100")):
    """The numbers behind Figure 4: overall MAP per method and band, for all users and for the filtered users."""
    cols = [(dsh, b) for b in bands] + [(dhk, b) for b in bands] + [(dshf, b) for b in bands]
    starts = {len(bands), 2 * len(bands)}
    head = (f"<tr class='super'><th></th><th colspan='2'>{esc(model_name)}, all users</th>"
            f"<th colspan='2' class='grp'>{HAIKU}, all users</th>"
            f"<th colspan='2' class='grp'>{esc(model_name)}, &lt; {demo['filter']['max_follows']:,} follows, ≥ {demo['filter']['min_entities']} entities</th></tr>"
            "<tr><th>Method</th>" + "".join(f"<th{' class=grp' if i in starts else ''}>Ranks {b.replace('-', '–')}</th>"
                                             for i, (_, b) in enumerate(cols)) + "</tr>")
    cell = lambda i, text: f"<td{' class=grp' if i in starts else ''}>{text}</td>"
    rows = ["<tr><th>Users</th>" + "".join(cell(i, f"{d[b]['users']:,}") for i, (d, b) in enumerate(cols)) + "</tr>"]
    for c, label in DEMO_METHODS:
        rows.append(f"<tr><th>{esc(label)}</th>" + "".join(cell(i, f"{d[b]['map'][c]:.3f}") for i, (d, b) in enumerate(cols)) + "</tr>")
    sig_cols = [SIG[key][b] for key in ("luna", "haiku", "luna_f") for b in bands]
    for pair, label in [(("ap_gpt-4o", "ap_baseline"), "p, demographics vs. no information"),
                        (("ap_k12", "ap_baseline"), "p, 12 names vs. no information"),
                        (("ap_k25", "ap_baseline"), "p, 25 names vs. no information"),
                        (("ap_k12of50", "ap_baseline"), "p, 12 of the 50 names vs. no information"),
                        (("ap_k25of50", "ap_baseline"), "p, 25 of the 50 names vs. no information"),
                        (("ap_k50", "ap_baseline"), "p, 50 names vs. no information"),
                        (("ap_k12", "ap_gpt-4o"), "p, 12 names vs. demographics"),
                        (("ap_k25", "ap_gpt-4o"), "p, 25 names vs. demographics"),
                        (("ap_k12of50", "ap_gpt-4o"), "p, 12 of the 50 names vs. demographics"),
                        (("ap_k25of50", "ap_gpt-4o"), "p, 25 of the 50 names vs. demographics"),
                        (("ap_k50", "ap_gpt-4o"), "p, 50 names vs. demographics"),
                        (("ap_k50", "socialvec"), "p, 50 names vs. SocialVec"),
                        (("ap_k12of50", "ap_k12"), "p, 12 of the 50 names vs. 12 names"),
                        (("ap_k25of50", "ap_k25"), "p, 25 of the 50 names vs. 25 names"),
                        (("ap_k50", "ap_k25of50"), "p, 50 names vs. 25 of the 50")]:
        rows.append(f"<tr class='muted'><th>{label}</th>" + "".join(cell(i, p_text(t[pair])) for i, t in enumerate(sig_cols)) + "</tr>")
    return (f"<div class='scroll'><table class='data compact'><thead>{head}</thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def k25_decomposition():
    """Per model and band: MAP with 12 and 25 closed-set names, 12, 25 and 50 names from the k = 50 list, and the steps between them."""
    cols = ["ap_k12", "ap_k25", "ap_k12of50", "ap_k25of50", "ap_k50"]
    steps = {"source12": ("ap_k12of50", "ap_k12"), "source25": ("ap_k25of50", "ap_k25"), "size_closed": ("ap_k25", "ap_k12"),
             "size_full": ("ap_k25of50", "ap_k12of50"), "size_50": ("ap_k50", "ap_k25of50")}
    out = []
    for model, users, key in [(model_name, _luna_users, "luna"), (HAIKU, _hk_users, "haiku")]:
        for band in ("1-20", "81-100"):
            g = users[users["band"] == band]
            m = g.groupby("topic")[cols].mean().mean()
            t = SIG[key][band]
            r = {"model": model, "band": band, "users": len(g), **{c[3:]: m[c] for c in cols}}
            for name, (x, y) in steps.items():
                r[name], r[f"p_{name}"] = m[x] - m[y], t[(x, y)]
            r["share"] = r["source25"] / (r["source25"] + r["size_50"])
            out.append(r)
    return out


k25d = k25_decomposition()
closed_lt25 = (_band_users["n_k25_accounts"] < 25).mean()  # users whose closed-set k = 25 list is short


def k25_decomposition_html():
    def step(r, name):
        return f"<td class='grp'>{r[name]:+.3f}<br><small class='muted'>p {p_text(r['p_' + name])}</small></td>"
    rows = "".join(
        f"<tr><th>{esc(r['model'])}</th><td>Ranks {r['band'].replace('-', '–')}</td><td>{r['users']:,}</td>"
        f"<td class='grp'>{r['k12']:.3f}</td><td>{r['k25']:.3f}</td><td class='grp'>{r['k12of50']:.3f}</td><td>{r['k25of50']:.3f}</td>"
        f"<td>{r['k50']:.3f}</td>" + "".join(step(r, n) for n in ("source12", "source25", "size_closed", "size_full", "size_50"))
        + "</tr>" for r in k25d)
    return ("<div class='scroll'><table class='data compact'><thead><tr class='super'><th></th><th></th><th></th>"
            "<th colspan='2' class='grp'>Closed set</th><th colspan='3' class='grp'>Full follow list (k = 50 prompt)</th>"
            "<th colspan='2' class='grp'>Source: full list − closed set</th><th colspan='3' class='grp'>Size</th></tr>"
            "<tr><th>Model</th><th>Band</th><th>Users</th><th class='grp'>12</th><th>25</th><th class='grp'>12</th><th>25</th><th>50</th>"
            "<th class='grp'>at 12</th><th class='grp'>at 25</th><th class='grp'>12 → 25, closed set</th>"
            "<th class='grp'>12 → 25, full list</th><th class='grp'>25 → 50, full list</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


# The exact chat messages dspy sends for the two stages, with placeholders for the inputs
def demographic_prompts():
    from reproduction.llm_demographics import make_demographic_programs
    _, _, render = make_demographic_programs(None, single_value=True)
    calls = {"describe": dict(followed_accounts="{names of up to 50 accounts the user follows, comma-separated}"),
             "rank": dict(user_description="{the description written by GPT-4o}",
                          potential_accounts="{JSON list of the candidates: twitter_user_id and twitter_name}")}
    out = {}
    for kind, inputs in calls.items():
        msgs = render(kind, **inputs)
        out[kind] = "\n\n".join(f"── {m_['role']} message ──\n" + re.sub(r"(?m)^ {8}", "", m_["content"]).strip() for m_ in msgs)
    return out


demo_prompts = demographic_prompts()
stage2 = demo["stage2_example"]
demo_n_k25of50 = demo["user_stats"]["names_k25of50"]["mean"]  # names in the 25-of-50 prompts


def stage2_example_html():
    """The user message of one real Stage 2 call, the model's answer and its score."""
    from reproduction.llm_demographics import make_demographic_programs
    _, _, render = make_demographic_programs(None, single_value=True)
    user_msg = render("rank", user_description=stage2["description"], potential_accounts=stage2["candidates_json"])[-1]["content"]
    ranked = ", ".join(f"<b>{esc(n)}</b>" if rel else esc(n) for _, n, rel in stage2["ranked"])
    return (f"<pre class=\"prompt\" style=\"white-space:pre-wrap;font-size:12.5px;margin:0\">{esc(user_msg)}</pre>"
            f"<p><b>Answer</b> (the <code>ranked_ids</code> field): <code>{esc(stage2['response'])}</code></p>"
            f"<p><b>The same ranking by name</b> (the user's followed candidates in bold): {ranked}.</p>"
            f"<p>The user follows {len(stage2['relevant'])} of the {len(stage2['ranked'])} candidates; AP = {stage2['ap']:.3f}.</p>")
demo_multiple = sum(a["multiple"] for a in demo["attributes"].values())


def demo_users_html():
    """Follow statistics of the demographics experiment's users."""
    rows = []
    for key, label in [("follows", "Accounts followed"), ("category_entities", "Category entities followed (all 14 lists)"),
                       ("names_given", "Account names given to GPT-4o")]:
        st = demo["user_stats"][key]
        rows.append(f"<tr><th>{label}</th><td>{st['min']:,}</td><td>{st['mean']:,.1f}</td><td>{st['median']:,.0f}</td><td>{st['max']:,}</td></tr>")
    return ("<div class='scroll'><table class='data compact'><thead><tr><th></th><th>Minimum</th><th>Mean</th><th>Median</th>"
            f"<th>Maximum</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def demo_examples_html():
    """A few users: the known entities among the account names GPT-4o saw, and the description it wrote."""
    rows = []
    for e in demo["examples"]:
        names = ", ".join(esc(entity_name(n, h)) for n, h, _ in e["known_entities"])
        rows.append(f"<tr><th>{esc(e['category'])}</th><td class='names'>{names}</td>"
                    f"<td class='nowrap'>{len(e['known_entities'])} of {e['n_names']}</td><td class='names'>{esc(e['description'])}</td></tr>")
    return ("<div class='scroll'><table class='data compact'><thead><tr><th>Sampled for</th>"
            "<th>Known entities among the accounts GPT-4o saw</th><th>Known / all</th>"
            f"<th>GPT-4o's description</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


dshf = demo["shuffled_filtered"]  # the users who pass the follow filter
demo_attrs = demo["attributes"]
SEGMENT_FILLS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--axis)", "var(--ink-2)", "var(--grid)"]


def demo_attrs_chart():
    """One 100% stacked bar per attribute: the share of descriptions stating each value."""
    W, L, Rm, T, rowh = 760, 100, 10, 6, 46
    H = T + rowh * len(demo_attrs)
    total = demo["users"]
    px = lambda v: L + v * (W - L - Rm)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Demographic attributes in GPT-4o descriptions">']
    for i, (attr, a) in enumerate(demo_attrs.items()):
        y = T + i * rowh
        parts.append(f'<text class="cat" x="{L-10}" y="{y+21}" text-anchor="end">{esc(attr)}</text>')
        x = 0.0
        for j, value in enumerate(v for v in a["values"] if a["counts"].get(v)):
            share = a["counts"][value] / total
            fill = SEGMENT_FILLS[j % len(SEGMENT_FILLS)]
            label_fill = "var(--ink)" if fill in ("var(--axis)", "var(--grid)") else "#fff"  # readable on light segments
            parts.append(f'<rect x="{px(x):.1f}" y="{y+4}" width="{px(x + share) - px(x):.1f}" height="26" style="fill:{fill}" stroke="var(--bg)" stroke-width="1"/>')
            if share >= 0.12:
                parts.append(f'<text x="{px(x + share / 2):.1f}" y="{y+21}" text-anchor="middle" style="fill:{label_fill};font-size:12px;font-weight:600">{esc(value)} {share:.0%}</text>')
            parts.append(f'<rect class="hit" x="{px(x):.1f}" y="{y+4}" width="{px(x + share) - px(x):.1f}" height="26" '
                         f'data-tip="{esc(attr)}: {esc(value)}<br><b>{a["counts"][value]}</b> of {total} descriptions ({share:.0%})"/>')
            x += share
    parts.append("</svg>")
    return "\n".join(parts)


def demo_attrs_table():
    total = demo["users"]
    rows = []
    for attr, a in demo_attrs.items():
        values = [v for v in a["values"] if a["counts"].get(v)]
        for k, value in enumerate(values):
            n = a["counts"][value]
            head = f"<th rowspan='{len(values)}'>{esc(attr)}<span class='paper'>{a['multiple']} give more than one value</span></th>" if k == 0 else ""
            rows.append(f"<tr>{head}<td style='text-align:left'>{esc(value)}</td><td>{n}</td><td>{n / total:.0%}</td></tr>")
    return ("<div class='scroll'><table class='data compact'><thead><tr><th>Attribute</th><th>First stated value</th><th>Descriptions</th>"
            f"<th>Share</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


# every age group that occurs in the descriptions; the two most frequent values of the other attributes
DEMO_CAT_COLS = ([("Gender", "Female"), ("Gender", "Male")]
                 + [("Age", v) for v in demo_attrs["Age"]["values"] if demo_attrs["Age"]["counts"].get(v, 0) >= 0.005 * demo["users"]]
                 + [("Ethnicity", "Caucasian / White"), ("Ethnicity", "African American / Black"), ("Education", "High school"),
                    ("Education", "Higher education, level not stated")])
_age_rare = [(v, n) for v, n in demo_attrs["Age"]["counts"].items() if v != "Not stated" and n < 0.005 * demo["users"]]
age_rare_note = ("; " + word_list([f'"{v.lower()}" appears in only {n} descriptions' for v, n in _age_rare]) + " and is left out"
                 if _age_rare else "")


def demo_attrs_cat_table():
    cats = [c for c in PAPER_CATEGORY_ORDER if c in demo_attrs["Gender"]["by_category"]]
    rows = []
    for cat in cats + ["Overall"]:
        def share(attr, value):
            counts = demo_attrs[attr]["counts"] if cat == "Overall" else demo_attrs[attr]["by_category"][cat]
            return counts.get(value, 0) / sum(counts.values())
        cls = " class='total'" if cat == "Overall" else ""
        cells = "".join(f"<td{' class=grp' if i == 0 or DEMO_CAT_COLS[i - 1][0] != attr else ''}>{share(attr, value):.0%}</td>"
                        for i, (attr, value) in enumerate(DEMO_CAT_COLS))
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th>{cells}</tr>")
    groups = []
    for attr in dict.fromkeys(a for a, _ in DEMO_CAT_COLS):
        groups.append(f"<th colspan='{sum(a == attr for a, _ in DEMO_CAT_COLS)}' class='grp'>{esc(attr)}</th>")
    head = "".join(f"<th{' class=grp' if i == 0 or DEMO_CAT_COLS[i - 1][0] != attr else ''}>{esc(value.replace(', level not stated', ''))}</th>"
                   for i, (attr, value) in enumerate(DEMO_CAT_COLS))
    return ("<div class='scroll'><table class='data compact'><thead><tr class='super'><th></th>" + "".join(groups) + "</tr>"
            f"<tr><th>Category</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


# ranking all 200 candidates (run_popularity_band.py --start 1 --end 200 --source extended; run_llm_200.py)
c200 = pd.read_csv(LLM / "candidates_200_per_user.csv.gz")
C200_METHODS = [("random", "Random ranking (expected)"), ("popularity", "Popularity (follower count)"),
                ("llm_baseline", "LLM, no user information"), ("llm_k200", "LLM, up to 200 account names"),
                ("socialvec_k200", "SocialVec, the same accounts"), ("socialvec", "SocialVec, all followed entities")]
C200_COLORS = {"llm_k200": "s3", "socialvec_k200": "s1", "socialvec": "s1"}
c200_cat = c200.groupby("topic")[[c for c, _ in C200_METHODS]].mean()
c200_cat = c200_cat.reindex([c for c in PAPER_CATEGORY_ORDER if c in c200_cat.index])
c200_o = c200_cat.mean()  # unweighted mean over categories
c200_p = {name: wilcoxon(c200[a] - c200[b]).pvalue for name, a, b in [
    ("k200_vs_baseline", "llm_k200", "llm_baseline"), ("k200_vs_sv200", "llm_k200", "socialvec_k200"),
    ("k200_vs_sv", "llm_k200", "socialvec"), ("baseline_vs_popularity", "llm_baseline", "popularity"),
    ("baseline_vs_random", "llm_baseline", "random"), ("sv200_vs_sv", "socialvec_k200", "socialvec")]}
c200_frac = {k: c200[f"returned_{k}"] / c200["n_candidates"] for k in ("baseline", "k200")}
c200_complete = c200[(c200_frac["baseline"] >= 0.95) & (c200_frac["k200"] >= 0.95)]
c200_complete_o = c200_complete.groupby("topic")[["llm_k200", "socialvec_k200"]].mean().mean()
c200_complete_p = wilcoxon(c200_complete["llm_k200"] - c200_complete["socialvec_k200"]).pvalue
c200_wins = int((c200_cat["llm_k200"].round(3) > c200_cat["socialvec_k200"].round(3)).sum())
c200_appended = {k: c200.groupby("topic")[f"llm_{k}_appended"].mean().mean() for k in ("baseline", "k200")}
c200_rel_ranked = {k: c200[f"returned_relevant_{k}"].sum() / c200["n_relevant"].sum() for k in ("baseline", "k200")}
c200_short_list = c200[c200_frac["k200"] < 0.6]  # answers that stop before 60% of the candidates
# the 20-candidate shuffled prompts of Section B3 (ranks 181-200), scored with the same rule
_short = pd.read_csv(LLM / "demographics_bands_per_user.csv.gz")
_short = _short[_short["band"] == "181-200"][["user", "socialvec"]]
c200_sub = c200.dropna(subset=["short_k50_181_200"]).merge(_short, on="user", suffixes=("", "_20"))
C200_SUB_ROWS = [  # label, 200-candidate column, 20-candidate column, what each was given
    ("LLM, no user information", "llm_baseline_181_200", "short_baseline_181_200", "no user information", "no user information"),
    ("LLM, account names", "llm_k200_181_200", "short_k50_181_200", "up to 200 names", "50 names"),
    ("SocialVec", "socialvec_k200_181_200", "socialvec_20", "the LLM's accounts", "all followed entities")]
c200_sub_p = {long: wilcoxon(c200_sub[long] - c200_sub[short]).pvalue for _, long, short, _, _ in C200_SUB_ROWS}
c200_sub_m = {c: c200_sub.groupby("topic")[c].mean().mean() for _, long, short, _, _ in C200_SUB_ROWS for c in (long, short)}
c200_gap_p = {"short": wilcoxon(c200_sub["short_k50_181_200"] - c200_sub["socialvec_20"]).pvalue,
              "long": wilcoxon(c200_sub["llm_k200_181_200"] - c200_sub["socialvec_k200_181_200"]).pvalue}
c200_runs = [r for r in json.loads((LLM / "run_info.json").read_text()) if r.get("parts") == ["candidates-200"]]
c200_cost = sum(r["cost_usd"] for r in c200_runs)


def c200_category_table():
    rows = []
    for cat, r in c200_cat.iterrows():
        best = r.max()
        cells = "".join(f"<td>{'<b>' if r[c] == best else ''}{r[c]:.3f}{'</b>' if r[c] == best else ''}</td>" for c, _ in C200_METHODS)
        rows.append(f"<tr><th>{esc(CATEGORY_LABELS[cat])}</th><td>{(c200['topic'] == cat).sum()}</td>{cells}</tr>")
    rows.append("<tr class='total'><th>Overall</th><td>" + f"{len(c200)}</td>" + "".join(f"<td>{c200_o[c]:.3f}</td>" for c, _ in C200_METHODS) + "</tr>")
    head = "".join(f"<th>{esc(label)}</th>" for _, label in C200_METHODS)
    return f"<div class='scroll'><table class='data compact'><thead><tr><th>Category</th><th>Users</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def c200_subset_table():
    body = "".join(
        f"<tr><th>{esc(label)}</th><td>{c200_sub_m[short]:.3f}<span class='paper'>{esc(short_info)}</span></td>"
        f"<td>{c200_sub_m[long]:.3f}<span class='paper'>{esc(long_info)}</span></td>"
        f"<td>{signed_short(c200_sub_m[long] - c200_sub_m[short])}</td><td>{pval(c200_sub_p[long])}</td></tr>"
        for label, long, short, long_info, short_info in C200_SUB_ROWS)
    return ("<div class='scroll'><table class='data compact'><thead><tr><th>Method</th><th>20-candidate prompt</th>"
            "<th>200-candidate ranking, cut to the same 20</th><th>Δ</th><th>p (Wilcoxon)</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def c200_coverage():
    """How many candidates each answer ranks: per-answer statistics for both prompts."""
    stats = {}
    for k in ("baseline", "k200"):
        n, frac = c200[f"returned_{k}"], c200_frac[k]
        listed = c200[f"resp_{k}"].map(lambda t: re.findall(r"\d+", str(t).split("[[ ##")[0]))  # every id written, repeats included
        repeats = listed.map(len) - listed.map(lambda x: len(set(x)))
        invalid = [len(set(map(int, x)) - set(json.loads(o))) for x, o in zip(listed, c200["order"])]
        stats[k] = {
            "mean": n.mean(), "median": n.median(), "p10": n.quantile(0.1), "min": n.min(),
            "written": listed.map(len).median(), "all": (frac == 1).mean(), "95": (frac >= 0.95).mean(),
            "half": (frac < 0.5).mean(), "repeats": (repeats > 0).mean(), "repeats_mean": repeats.mean(),
            "invalid": (pd.Series(invalid) > 0).mean(), "relevant": c200_rel_ranked[k],
            "missing_user": (c200[f"returned_relevant_{k}"] < c200["n_relevant"]).mean(),
            "by_cat": c200.groupby("topic")[f"returned_{k}"].median(),
        }
    return stats


c200_cov = c200_coverage()


def c200_coverage_table():
    rows = [
        ("Distinct candidates ranked: mean", "mean", "{:.0f}"), ("… median", "median", "{:.0f}"),
        ("… 10th percentile", "p10", "{:.0f}"), ("… minimum", "min", "{:.0f}"),
        ("Ids written per answer, repeats included: median", "written", "{:.0f}"),
        ("Answers ranking every candidate", "all", "{:.0%}"), ("Answers ranking at least 95%", "95", "{:.0%}"),
        ("Answers ranking less than half", "half", "{:.0%}"),
        ("Answers repeating an id", "repeats", "{:.0%}"), ("Repeated ids per answer: mean", "repeats_mean", "{:.1f}"),
        ("Answers with an id that is not a candidate", "invalid", "{:.0%}"),
        ("Relevant candidates ranked (all users)", "relevant", "{:.1%}"),
        ("Users with a relevant candidate left out", "missing_user", "{:.0%}"),
    ]
    body = "".join(f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{fmt.format(c200_cov[k][key])}</td>" for k in ("baseline", "k200")) + "</tr>"
                   for label, key, fmt in rows)
    cats = [c for c in PAPER_CATEGORY_ORDER if c in c200_cov["k200"]["by_cat"].index]
    cat_body = "".join(f"<tr><th>{esc(CATEGORY_LABELS[c])}</th><td>{c200_cov['baseline']['by_cat'][c]:.0f}</td>"
                       f"<td>{c200_cov['k200']['by_cat'][c]:.0f}</td></tr>" for c in cats)
    head = "<thead><tr><th>{}</th><th>No user information</th><th>Up to 200 account names</th></tr></thead>"
    return (f"<h4>{ttitle('c200_stats')} ({len(c200)} users, {c200['n_candidates'].mean():.0f} candidates each)</h4><div class='scroll'>"
            f"<table class='data compact'>{head.format('Statistic')}<tbody>{body}</tbody></table></div>"
            f"<h4>{ttitle('c200_cat_median')}</h4><div class='scroll'>"
            f"<table class='data compact'>{head.format('Category')}<tbody>{cat_body}</tbody></table></div>")



page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Social Knowledge Results</title><style>{CSS}</style></head>
<body><div id="tip" role="tooltip"></div>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">Reproduction &amp; extension · arXiv:2603.10148</p>
  <h1>Social Knowledge for Cross-Domain User Preference Modeling</h1>
  <p class="lede">I re-implemented the paper's experiments from the released data and SocialVec embeddings,
  checked every reported number, and extended the evaluation to less popular entities, where a
  one-size-fits-all popularity ranking has the least to offer.</p>
  <nav class="toc"><a href="#reproduction">1 · Reproduction</a><a href="#legacy">2 · Notes on the original code</a>
  <a href="#tail">3 · Less popular entities</a><a href="#order">4 · LLM candidate order</a>
  <a href="#demographics">5 · Demographics-only LLM</a><a href="#supplement">S · Claude Haiku 5.5</a><a href="#open">6 · Open items</a></nav>
  <div class="tiles">
    <div class="tile"><div class="label">Values matching the paper's Table 2</div><div class="value">42 / 42</div><div class="sub">largest difference {max_t2:.4f} ({tnum('ranking')})</div></div>
    <div class="tile"><div class="label">Overall MAP, social vs. popularity</div><div class="value">{ov['se1_all_legacy']:.3f}</div><div class="sub">vs. {ov['popularity']:.3f} · {pct(ov['se1_all_legacy_gain'])}, as in the paper</div></div>
    <div class="tile"><div class="label">Gain on less popular entities</div><div class="value">{pct(A_b['gain'])}</div><div class="sub">ranks 81–100, vs. {pct(A_h['gain'])} on the top 20 for the same users</div></div>
    <div class="tile"><div class="label">Popularity vs. chance, ranks 81–100</div><div class="value">{signed_short(A_b['popularity']-A_b['random'])}</div><div class="sub">MAP; social similarity is {signed_short(A_b['socialvec']-A_b['random'])}</div></div>
    <div class="tile"><div class="label">LLM gain from 12 favored entities</div><div class="value">{pct(llm_gain['llm_k12'])}</div><div class="sub">{model_name}; GPT-4o in the paper: +13%</div></div>
  </div>
</header>

<section id="reproduction">
  <h2><span class="num">1</span>Reproduction</h2>
  <p class="intro">{len(per_user):,} users who follow more than 10 of the {n_entities} candidate entities,
  14 categories, link-prediction evaluation with MAP. On average {per_user['n_followed_entities'].mean():,.0f} of the accounts a user
  follows are SocialVec entities. Follow statistics of all {len(_all_users):,} users:
  {follow_stats(_all_users.index)}. No user follows more than 5,000 accounts, which looks like a limit of the data collection.</p>

  <h3>{ttitle('ranking')}</h3>
  <p>Every per-category value matches the paper to its three published decimals. The last two columns show the
  protocol as the paper describes it, with the target category's candidates actually removed from the user profile
  (see section 2).</p>
  <div class="card">{table2_html()}
  <p class="caption">The {len(CATS)} categories that also have a ranks 181–200 list (films, food chains, car makers and fashion are left
  out, so that all results cover the same categories). Reproduced values; the paper's value is shown beneath each one, and the Overall
  row averages the paper's values over the same {len(CATS)} categories. Random is the expected MAP of a random ranking. Δ is the gain over popularity.
  Sampled is the closed-world setting, where profiles use only the experiment's entities from the other 13 categories.</p></div>

  <h3>Figure 1 · How much data is needed?</h3>
  <p>Figure 1 uses the full-scale setup of {tnum('band_all')} (Section 3): all {f1_pairs:,} user–category pairs ({f1_users:,} distinct
  users) in which a user follows at least one entity ranked 1–20 and one ranked 81–100 in the {len(F1_CATS)} categories that list 100
  entities. As there, every entity known to belong to the target category is removed from the profile. The profile averages the
  embeddings of k accounts sampled at random from the rest of what the user follows (the paper's SE1 sampling, seed 76), and the
  same profile ranks both candidate sets.</p>
  <ul class="highlights">
    <li><b>Small profiles are enough.</b> With k = 50 sampled accounts SocialVec reaches
    {f1b['head'].loc['Overall','k50']/f1b['head'].loc['Overall','all']:.1%} of its full-profile score on the top 20 and
    {f1b['band'].loc['Overall','k50']/f1b['band'].loc['Overall','all']:.1%} at ranks 81–100; both curves level off beyond about 100 accounts.</li>
    <li><b>Both bands beat popularity from the smallest profile.</b> With k = 10 accounts SocialVec scores
    {f1b['head'].loc['Overall','k10']:.3f} vs. popularity's {f1b['head'].loc['Overall','popularity']:.3f} on the top 20 and
    {f1b['band'].loc['Overall','k10']:.3f} vs. {f1b['band'].loc['Overall','popularity']:.3f} at ranks 81–100; the full profiles score
    {f1b['head'].loc['Overall','all']:.3f} and {f1b['band'].loc['Overall','all']:.3f}.</li>
  </ul>
  <p>All values average the {len(F1_CATS)} categories; the values are in {tnum('fig1_data')}. The full-profile scores equal
  {tnum('band_all')}'s.</p>
  <div class="card">
    <div class="legend"><span class="key"><i style="background:var(--s1)"></i>SocialVec, ranks 1–20</span>
    <span class="key"><i style="background:var(--s2)"></i>SocialVec, ranks 81–100</span>
    <span class="key"><i style="background:none;border-top:2px dotted var(--ink-2);height:0;border-radius:0"></i>Popularity baseline (dotted, same color)</span>
    </div>
    {line_chart()}
    {table1d_html()}
    <details open><summary>Per-category panels</summary><img class="figure" alt="Figure 1 per category" src="data:image/png;base64,{fig1_png}"></details>
  </div>

  <h3>How the LLM prompts are built</h3>
  <p>The personalized prompts use the authors' wording and list the <b>names</b> of accounts the user follows. The names are the
  accounts' Twitter display names, as recorded in the SocialVec account metadata (not their handles); accounts without a recorded name
  are dropped from the list, so a prompt can list fewer accounts than its k. The accounts are drawn at random with a fixed seed
  (the same draw for a user wherever the condition appears) and listed in the order drawn, which is random:</p>
  <ul>
    <li><b>k = 12 and k = 25</b>: drawn from the user's <i>closed set</i>, the candidate entities of the other categories' lists that the user
    follows (the paper's "favored entities"). The target category's own entities are never in this set.</li>
    <li><b>k = 50</b>: drawn from everything the user follows. In {tnum('llm')}, as in the paper, the target category's own candidates can be among
    them (see Section 2); in {tnum('band_llm')}–{tnum('top_llm')} and Section 5, every listed entity of the target category is removed before drawing.</li>
  </ul>
  <p>The candidates to rank are given as a JSON list of ids and names. Where they appear in random order is summarized in {tnum('prompt_orders')} (Section 4).</p>

  <h3>{ttitle('llm')}</h3>
  <p>I re-ran the LLM experiment with {model_name} (reasoning effort <i>{llm_effort}</i>) and the authors' prompts, on a stratified
  random sample of 50 users per category. The paper's finding replicates: listing 12 favored entities lifts MAP from
  {lo['llm_baseline']:.3f} to {lo['llm_k12']:.3f} ({pct(llm_gain['llm_k12'])}), and 50 accounts lift it to
  {lo['llm_k50']:.3f} ({pct(llm_gain['llm_k50'])}). The paper reported +13% and +23% for GPT-4o. The social-embedding column (SE1, k = 50) matches
  the paper on all users: {fig1_all.loc[PAPER_CATEGORY_ORDER, 'se1_legacy_k50'].mean():.3f} vs. 0.562. On the same 700 users, SocialVec scores
  {lo['socialvec_k12']:.3f} with the 12 accounts and {lo['socialvec_k25']:.3f} with the 25 accounts listed in the LLM prompts
  (k = 50: {lo['se1_legacy_k50']:.3f}), so the LLM is ahead at 12 and 25 accounts, as it is at ranks 81–100 ({tnum('band_llm')}).</p>
  <div class="card">
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s1)"></i>{model_name}, 700 sampled users</span>
    <span class="key"><i class="sq" style="background:var(--s2)"></i>GPT-4o, as reported in the paper</span></div>
    {llm_chart()}
    <details open><summary>Show {tnum('llm')} (per category)</summary>{table3_html()}</details>
    <p class="caption">Overall MAP (mean over 14 categories). The dashed reference lines are computed on the same 700 users.
    The baseline (no user information) is the mean of 10 calls per category with the candidates in random order (Section 4); the
    personalized prompts list the candidates in one random order per user and condition ({tnum('prompt_orders')}). With the candidates
    sorted by follower count the baseline scores {_trials[_trials['order_type'] == 'original'].groupby('topic')['MAP'].mean().mean():.3f}.
    {legacy_fail:.0%} of the personalized responses had an end-of-output marker appended after the id list, which the original
    parser scores as a failure. I read the id list itself. Details are in <code>results/LLM_REPORT.md</code>.</p>
  </div>

  <h3>{ttitle('grid')}</h3>
  <p>The code behind this table is not in the original notebooks. Taking n entities from each of k categories, as the
  paper describes, matches the first row and the last column. Drawing n entities in total from the union of the k
  categories matches the published grid more closely, including its flat n = 1 column. Both readings support the
  paper's conclusion.</p>
  <div class="card"><div class="heats">
    {heatmap(paper.TABLE4, 'Paper<small>as published</small>')}
    {heatmap(grid, f'n per category<small>mean |difference| {d_grid:.4f}</small>')}
    {heatmap(pooled, f'n in total, pooled<small>mean |difference| {d_pool:.4f}</small>')}
  </div><p class="caption">Overall MAP. Rows: k categories; columns: n entities. Darker cells mean higher MAP (scale 0.465–0.525).</p></div>

  <h3>Figure 2 · Socio-demographic profiles of followers</h3>
  <p>User embeddings are classified with SocialVec's five trait classifiers, which I ported to NumPy. The per-entity
  follower profiles match the values behind the paper's figure to within {radar_mean:.3f} on average (largest difference {radar_max:.3f}).</p>
  <div class="card"><img class="figure" alt="Radar plots of follower profiles" src="data:image/png;base64,{fig2_png}"></div>
</section>

<section id="legacy">
  <h2><span class="num">2</span>Notes on the original code</h2>
  <p class="intro">Things I found while matching the notebooks. None of them changes the paper's conclusions.</p>
  <div class="callouts">
    <div class="callout"><span class="tag">Bug</span><h4>Target entities were never removed</h4>
    <p>The "All" profile subtracts screen names from a set of numeric ids, which removes nothing, so the target category
    stays in the profile. Removing it properly moves overall MAP from {ov['se1_all_legacy']:.3f} to {ov['se1_all']:.3f}
    ({pct(ov['se1_all_legacy_gain'])} → {pct(ov['se1_all_gain'])}).</p></div>
    <div class="callout"><span class="tag">Sampling</span><h4>SE1 samples can't be redrawn exactly</h4>
    <p>The original order came from a set of string ids, which changes with every Python run. Redrawn with the same seed,
    the overall values land within ±0.004. The closed-world (SE2) samples match exactly.</p></div>
    <div class="callout"><span class="tag">Detail</span><h4>"Entities" are followed accounts</h4>
    <p>SE1 samples k followed <i>accounts</i>, and only about 43% of them are SocialVec entities. The paper's "1,972
    entities per user" counts all followed accounts.</p></div>
  </div>
</section>

<section id="tail">
  <h2><span class="num">3</span>Extension · Less popular entities</h2>
  <p class="intro">Does social user modeling still help when the candidates aren't the most famous entities in their category?</p>
  <p>In the 9 categories that list 100 typed entities, I sorted the entities by Twitter follower count and ranked those
  at positions <b>81–100</b> instead of the top 20. No original sheet lists more.
  User profiles exclude <i>every</i> entity known to belong to the target category: its original sheet, its top-20 list and
  its extended list, so none of the 40 candidates (ranks 1–20 and 81–100) is in any profile. The main setup
  ({tnum('band_all')}) evaluates each of the 11,153 users in every category in which they follow at least one entity of the
  band; as a matched control, the original top 20 are ranked for the same users with the same profiles. {tnum('band_sampled')} repeats
  this for only the users sampled for each category. A random ranking (exact expected AP) provides a chance level, and
  paired Wilcoxon tests compare the two methods.</p>

  <ul class="highlights">
    <li><b>Popularity breaks down in the tail.</b> On all {B_b['users']:,.0f} user–category pairs, at ranks 81–100 it scores
    {B_b['popularity']:.3f}, barely above a random ranking ({B_b['random']:.3f}), and below it for {word_list(below)}.</li>
    <li><b>Social similarity holds up.</b> It scores {B_b['socialvec']:.3f}, above chance in {sv_above_random_B} of 9 categories, and beats
    popularity in {sv_wins_B} of 9. Its relative gain over popularity grows from {pct(B_h['gain'])} on the top 20 to {pct(B_b['gain'])}
    (p &lt; 0.001 overall and in every category).</li>
    <li><b>The matched control on the users sampled for each category agrees.</b> On those {A_b['users']:,.0f} users the gain is
    {pct(A_h['gain'])} on the top 20 and {pct(A_b['gain'])} at ranks 81–100 ({tnum('band_sampled')}).</li>
    <li class="warn"><b>The exception is {word_list(sv_losses_B)} ({pct(mB['band'].loc['Comedian','gain'])}).</b> Follows in that
    band concentrate on its two most-followed accounts, both mainly political commentators, which popularity ranks first.</li>
  </ul>

  <div class="card">
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s2)"></i>Popularity</span>
    <span class="key"><i class="sq" style="background:var(--s1)"></i>Social similarity</span>
    <span class="key"><i class="tick"></i>Random ranking (expected)</span></div>
    <div class="panels">
      <div class="panel"><h4>Original top 20</h4><p class="note">Same user–category pairs and profiles as on the right</p>{bar_panel(mB['head'], 'Ranks 1–20')}</div>
      <div class="panel"><h4>Ranks 81–100</h4><p class="note">All experiment users who follow a band entity</p>{bar_panel(mB['band'], 'Ranks 81–100')}</div>
    </div>
    <p class="caption">MAP per category; Δ is the gain of social similarity over popularity. The values are in {tnum('band_all')} below.</p>
  </div>

  <h3>{ttitle('band_sampled')}</h3>
  <div class="card">{matched_html(mA)}
  <p class="caption">{A_b['users']:,.0f} users who follow at least one entity ranked 81–100 in their category. Rel./user is the number of
  candidates a user follows. Dotted red marks popularity scores below the random ranking. p is from a Wilcoxon signed-rank test,
  social vs. popularity.</p></div>

  <h3>{ttitle('band_all')}</h3>
  <p>The main setup. Each of the 11,153 users is evaluated in every category in which they follow at least one entity from each band.
  That gives {B_b['users']:,.0f} user–category pairs from {n_distinct_users:,} distinct users; only {A_b['users']/B_b['users']:.0%} of these pairs
  come from users sampled for the category ({tnum('band_sampled')}). The overall test pools all pairs.
  The candidates are listed in {tnum('band_entities')}.</p>
  <div class="card">{matched_html(mB)}{entities_html()}</div>

  <h3>How popular are the candidates overall?</h3>
  <p>The bands are defined within each category. To see where the candidates stand among all accounts, I placed every candidate in
  the popularity ranking of all {pop_n_vocab:,} SocialVec accounts, which are sorted by how many users follow them (rank 1 is the
  most followed). In these 9 categories, the original top 20 have a median global rank of {pop_overall_main[POP_BANDS[0][0]]['median']:,.0f}; the
  ranks 81–100 candidates {pop_overall_main[POP_BANDS[1][0]]['median']:,.0f}, with {pop_overall_main[POP_BANDS[1][0]]['top10k']:.0%} of them in the
  global top 10,000.</p>
  <div class="card">
    <h4 style="margin:0 0 6px">Figure 3 · Global popularity of the candidates, by category</h4>
    <div class="legend">{''.join(f'<span class="key"><i class="sq" style="background:var(--{cls})"></i>{esc(b)}</span>' for b, cls in STRIP_BANDS)}
    <span class="key"><i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ink-2)"></i>One candidate account</span>
    <span class="key"><i class="tick"></i>Median of the band</span></div>
    {pop_strip_chart()}
    <p class="caption">Global SocialVec popularity rank of every candidate (log scale; left is more popular), by category. Each dot is
    one account; the bar marks the band's median. {int(sum(pop_overall_main[b]['n'] - pop_overall_main[b]['in_sv'] for b, _ in STRIP_BANDS))}
    candidates without a SocialVec embedding are not shown.</p>
  </div>
  <h3>{ttitle('popularity')}</h3>
  <div class="card">{pop_table([b for b, _ in STRIP_BANDS], pop_overall_main, POSITIONS)}
  <p class="caption">Global rank: position among all SocialVec accounts, 1 being the most followed. <i>Global rank at category
  position</i> gives the account at positions {', '.join(map(str, POSITIONS[:-1]))} and {POSITIONS[-1]} of the category's original sheet
  sorted by Twitter follower count, the order that defines ranks 81–100 (account name beneath). Positions 1–20 by follower count are not
  the same as the paper's top-20 candidates, which the medians use. Overall pools all candidates of the band.</p></div>

  <h3>{ttitle('band_llm')}</h3>
  <p>The {tnum('llm')} protocol applied to the less popular candidates, for 200 random users per category who follow at least one
  candidate (TV stations have only 198), without {_band_n_before - len(_band_users)} organizations and likely bots (below):
  {band_llm.loc['Overall','users']:,.0f} users. Every listed entity of the
  target category is removed from the prompts. Without user information, the LLM is no better than popularity
  ({band_llm.loc['Overall','llm_baseline']:.3f} vs. {band_llm.loc['Overall','popularity']:.3f}). With user information it personalizes well:
  {band_llm.loc['Overall','llm_k12']:.3f} with 12 entities and {band_llm.loc['Overall','llm_k50']:.3f} with 50 accounts. Social similarity from all of
  the user's entities scores {band_llm.loc['Overall','socialvec']:.3f}. The two are complementary: with 12 entities the LLM is stronger
  for {word_list(band_llm_llm_better)}; social similarity from all entities for {word_list(band_llm_sv_better)}. Ranking from GPT-4o's demographic description alone (Section 5) scores
  {band_llm.loc['Overall','llm_demo']:.3f}, between no user information ({band_llm.loc['Overall','llm_baseline']:.3f}) and 12 entities
  ({band_llm.loc['Overall','llm_k12']:.3f}); it is ahead of 12 entities for {word_list(band_demo_better)}.</p>
  <p>For a like-for-like comparison, the <i>SocialVec k = 12</i> and <i>k = 25</i> columns build the user embedding from exactly the
  accounts listed in the LLM's k = 12 and k = 25 prompts ({band_llm_n[12]:.1f} and {band_llm_n[25]:.1f} on average, as some users have fewer).
  Follow statistics of these {band_llm.loc['Overall','users']:,.0f} users: {follow_stats(_band_users['user'])}.
  Given the same evidence, the LLM ranks better at both sizes:
  {band_llm.loc['Overall','llm_k12']:.3f} vs. {band_llm.loc['Overall','socialvec_k12']:.3f} with 12 accounts
  (paired Wilcoxon p {p_text(band_tests['llm_k12 vs socialvec_k12'])}; ahead in {band_llm_wins[12]} of 9 categories), and
  {band_llm.loc['Overall','llm_k25']:.3f} vs. {band_llm.loc['Overall','socialvec_k25']:.3f} with 25
  (p {p_text(band_tests['llm_k25 vs socialvec_k25'])}; ahead in {band_llm_wins[25]} of 9).</p>
  <p><b>Organizations and likely bots.</b> The profiles of these users (bio, follower, tweet and like counts and account age, as
  collected) were screened: {excl_counts['organization']} are organizations (an organization keyword search of bios and handles, every
  match read by hand) and {excl_counts['possible organization']} more possibly are; {excl_counts['bot: very active']} post more than 144
  tweets a day on average over the account's life; and {excl_counts['bot: 2+ weak signals']} show at least two weaker bot signals
  (more than 72 tweets a day, following 4,900 or more accounts, followers under 5% of follows, a handle ending in six or more digits
  with an empty bio, no likes, an account under 90 days old). No profile looked like a clear bot, so the {_band_n_before - len(_band_users)}
  are a cautious upper bound. They are left out of {tnum('band_llm')}–{tnum('top_llm')} and Section 5.</p>
  <p>The <i>k = 12 of 50</i> and <i>k = 25 of 50</i> columns draw 12 and 25 names from the user's k = 50 list instead of the closed set (see
  {tnum('k25_decomp')} in Section 5).</p>
  <div class="card">{band_llm_html()}
  <p class="caption">Bold marks the best method in each row. With about 190 users per category, per-category MAP has a standard error of
  about 0.02. The first 30 users of each category are those of the earlier 30-user version of this table. Every condition of a user gets the candidates in the same random order (the baseline, demographics and k = 12, 25, 12 of 50, 25 of 50, 50), as in Section 5.
  <i>Demographics</i> is the ranking from GPT-4o's demographic description alone (Section 5, same users and candidate orders).</p></div>

  <h3>{ttitle('band_compare')}</h3>
  <p>The protocol of {tnum('band_llm')} repeated on the 20 most popular candidates of each category, for the
  {top_llm.loc['Overall','users']:,.0f} users of {tnum('band_llm')} who follow at least one of them. Unlike {tnum('llm')}, every
  listed entity of the target category is removed from the prompts here too, so the two bands differ only in their candidates.
  Among the most popular candidates, popularity is a strong ranking ({top_llm.loc['Overall','popularity']:.3f}), and the LLM without user
  information beats it ({top_llm.loc['Overall','llm_baseline']:.3f}). Given the same accounts, the LLM is ahead of SocialVec in both bands:
  {top_llm.loc['Overall','llm_k12']:.3f} vs. {top_llm.loc['Overall','socialvec_k12']:.3f} with 12 accounts at ranks 1–20
  (p {p_text(top_tests['llm_k12 vs socialvec_k12'])}; ahead in {top_llm_wins[12]} of 9 categories).</p>
  <div class="card">{band_compare_html()}
  <p class="caption">Overall MAP (mean over the 9 categories). The ranks 81–100 column is {tnum('band_llm')}; its other
  {band_llm.loc['Overall','users'] - top_llm.loc['Overall','users']:,.0f} users follow none of the top-20 candidates. In both bands every condition of a user gets the candidates in the same random order. p values are paired Wilcoxon tests over users.</p>
  <details open><summary>{ttitle('top_llm')}</summary>{band_llm_html(top_llm)}</details></div>

  <h3>Limitations</h3>
  <p>Only the original top-20 candidates were manually validated; the lower ranks include some mistyped entities (esports teams
  among sports teams, party accounts among TV stations). Users were sampled as followers of the top-20 entities. Users follow
  only {mA['band']['relevant_per_user'].min():.1f}–{mA['band']['relevant_per_user'].max():.1f} band candidates each, so per-user AP is noisy in the smaller categories.</p>
</section>

<section id="order">
  <h2><span class="num">4</span>Does the LLM depend on the order of the candidates?</h2>
  <p class="intro">The original prompts list the candidates sorted by follower count, which an order-following model could
  turn into popularity ranking for free.</p>
  <p>For the same 700 users, I re-ran the k = 25 prompts twice: once unchanged, to measure run-to-run noise, and once with the
  candidates shuffled. I also called the non-personalized baseline 10 times per category in its original order and 10 times
  shuffled. The results are in {tnum('order_personal')} and {tnum('order_baseline')}.</p>
  <ul class="highlights">
    <li><b>Personalization barely depends on the order.</b> Shuffling lowers MAP from {order_st.loc['MAP','original']:.3f}
    to {order_st.loc['MAP','shuffled']:.3f}. The drop is significant (p {pval(order_st.loc['p vs original (Wilcoxon)','shuffled'])}) but small,
    and repeating the identical prompt gives {order_st.loc['MAP','repeat']:.3f}.</li>
    <li><b>Some of the popularity signal comes from the input order.</b> The output's correlation with popularity falls from
    τ = {order_st.loc['τ with popularity order','original']:.2f} to {order_st.loc['τ with popularity order','shuffled']:.2f} when the
    candidates are shuffled.</li>
    <li class="warn"><b>The non-personalized baseline follows the input order more closely</b> (τ = {base_st.loc['original','τ with presented order']:.2f}).
    Shuffling costs it {base_st.loc['original','overall MAP (mean over trials)'] - base_st.loc['shuffled','overall MAP (mean over trials)']:.3f} MAP,
    several times the spread across repeated calls. Non-personalized LLM rankings should be evaluated with randomized orders.</li>
    <li><b>Outputs are not deterministic.</b> Ten identical baseline calls gave {base_st.loc['original','distinct rankings per category (of 10)']:.1f}
    distinct rankings on average. Two identical personalized calls change a user's AP by {order_st.loc['mean |ΔAP| vs original','repeat']:.3f}
    on average, but overall MAP is stable.</li>
  </ul>
  <div class="card">{order_html()}</div>

  <h3>{ttitle('prompt_orders')}</h3>
  <p>Because the order matters for the baseline, every ranking on this page that is compared across methods presents the candidates in a
  random order; the follower-count order of the authors' prompts appears only in the comparison above.</p>
  <div class="card"><div class="scroll"><table class="data compact"><thead><tr><th>Where</th><th>No user information (baseline)</th>
  <th>Personalized prompts (k = 12, 25, 50 names; demographics)</th></tr></thead><tbody>
  <tr><th>{tnum('llm')}</th><td class="names">10 calls per category, each with a different random order; MAP averaged over the calls</td>
  <td class="names">One random order per user and condition</td></tr>
  <tr><th>{tnum('order_personal')}–{tnum('order_baseline')}</th><td class="names">10 calls per category in follower-count order and 10 shuffled</td>
  <td class="names">k = 25 only: follower-count order (twice) and one random order per user</td></tr>
  <tr><th>{tnum('band_llm')}–{tnum('top_llm')}</th><td class="names">One random order per user and band</td>
  <td class="names">The same order as the user's baseline</td></tr>
  <tr><th>Section 5 ({tnum('demo_bands')})</th><td class="names">One random order per user and band</td>
  <td class="names">The same order for every condition of the user, including the demographic description</td></tr>
  </tbody></table></div>
  <p class="caption">All random orders come from fixed seeds, so every run can be reproduced.</p></div>
  <p class="caption">{llm_calls:,} model calls in total across all LLM experiments (including those in the backup page), costing ${llm_cost:.2f}.</p>
</section>

<section id="demographics">
  <h2><span class="num">5</span>Can demographics alone personalize the LLM?</h2>
  <p class="intro">The second version of this experiment, on the users of {tnum('band_llm')}–{tnum('top_llm')} and with a
  one-value-per-attribute instruction. The first version (270 users) is in <a href="results.html">results.html</a> and <a href="LLM_DEMOGRAPHICS_REPORT.html">the
  demographics report</a>.</p>
  <p>GPT-4o first guessed each user's gender, age, ethnicity and education from the names of accounts they follow, and was told to rely
  on stereotypes rather than decline and to state exactly one value per attribute. {model_name} then ranked the candidates given only
  that description, both for the most popular candidates of each category (ranks 1–20, as in {tnum('llm')}) and for ranks 81–100;
  {HAIKU} repeated every ranking (same users, candidate orders, descriptions and prompts) as a second, low-cost ranking model. Every
  ranking lists the candidates in one random order per user, shared by all of that user's conditions. For comparison, the same
  shuffled candidates were also ranked from the names of 12, 25 or 50 followed accounts, and from 12 or 25 names drawn from the 50.</p>

  <h3>{ttitle('demo_users')}</h3>
  <p>The {demo['users']:,} users of {tnum('band_llm')} (200 random users per category who follow at least one ranks 81–100 candidate; TV
  stations have only 198; without the {_band_n_before - len(_band_users)} organizations and likely bots described there), so that every count matches {tnum('band_llm')}–{tnum('top_llm')}: all of them are evaluated at ranks 81–100,
  and the {dsh['1-20']['users']:,} who follow at least one top-20 candidate at ranks 1–20. {tnum('demo_bands')} also reports the
  {demo['filter']['users']:,} users who follow fewer than {demo['filter']['max_follows']:,} accounts in all (fewer likely bots) and at least
  {demo['filter']['min_entities']} of the experiment's category entities (enough evidence for an informed guess).</p>
  <div class="card">{demo_users_html()}
  <p class="caption">Category entities are the candidate entities of all 14 category lists that the user follows.</p></div>
  <p><b>Which accounts GPT-4o saw.</b> The accounts of the user's k = 50 prompt: 50 accounts drawn at random (fixed seed) from
  everything the user follows, after removing every listed entity of the target category. Only accounts with a known name are kept,
  which leaves the counts in the last row above. The names are the accounts' Twitter display names from the SocialVec account metadata,
  listed in the random order in which they were drawn. GPT-4o never sees the user's own name or handle. {tnum('demo_examples')} at
  the end of this section shows a few of these lists.</p>

  <h3>Prompts</h3>
  <p>Both stages use dspy signatures, as in the original experiments. These are the exact messages dspy sends, with the inputs shown
  as placeholders in braces. The last instruction of Stage 1 is new in this version.</p>
  <div class="card"><h4 style="margin:0 0 6px">Stage 1 · Generating the demographic description (GPT-4o)</h4>
  <pre class="prompt" style="white-space:pre-wrap;font-size:12.5px;margin:0">{esc(demo_prompts['describe'])}</pre></div>
  <div class="card"><h4 style="margin:0 0 6px">Stage 2 · Ranking the candidates from the description ({model_name})</h4>
  <pre class="prompt" style="white-space:pre-wrap;font-size:12.5px;margin:0">{esc(demo_prompts['rank'])}</pre>
  <p class="caption">The model's answer is read from the <code>[[ ## ranked_ids ## ]]</code> field. The ranking prompts with account
  names and without user information are the authors' prompts used throughout this page. {HAIKU} receives exactly the same system and
  user messages through the Anthropic API (adaptive thinking at effort <i>low</i>, Message Batches API).</p></div>
  <div class="card"><h4 style="margin:0 0 6px">Example · One Stage 2 call ({esc(stage2['band'].replace('-', '–'))}, a user sampled for musical artists)</h4>
  <p class="note">The system message is the one above; this is the user message with the real inputs: GPT-4o's description of the user
  (the first user of {tnum('demo_examples')}) and the candidates in the user's random order.</p>
  {stage2_example_html()}</div>

  <ul class="highlights">
    <li><b>Demographics help, most in the tail.</b> GPT-4o's descriptions beat the no-information ranking at ranks 81–100,
    {dsh['81-100']['map']['ap_gpt-4o']:.3f} vs. {dsh['81-100']['map']['ap_baseline']:.3f} (p {p_text(dsh['81-100']['tests']['4o_vs_baseline'])}),
    and by less at ranks 1–20, {dsh['1-20']['map']['ap_gpt-4o']:.3f} vs. {dsh['1-20']['map']['ap_baseline']:.3f}
    (p {p_text(dsh['1-20']['tests']['4o_vs_baseline'])}).</li>
    <li><b>Among the most popular candidates the no-information ranking is already strong:</b>
    {dsh['1-20']['map']['ap_baseline']:.3f} at ranks 1–20, against popularity's {dsh['1-20']['map']['popularity']:.3f}
    (p {p_text(dsh['1-20']['tests']['baseline_vs_popularity'])}).</li>
    <li><b>Account names say more than demographics.</b> 12 names score {dsh['81-100']['map']['ap_k12']:.3f} at ranks 81–100 and
    {dsh['1-20']['map']['ap_k12']:.3f} at ranks 1–20, above the descriptions in both (p {p_text(dsh['81-100']['tests']['k12_vs_4o'])} and
    {p_text(dsh['1-20']['tests']['k12_vs_4o'])}).</li>
    <li><b>50 names beat SocialVec's full profile:</b> {dsh['81-100']['map']['ap_k50']:.3f} vs. {dsh['81-100']['map']['socialvec']:.3f} at ranks 81–100
    (p {p_text(dsh['81-100']['tests']['k50_vs_socialvec'])}) and {dsh['1-20']['map']['ap_k50']:.3f} vs. {dsh['1-20']['map']['socialvec']:.3f} at ranks 1–20
    (p {p_text(dsh['1-20']['tests']['k50_vs_socialvec'])}).</li>
    <li><b>On the {demo['filter']['users']:,} users who pass the follow filter</b>, the demographic gain holds at ranks 81–100
    ({dshf['81-100']['map']['ap_gpt-4o']:.3f} vs. {dshf['81-100']['map']['ap_baseline']:.3f}, p {p_text(dshf['81-100']['tests']['4o_vs_baseline'])}) but not
    at ranks 1–20 ({dshf['1-20']['map']['ap_gpt-4o']:.3f} vs. {dshf['1-20']['map']['ap_baseline']:.3f}, p {p_text(dshf['1-20']['tests']['4o_vs_baseline'])});
    at ranks 81–100 they come close to 12 account names ({dshf['81-100']['map']['ap_k12']:.3f}, p {p_text(dshf['81-100']['tests']['k12_vs_4o'])}).</li>
    <li><b>{HAIKU} ranks a little lower but shows the same ordering.</b> Demographics beat no information at ranks 81–100,
    {dhk['81-100']['map']['ap_gpt-4o']:.3f} vs. {dhk['81-100']['map']['ap_baseline']:.3f} (p {p_text(dhk['81-100']['tests']['4o_vs_baseline'])}), and at ranks 1–20,
    {dhk['1-20']['map']['ap_gpt-4o']:.3f} vs. {dhk['1-20']['map']['ap_baseline']:.3f} (p {p_text(dhk['1-20']['tests']['4o_vs_baseline'])}), and account names beat
    demographics. Unlike {model_name}, its 50 account names only match SocialVec's full profile:
    {dhk['81-100']['map']['ap_k50']:.3f} vs. {dhk['81-100']['map']['socialvec']:.3f} at ranks 81–100 (p {p_text(dhk['81-100']['tests']['k50_vs_socialvec'])}) and
    {dhk['1-20']['map']['ap_k50']:.3f} vs. {dhk['1-20']['map']['socialvec']:.3f} at ranks 1–20 (p {p_text(dhk['1-20']['tests']['k50_vs_socialvec'])}).</li>
    <li><b>The descriptions stay on demographics.</b> Only {demo['leak_4o']} of the {demo['users']:,} descriptions also name the user's
    interests, against the instruction (60 of 270 in the first version).</li>
  </ul>

  <div class="card">
    <h4 style="margin:0 0 6px">Figure 4 · Ranking from demographics vs. account names, ranks 1–20 and 81–100, {model_name} and {HAIKU}</h4>
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s2)"></i>Demographics by GPT-4o</span>
    <span class="key"><i class="sq" style="background:var(--s3)"></i>Followed account names</span></div>
    <div class="panels">
      <div class="panel"><h4>Ranks 1–20 · {model_name} · {dsh['1-20']['users']:,} users</h4>{demographics_chart('1-20')}</div>
      <div class="panel"><h4>Ranks 1–20 · {HAIKU} · {dhk['1-20']['users']:,} users</h4>{demographics_chart('1-20', dhk, HAIKU)}</div>
      <div class="panel"><h4>Ranks 81–100 · {model_name} · {dsh['81-100']['users']:,} users</h4>{demographics_chart('81-100')}</div>
      <div class="panel"><h4>Ranks 81–100 · {HAIKU} · {dhk['81-100']['users']:,} users</h4>{demographics_chart('81-100', dhk, HAIKU)}</div>
    </div>
    <p class="caption">Overall MAP (mean over categories) with the candidates in random order. Each band uses the users of
    {tnum('demo_users')} who follow at least one of its candidates. Markers are paired Wilcoxon tests over users:
    * vs. the LLM with no user information, † vs. the demographics condition (*/† p &lt; 0.05, **/†† p &lt; 0.01, ***/††† p &lt; 0.001,
    n.s. not significant). {f'All {n_marked} comparisons are significant at p &lt; 0.001, so they stay significant after a Bonferroni correction.' if sig_all else ''}
    The p values are in {tnum('demo_bands')}.</p>
  </div>
  <div class="card"><h4 style="margin:0 0 6px">{ttitle('demo_bands')}</h4>{demographics_table()}
  <p class="caption">Overall MAP is the mean over categories; p values are paired Wilcoxon tests over users. The all-users columns are the
  users of {tnum('band_llm')} (ranks 81–100) and {tnum('top_llm')} (ranks 1–20), with the same random candidate orders, so their
  no-information, account-name and SocialVec scores are those tables' values.</p></div>

  <h3>Where the account names come from vs. how many there are</h3>
  <p>The 12 and 25 names come from the user's closed set (the entities of the other categories' lists the user follows), while the 50
  come from everything the user follows. Going from 25 to 50 names therefore changes two things at once: the number of accounts and
  where they come from. To separate the two, both models also ranked the candidates, in the same candidate order, from 25 names drawn at
  random (fixed seed) from the names of the user's k = 50 prompt, and from the first 12 of those 25, so that the 12 are nested in the
  25 as in the closed set. The prompts list {band_llm_n[12]:.1f} and {demo_n_k25:.1f} names on average from the closed set, and
  {demo['user_stats']['names_k12of50']['mean']:.1f} and {demo_n_k25of50:.1f} from the full follow list ({demo['user_stats']['names_given']['mean']:.1f}
  with k = 50), so each pair of conditions differs in one factor.</p>
  <ul class="highlights">
    <li><b>Where the names come from matters at every size.</b> At ranks 81–100, names from the full follow list beat closed-set names
    with 12 names ({model_name} {k25d[1]['k12of50']:.3f} vs. {k25d[1]['k12']:.3f}, {HAIKU} {k25d[3]['k12of50']:.3f} vs. {k25d[3]['k12']:.3f})
    and with 25 ({k25d[1]['k25of50']:.3f} vs. {k25d[1]['k25']:.3f} and {k25d[3]['k25of50']:.3f} vs. {k25d[3]['k25']:.3f}); at ranks 1–20 the
    gap is smaller ({min(min(r['source12'], r['source25']) for r in k25d[0::2]):+.3f} to {max(max(r['source12'], r['source25']) for r in k25d[0::2]):+.3f}).
    {'All eight differences are significant (p &lt; 0.001).' if all(r[f'p_{n}'] < 0.001 for r in k25d for n in ('source12', 'source25')) else ''}</li>
    <li><b>How many names matters less.</b> From the closed set, 25 names add only {min(r['size_closed'] for r in k25d):+.3f} to
    {max(r['size_closed'] for r in k25d):+.3f} MAP over 12 (the k = 25 list of {closed_lt25:.0%} of users has fewer than 25 names, as they follow fewer
    closed-set entities). From the full follow list they add {min(r['size_full'] for r in k25d):+.3f} to {max(r['size_full'] for r in k25d):+.3f},
    most at ranks 81–100 ({model_name} {k25d[1]['size_full']:+.3f}, {HAIKU} {k25d[3]['size_full']:+.3f}, p {p_text(k25d[1]['p_size_full'])} and
    {p_text(k25d[3]['p_size_full'])}). Going on to 50 names adds {min(r['size_50'] for r in k25d):+.3f} to {max(r['size_50'] for r in k25d):+.3f}.</li>
    <li><b>So the jump from 25 to 50 names is mostly the change of source:</b> {k25d[1]['share']:.0%} of {model_name}'s gain at ranks
    81–100 ({k25d[1]['k25']:.3f} → {k25d[1]['k50']:.3f}) and {k25d[3]['share']:.0%} of {HAIKU}'s; {k25d[0]['share']:.0%} and
    {k25d[2]['share']:.0%} at ranks 1–20.</li>
  </ul>
  <p>For the {demo['user_stats']['names_given_le_25']:.0%} of users whose k = 50 prompt lists 25 names or fewer (accounts without a known
  name are dropped), the 25 names from the 50 are the k = 50 list itself, which is one reason the last size step is small.</p>
  <div class="card"><h4 style="margin:0 0 6px">{ttitle('k25_decomp')}</h4>{k25_decomposition_html()}
  <p class="caption">Overall MAP (mean over categories). Closed set: the entities of the other categories' lists the user follows (the
  k = 12 and k = 25 prompts). Full follow list: names drawn from the user's k = 50 prompt (25 at random, 12 of those 25, or all of them).
  The source columns compare the two sources at the same number of names, the size columns the same source at different numbers, with
  paired Wilcoxon p values over users.</p></div>

  <h3>What GPT-4o stated about the users</h3>
  <p>Each description is parsed with fixed rules for the value it states for each attribute. With the one-value instruction,
  {demo_multiple} attribute values across the {demo['users']} descriptions still name more than one value (174 across 270 in the first
  version); those count under the first value stated. "Middle-aged" is kept as stated, without assigning it an age range.</p>
  <ul class="highlights">
    <li><b>The guesses are concentrated.</b> {demo_attrs['Ethnicity']['counts'].get('Caucasian / White', 0)} of {demo['users']} descriptions say
    Caucasian or white and {demo_attrs['Education']['counts'].get('Higher education, level not stated', 0)} say higher education without a level.
    Gender is the most balanced attribute ({demo_attrs['Gender']['counts'].get('Male', 0)} male, {demo_attrs['Gender']['counts'].get('Female', 0)} female).</li>
  </ul>
  <div class="card">
    <h4 style="margin:0 0 6px">Figure 5 · Demographic attributes stated in GPT-4o's descriptions · {demo['users']} users</h4>
    {demo_attrs_chart()}
    <p class="caption">Share of descriptions by the value stated for each attribute. Hover a segment for the count.</p>
  </div>
  <h3>{ttitle('demo_attrs')}</h3>
  <div class="card">{demo_attrs_table()}
  <p class="caption">Education counts a specific level (high school, some higher education, a degree) when one is stated anywhere in the
  description, and "level not stated" otherwise.</p></div>
  <h3>{ttitle('demo_attrs_cat')}</h3>
  <div class="card">{demo_attrs_cat_table()}
  <p class="caption">Share of the category's descriptions stating each value. Age shows every age group the descriptions use
  ({word_list([v.lower() if v != 'Under 30' else 'under 30' for _, v in DEMO_CAT_COLS if _ == 'Age'])}){age_rare_note}; for ethnicity and
  education only the two most frequent values are shown, and the rest are in {tnum('demo_attrs')}.</p></div>

  <h3>{ttitle('demo_examples')}</h3>
  <p>One user from each of {len(demo['examples'])} categories: the user whose list of account names (the input to GPT-4o) contains the
  most known entities, i.e. accounts in the original or extended category lists. The table shows those known entities by name, in the
  order GPT-4o saw them; GPT-4o also saw the user's other, less known accounts (the "Known / all" column), and the description is
  the one it wrote from the full list. No model was called again for these examples. Possible motivating examples.</p>
  <div class="card">{demo_examples_html()}</div>
  <p class="caption">The descriptions are stereotype-based guesses about real people, shown here identified only by category.
  They should not be read as findings about which groups like which entities.</p>
</section>

<section id="supplement">
  <h2><span class="num">S</span>Supplementary · Ranking with {HAIKU}</h2>
  <p class="intro">All results of the {HAIKU} rankings of Section 5: the same {dhk['81-100']['users']:,} users (ranks 81–100) and
  {dhk['1-20']['users']:,} users (ranks 1–20), candidate orders, GPT-4o descriptions and prompts as {model_name}.</p>
  <p>{HAIKU} received the exact chat messages dspy renders for each signature (system and user message), through the Anthropic API
  with adaptive thinking at effort <i>low</i> and the Message Batches API: {hk_calls:,} calls costing ${hk_cost:.2f}. The answer's
  <code>ranked_ids</code> field is scored as for {model_name}: candidates it leaves out count as misses. The overall figures are in
  {tnum('demo_bands')} and Figure 4.</p>

  <h3>{ttitle('haiku_1_20')}</h3>
  <div class="card">{haiku_category_table('1-20')}
  <p class="caption">MAP per category with the candidates in each user's random order; Overall is the mean over categories. Bold marks
  the best method in each row. The last row gives {model_name}'s overall MAP in the same conditions ({tnum('demo_bands')}).</p></div>

  <h3>{ttitle('haiku_81_100')}</h3>
  <div class="card">{haiku_category_table('81-100')}
  <p class="caption">As above, for the candidates at ranks 81–100.</p></div>

  <h3>{ttitle('haiku_complete')}</h3>
  <div class="card">{haiku_completeness_table()}
  <p class="caption">Share of answers whose <code>ranked_ids</code> contain every candidate, at least 95% of them, or no usable
  ranking at all, per model, band and condition.</p></div>
</section>

<section id="open">
  <h2><span class="num">6</span>Open items</h2>
  <div class="callouts">
    <div class="callout"><span class="tag">Scale</span><h4>LLM experiments on all users</h4>
    <p>The LLM results still use samples: 50 users per category for {tnum('llm')} (700 of the {len(_all_users):,} users), and
    200 per category for {tnum('band_llm')}–{tnum('top_llm')}, Section 5 and the supplement ({band_llm.loc['Overall','users']:,.0f} of the
    3,798 users who follow a ranks 81–100 candidate; TV stations have only 198). Overall MAP is then within about ±0.01. Running every
    remaining user would take about 73K more calls, about $19 at this page's measured cost per call: {tnum('llm')} ~31K calls (~$8),
    {tnum('band_llm')}–{tnum('top_llm')} ~16K (~$4), Section 5 ~6K (~$4, with the GPT-4o descriptions), {HAIKU} ~20K (~$2). The same
    scripts run larger samples, and cached responses are reused.</p></div>
    <div class="callout"><span class="tag">Next</span><h4>Description-based tail candidates</h4>
    <p>The requested variant selects 50–100 accounts per category by their <i>textual descriptions</i>, below the top 10K/20K
    SocialVec entities. The ranks 81–100 band does not meet that criterion fully: {n_below_10k} of its {n_band_in_sv} candidates
    ({n_below_10k / n_band_in_sv:.0%}) fall below the top 10K, and it uses the typed lists rather than descriptions.</p></div>
  </div>
</section>

<footer>
  <p>Regenerate the underlying results with <code>python run_reproduction.py</code>,
  <code>python run_popularity_band.py --start 81 --end 100</code> and <code>python run_llm_experiments.py</code>.
  For Section 5, run <code>python run_llm_demographics.py</code> (with <code>--describe-model gpt-4o</code> for the repeat),
  <code>python run_llm_demographics_bands.py</code> and <code>python make_demographics_report.py</code>.
  Then rebuild this page with <code>python make_results_page.py</code>. Details are in <code>results/REPORT.md</code>,
  <code>results/POPULARITY_BAND_REPORT.md</code> and <code>results/LLM_REPORT.md</code>. Results that use the extended category
  lists are kept in <a href="results_backup.html">results_backup.html</a>.</p>
</footer>
</div>
<script>{JS}</script>
</body></html>
"""
(R / "results_v4.html").write_text(page, encoding="utf-8")
print("wrote", R / "results_v4.html", f"{len(page)/1e6:.2f} MB")
