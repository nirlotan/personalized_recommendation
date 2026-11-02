"""
Build results/results.html, a self-contained page of all results, from the result CSVs.

    python make_results_page.py

Run after run_reproduction.py, run_popularity_band.py and run_llm_experiments.py.
"""
import ast
import base64
import gzip
import html
import json
import pickle
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from reproduction import paper_results as paper
from reproduction.config import CATEGORY_LABELS, MODELS_DIR, PAPER_CATEGORY_ORDER, ROOT
from reproduction.data import load_categories
from reproduction.popularity_bands import load_category_sheets, popularity_band
from reproduction.page_style import CSS, JS
from run_llm_experiments import by_category

R = ROOT / "results"
LLM = R / "llm"
BAND = R / "popularity_band_81_100"
t2 = pd.read_csv(R / "table2.csv", index_col=0)
fig1 = pd.read_csv(R / "figure1.csv", index_col=0)
grid = pd.read_csv(R / "table4_per_category.csv", index_col=0)
pooled = pd.read_csv(R / "table4_pooled.csv", index_col=0)
mA = {c: pd.read_csv(BAND / f"matched_category_{c}.csv", index_col=0) for c in ("head", "band")}
mB = {c: pd.read_csv(BAND / f"matched_all_{c}.csv", index_col=0) for c in ("head", "band")}
band_entities = pd.read_csv(BAND / "band_entities.csv")
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
    "fig1_data": "Overall MAP by number of sampled entities (data of Figure 1)",
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
    "demo_bands": "Overall MAP of demographics-only and name-based ranking, ranks 1–20 and 81–100",
    "demo_attrs": "Demographic attributes stated in GPT-4o's descriptions",
    "demo_attrs_cat": "Demographic attributes stated in GPT-4o's descriptions, by category",
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
    """Overall MAP vs number of sampled entities (Figure 1, overall panel)."""
    o = fig1.loc["Overall"]
    xs = ["10", "20", "30", "40", "50", "80", "100", "300", "500", "All"]
    W, H, L, Rm, T, B = 760, 360, 56, 24, 16, 44
    y0, y1 = 0.46, 0.58
    px = lambda i: L + i * (W - L - Rm) / (len(xs) - 1)
    py = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Overall MAP versus number of sampled entities">']
    for v in np.arange(y0, y1 + 1e-9, 0.02):
        parts.append(f'<line class="grid" x1="{L}" x2="{W-Rm}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="tick" x="{L-8}" y="{py(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    for i, x in enumerate(xs):
        parts.append(f'<text class="tick" x="{px(i):.1f}" y="{H-B+20}" text-anchor="middle">{x}</text>')
    parts.append(f'<text class="axis-title" x="{(L+W-Rm)/2}" y="{H-6}" text-anchor="middle">Accounts sampled per user (SE1) / closed-world entities (SE2)</text>')
    # popularity reference
    yp = py(o["popularity"])
    parts.append(f'<line class="ref" x1="{L}" x2="{W-Rm}" y1="{yp:.1f}" y2="{yp:.1f}"/>'
                 f'<text class="ref-label" x="{W-Rm-4}" y="{yp-6:.1f}" text-anchor="end">Popularity baseline {o["popularity"]:.3f}</text>')
    se1 = [o[f"se1_legacy_k{k}"] for k in xs[:-1]] + [o["se1_all_legacy"]]
    se1c = [o[f"se1_k{k}"] for k in xs[:-1]] + [o["se1_all"]]
    se2 = [o[f"se2_k{k}"] for k in xs[:5]]
    series = [("s1", "SE1 · paper setup", se1), ("s2", "SE1 · target removed", se1c), ("s3", "SE2 · closed world", se2)]
    for cls, name, vals in series:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline class="line {cls}" points="{pts}"/>')
    paper_vals = {
        "s1": {**{str(k): v for k, v in paper.FIGURE1_SE1.items()}, "All": paper.TABLE2_OVERALL["all"]},
        "s3": {str(k): v for k, v in paper.FIGURE1_SE2.items()},
    }
    for cls, marks in paper_vals.items():
        for k, v in marks.items():
            x, y = px(xs.index(k)), py(v)
            parts.append(f'<rect class="paper-mark" x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" transform="rotate(45 {x:.1f} {y:.1f})"/>')
    for cls, name, vals in series:
        for i, v in enumerate(vals):
            parts.append(f'<circle class="dot {cls}" cx="{px(i):.1f}" cy="{py(v):.1f}" r="4"/>')
    for cls, name, vals in series:
        for i, v in enumerate(vals):
            tip = f"{name}<br>k = {xs[i]}: <b>{v:.3f}</b>"
            if xs[i] in paper_vals.get(cls, {}):
                tip += f"<br>Paper: {paper_vals[cls][xs[i]]:.3f}"
            parts.append(f'<circle class="hit" cx="{px(i):.1f}" cy="{py(v):.1f}" r="11" data-tip="{esc(tip)}"/>')
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
    for cat in PAPER_CATEGORY_ORDER + ["Overall"]:
        o = t2.loc[cat]
        if cat == "Overall":
            ref = {"n_relevant": np.nan, **paper.TABLE2_OVERALL}
        else:
            ref = paper.TABLE2.loc[cat]
        ent = "" if cat == "Overall" else with_paper(o["n_relevant"], ref["n_relevant"], "{:.1f}")
        cls = ' class="total"' if cat == "Overall" else ""
        neg = lambda g: ' class="neg"' if g < 0 else ""
        rows.append(
            f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th>"
            f"<td>{len(per_user) if cat == 'Overall' else users_per_category[cat]:,}</td><td>{ent}</td>"
            f"<td>{with_paper(o['popularity'], ref['popularity'])}</td>"
            f"<td>{with_paper(o['se1_all_legacy'], ref['all'])}</td><td{neg(o['se1_all_legacy_gain'])}>{pct(o['se1_all_legacy_gain'])}</td>"
            f"<td>{with_paper(o['se2_all'], ref['sampled'])}</td><td{neg(o['se2_all_gain'])}>{pct(o['se2_all_gain'])}</td>"
            f"<td><span class='num'>{o['se1_all']:.3f}</span></td><td{neg(o['se1_all_gain'])}>{pct(o['se1_all_gain'])}</td></tr>")
    return ("<div class='scroll'><table class='data'><thead><tr><th>Category</th><th>Users</th><th>Entities / user</th><th>Popularity</th>"
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
            f"<td>{rel(h['relevant_per_user'])}</td><td>{h['popularity']:.3f}</td><td>{h['socialvec']:.3f}</td>"
            f"<td{neg(h['gain'])}>{pct(h['gain'])}</td><td class='p'>{pval(h['p_value'])}</td>"
            f"<td class='grp'>{rel(b['relevant_per_user'])}</td><td>{b['random']:.3f}</td><td{below}>{b['popularity']:.3f}</td>"
            f"<td>{b['socialvec']:.3f}</td><td{neg(b['gain'])}>{pct(b['gain'])}</td><td class='p'>{pval(b['p_value'])}</td></tr>")
    return ("<div class='scroll'><table class='data'><thead>"
            f"<tr class='super'><th></th><th></th><th colspan='5'>Ranks 1–20 (original candidates)</th><th colspan='6' class='grp'>{band_label}</th></tr>"
            "<tr><th>Category</th><th>Users</th><th>Rel./user</th><th>Popularity</th><th>Social</th><th>Δ</th><th>p</th>"
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
    cols = ["random", "popularity", "socialvec", "socialvec_k12", "socialvec_k25"] + LLM_COLS
    group_start = {"socialvec", "llm_baseline"}
    rows = []
    for cat, r in table.iterrows():
        cls = " class='total'" if cat == "Overall" else ""
        best = max(r[c] for c in cols)
        cells = "".join(f"<td{' class=grp' if c in group_start else ''}>{'<b>' if r[c] == best else ''}{r[c]:.3f}{'</b>' if r[c] == best else ''}</td>" for c in cols)
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th><td>{r['users']:,.0f}</td>{cells}</tr>")
    return ("<div class='scroll'><table class='data'><thead><tr class='super'><th></th><th></th><th colspan='2'>Non-personalized</th>"
            f"<th colspan='3' class='grp'>SocialVec</th><th colspan='4' class='grp'>{esc(model_name)}</th></tr>"
            "<tr><th>Category</th><th>Users</th><th>Random</th><th>Popularity</th>"
            "<th class='grp'>All entities</th><th>k = 12</th><th>k = 25</th><th class='grp'>Baseline</th><th>k = 12</th><th>k = 25</th><th>k = 50</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def table1d_html():
    o = fig1.loc["Overall"]
    rows = []
    for k in ["10", "20", "30", "40", "50", "80", "100", "300", "500", "5000"]:
        se2 = f"{o[f'se2_k{k}']:.3f}" if f"se2_k{k}" in o else ""
        ref1 = paper.FIGURE1_SE1.get(int(k))
        ref2 = paper.FIGURE1_SE2.get(int(k))
        rows.append(f"<tr><th>{k}</th><td>{o[f'se1_legacy_k{k}']:.3f}</td><td class='muted'>{'' if ref1 is None else f'{ref1:.3f}'}</td>"
                    f"<td>{o[f'se1_k{k}']:.3f}</td><td>{se2}</td><td class='muted'>{'' if ref2 is None else f'{ref2:.3f}'}</td></tr>")
    rows.append(f"<tr class='total'><th>All</th><td>{o['se1_all_legacy']:.3f}</td><td class='muted'>{paper.TABLE2_OVERALL['all']:.3f}</td>"
                f"<td>{o['se1_all']:.3f}</td><td>{o['se2_all']:.3f}</td><td class='muted'>{paper.TABLE2_OVERALL['sampled']:.3f}</td></tr>")
    return (f"<details open><summary>{ttitle('fig1_data')}</summary><div class='scroll'><table class='data compact'><thead><tr><th>k</th>"
            "<th>SE1 · paper setup</th><th>Paper</th><th>SE1 · target removed</th><th>SE2</th><th>Paper</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>")


def entities_html(entities=None, key="band_entities"):
    entities = band_entities if entities is None else entities
    rows = []
    for cat, g in entities.groupby("category", sort=False):
        names = ", ".join(esc(n) + ("<sup>*</sup>" if not v else "") for n, v in zip(g["twitter_screen_name"], g["in_socialvec"]))
        rng = f"{g['followers_count'].min()/1e3:,.0f}K – {g['followers_count'].max()/1e3:,.0f}K"
        rows.append(f"<tr><th>{esc(CATEGORY_LABELS[cat])}</th><td class='nowrap'>{rng}</td><td class='names'>{names}</td></tr>")
    return (f"<details open><summary>{ttitle(key)} (* no SocialVec embedding, excluded)</summary><div class='scroll'><table class='data compact'>"
            "<thead><tr><th>Category</th><th>Followers</th><th>Entities</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>")


# ---------------------------------------------------------------- numbers for prose
ov = t2.loc["Overall"]
max_t2 = np.abs(t2.loc[PAPER_CATEGORY_ORDER, ["popularity", "se1_all_legacy", "se2_all"]].values
                - paper.TABLE2.loc[PAPER_CATEGORY_ORDER, ["popularity", "all", "sampled"]].values).max()
d_grid = np.abs(grid.values - paper.TABLE4.values).mean()
d_pool = np.abs(pooled.values - paper.TABLE4.values).mean()
A_h, A_b = mA["head"].loc["Overall"], mA["band"].loc["Overall"]
B_h, B_b = mB["head"].loc["Overall"], mB["band"].loc["Overall"]
below = [CATEGORY_LABELS[c] for c in mA["band"].index if c != "Overall" and mA["band"].loc[c, "popularity"] < mA["band"].loc[c, "random"]]

fig1_png, fig2_png = b64(R / "figure1.png"), b64(R / "figure2.png")

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
llm_t3 = use_shuffled_baseline(pd.read_csv(LLM / "table3_main.csv", index_col=0),
                               _trials[_trials["order_type"] == "shuffled"].groupby("topic")["MAP"].mean())
llm_gain = {c: llm_t3.loc["Overall", c] / llm_t3.loc["Overall", "llm_baseline"] - 1 for c in ["llm_k12", "llm_k25", "llm_k50"]}
order_st = pd.read_csv(LLM / "order_stats.csv", index_col=0)
base_st = pd.read_csv(LLM / "baseline_trial_stats.csv", index_col=0)
# 200 users per category (198 for TV stations), the first 30 of which are the users of Section 5 (run_band_more_users.py)
_band_users = pd.read_csv(LLM / "band_per_user_n200.csv.gz")
_band_users["llm_baseline"] = _band_users["llm_baseline_shuffled"]  # one random candidate order per user
band_llm = by_category(_band_users, ["random", "popularity", "socialvec", "socialvec_k12", "socialvec_k25"] + LLM_COLS)
band_tests = {f"llm_k{k} vs socialvec_k{k}": wilcoxon(_band_users[f"llm_k{k}"] - _band_users[f"socialvec_k{k}"]).pvalue for k in (12, 25)}
band_llm_n = {k: _band_users[f"n_k{k}_accounts"].mean() for k in (12, 25)}
_band_cats = band_llm.drop("Overall")
def word_list(items):
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else "".join(items)


band_llm_sv_better = [CATEGORY_LABELS[c].lower().replace("tv ", "TV ") for c in _band_cats.index if _band_cats.at[c, "socialvec"] > _band_cats.at[c, "llm_k12"]]
band_llm_llm_better = [CATEGORY_LABELS[c].lower().replace("tv ", "TV ") for c in _band_cats.index if _band_cats.at[c, "socialvec"] <= _band_cats.at[c, "llm_k12"]]
band_llm_wins = {k: int((band_llm.drop("Overall")[f"llm_k{k}"] > band_llm.drop("Overall")[f"socialvec_k{k}"]).sum()) for k in (12, 25)}

# The same protocol on the top-20 candidates, for the users of Table 9 who follow at least one of them
_top_users = pd.read_csv(LLM / "band_per_user_1_20_original_n200_users_81_100_original.csv.gz")
_top_users["llm_baseline"] = _top_users["llm_baseline_shuffled"]
top_llm = by_category(_top_users, ["random", "popularity", "socialvec", "socialvec_k12", "socialvec_k25"] + LLM_COLS)
top_tests = {f"llm_k{k} vs socialvec_k{k}": wilcoxon(_top_users[f"llm_k{k}"] - _top_users[f"socialvec_k{k}"]).pvalue for k in (12, 25)}
top_tests["llm_k50 vs socialvec"] = wilcoxon(_top_users["llm_k50"] - _top_users["socialvec"]).pvalue
band_tests["llm_k50 vs socialvec"] = wilcoxon(_band_users["llm_k50"] - _band_users["socialvec"]).pvalue
top_llm_wins = {k: int((top_llm.drop("Overall")[f"llm_k{k}"] > top_llm.drop("Overall")[f"socialvec_k{k}"]).sum()) for k in (12, 25)}
BAND_METHODS = [("random", "Random ranking (expected)"), ("popularity", "Popularity (follower count)"),
                ("socialvec", "SocialVec, all followed entities"), ("socialvec_k12", "SocialVec, the 12 accounts of the LLM prompt"),
                ("socialvec_k25", "SocialVec, the 25 accounts of the LLM prompt"), ("llm_baseline", "LLM, no user information"),
                ("llm_k12", "LLM, 12 followed accounts"), ("llm_k25", "LLM, 25 followed accounts"), ("llm_k50", "LLM, 50 followed accounts")]


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
demo = json.loads((LLM / "demographics_summary.json").read_text())
dsh = {b: demo["shuffled"][b] for b in ("1-20", "81-100", "181-200")}
DEMO_METHODS = [("random", "Random ranking (expected)"), ("popularity", "Popularity (follower count)"),
                ("ap_baseline", "LLM, no user information"),
                ("ap_gpt-4o", "LLM, demographics by GPT-4o"), ("ap_k12", "LLM, 12 followed account names"),
                ("ap_k25", "LLM, 25 followed account names"), ("ap_k50", "LLM, 50 followed account names"),
                ("socialvec", "SocialVec, all followed entities")]
DEMO_COLORS = {"ap_gpt-4o": "s2", "ap_k12": "s3", "ap_k25": "s3", "ap_k50": "s3"}


def method_chart(o, methods, colors, context, x1=0.6):
    """Horizontal bars of overall MAP per method; methods in `colors` are highlighted."""
    W, L, Rm, T, rowh = 520, 230, 50, 6, 28
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
                     f'<text class="delta" x="{px(v)+8:.1f}" y="{y+16}"{weight}>{v:.3f}</text>'
                     f'<rect class="hit" x="{L}" y="{y+3}" width="{w:.1f}" height="18" data-tip="{esc(label)}, {context}<br>MAP <b>{v:.3f}</b>"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def demographics_chart(band):
    """Overall MAP per method with shuffled candidates (as in Section 4 of LLM_DEMOGRAPHICS_REPORT.html)."""
    return method_chart(dsh[band]["map"], DEMO_METHODS, DEMO_COLORS, f"ranks {band}, shuffled candidates", x1=0.8)


def demographics_table(bands=("1-20", "81-100")):
    """The numbers behind Figure 4: overall MAP per method and band."""
    head = "".join(f"<th>Ranks {b.replace('-', '–')}</th>" for b in bands)
    rows = [f"<tr><th>Users</th>" + "".join(f"<td>{dsh[b]['users']}</td>" for b in bands) + "</tr>"]
    for c, label in DEMO_METHODS:
        vals = [dsh[b]["map"][c] for b in bands]
        rows.append(f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{v:.3f}</td>" for v in vals) + "</tr>")
    for name, label in [("4o_vs_baseline", "p, demographics vs. no information"), ("k12_vs_4o", "p, 12 names vs. demographics"),
                        ("k50_vs_socialvec", "p, 50 names vs. SocialVec")]:
        rows.append(f"<tr class='muted'><th>{label}</th>" + "".join(f"<td>{p_text(dsh[b]['tests'][name])}</td>" for b in bands) + "</tr>")
    return (f"<div class='scroll'><table class='data compact'><thead><tr><th>Method</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# The exact chat messages dspy sends for the two stages, with placeholders for the inputs
def demographic_prompts():
    from reproduction.llm_demographics import make_demographic_programs
    _, _, render = make_demographic_programs(None)
    calls = {"describe": dict(followed_accounts="{names of up to 50 accounts the user follows, comma-separated}"),
             "rank": dict(user_description="{the description written by GPT-4o}",
                          potential_accounts="{JSON list of the candidates: twitter_user_id and twitter_name}")}
    out = {}
    for kind, inputs in calls.items():
        msgs = render(kind, **inputs)
        out[kind] = "\n\n".join(f"── {m_['role']} message ──\n" + re.sub(r"(?m)^ {8}", "", m_["content"]).strip() for m_ in msgs)
    return out


demo_prompts = demographic_prompts()
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


DEMO_CAT_COLS = [("Gender", "Female"), ("Gender", "Male"), ("Age", "Under 30"), ("Age", "30s"), ("Age", "Middle-aged"),
                 ("Ethnicity", "Caucasian / White"), ("Ethnicity", "African American / Black"), ("Education", "High school"),
                 ("Education", "Higher education, level not stated")]


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
  <a href="#demographics">5 · Demographics-only LLM</a><a href="#open">6 · Open items</a></nav>
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
  14 categories, link-prediction evaluation with MAP. Users follow {per_user['n_followed'].mean():,.0f} accounts on average,
  of which {per_user['n_followed_entities'].mean():,.0f} are SocialVec entities.</p>

  <h3>{ttitle('ranking')}</h3>
  <p>Every per-category value matches the paper to its three published decimals. The last two columns show the
  protocol as the paper describes it, with the target category's candidates actually removed from the user profile
  (see section 2).</p>
  <div class="card">{table2_html()}
  <p class="caption">Reproduced values; the paper's value is shown beneath each one. Δ is the gain over popularity.
  Sampled is the closed-world setting, where profiles use only the experiment's entities from the other 13 categories.</p></div>

  <h3>Figure 1 · How much data is needed?</h3>
  <p>MAP rises quickly with the number of entities in the user representation and levels off beyond about 100 sampled
  accounts. With k = 50 sampled accounts, SE1 reaches {fig1.loc['Overall','se1_legacy_k50']/ov['se1_all_legacy']:.1%} of its full-profile
  score (paper: 98.4%). In the closed world, 10 entities already give {fig1.loc['Overall','se2_k10']/ov['se2_all']:.1%} of the full score.
  The values are in {tnum('fig1_data')}.</p>
  <div class="card">
    <div class="legend"><span class="key"><i style="background:var(--s1)"></i>SE1 · paper setup</span>
    <span class="key"><i style="background:var(--s2)"></i>SE1 · target removed</span>
    <span class="key"><i style="background:var(--s3)"></i>SE2 · closed world</span>
    <span class="key"><i class="dia"></i>Value reported in the paper</span></div>
    {line_chart()}
    {table1d_html()}
    <details open><summary>Per-category panels</summary><img class="figure" alt="Figure 1 per category" src="data:image/png;base64,{fig1_png}"></details>
  </div>

  <h3>{ttitle('llm')}</h3>
  <p>I re-ran the LLM experiment with {model_name} (reasoning effort <i>{llm_effort}</i>) and the authors' prompts, on a stratified
  random sample of 50 users per category. The paper's finding replicates: listing 12 favored entities lifts MAP from
  {lo['llm_baseline']:.3f} to {lo['llm_k12']:.3f} ({pct(llm_gain['llm_k12'])}), and 50 accounts lift it to
  {lo['llm_k50']:.3f} ({pct(llm_gain['llm_k50'])}). The paper reported +13% and +23% for GPT-4o. The social-embedding column (SE1, k = 50) matches
  the paper on all users: {fig1.loc[PAPER_CATEGORY_ORDER, 'se1_legacy_k50'].mean():.3f} vs. 0.562. On the same 700 users, SocialVec scores
  {lo['socialvec_k12']:.3f} with the 12 accounts and {lo['socialvec_k25']:.3f} with the 25 accounts listed in the LLM prompts
  (k = 50: {lo['se1_legacy_k50']:.3f}), so the LLM is ahead at 12 and 25 accounts, as it is at ranks 81–100 ({tnum('band_llm')}).</p>
  <div class="card">
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s1)"></i>{model_name}, 700 sampled users</span>
    <span class="key"><i class="sq" style="background:var(--s2)"></i>GPT-4o, as reported in the paper</span></div>
    {llm_chart()}
    <details open><summary>Show {tnum('llm')} (per category)</summary>{table3_html()}</details>
    <p class="caption">Overall MAP (mean over 14 categories). The dashed reference lines are computed on the same 700 users.
    The baseline (no user information) is the mean of 10 calls per category with the candidates in random order (Section 4);
    with the candidates sorted by follower count it scores {_trials[_trials['order_type'] == 'original'].groupby('topic')['MAP'].mean().mean():.3f}.
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
  User profiles exclude <i>every</i> listed entity of the target category. As a matched control, I ranked the original
  top 20 for the same users with the same profiles. A random ranking (exact expected AP) provides a chance level, and
  paired Wilcoxon tests compare the two methods.</p>

  <ul class="highlights">
    <li><b>Popularity breaks down in the tail.</b> At ranks 81–100 it scores {A_b['popularity']:.3f}, barely above a random
    ranking ({A_b['random']:.3f}), and below it for {', '.join(below[:-1])} and {below[-1]}.</li>
    <li><b>Social similarity holds up.</b> It scores {A_b['socialvec']:.3f}, well above chance in every category, and beats
    popularity in 8 of 9 categories. Its relative gain grows from {pct(A_h['gain'])} on the top 20 to {pct(A_b['gain'])}.</li>
    <li><b>The result holds on a population ten times larger.</b> On all experiment users ({B_b['users']:,.0f} user–category
    pairs), the gain is {pct(B_b['gain'])} vs. {pct(B_h['gain'])}, and every per-category difference is significant (p &lt; 0.001).</li>
    <li class="warn"><b>The exception is Comedians ({pct(mA['band'].loc['Comedian','gain'])}).</b> Follows in that
    band concentrate on its two most-followed accounts, both mainly political commentators, which popularity ranks first.</li>
  </ul>

  <div class="card">
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s2)"></i>Popularity</span>
    <span class="key"><i class="sq" style="background:var(--s1)"></i>Social similarity</span>
    <span class="key"><i class="tick"></i>Random ranking (expected)</span></div>
    <div class="panels">
      <div class="panel"><h4>Original top 20</h4><p class="note">Same users and profiles as on the right</p>{bar_panel(mA['head'], 'Ranks 1–20')}</div>
      <div class="panel"><h4>Ranks 81–100</h4><p class="note">Users sampled for each category</p>{bar_panel(mA['band'], 'Ranks 81–100')}</div>
    </div>
    <p class="caption">MAP per category; Δ is the gain of social similarity over popularity. The values are in {tnum('band_sampled')} below.</p>
  </div>

  <h3>{ttitle('band_sampled')}</h3>
  <div class="card">{matched_html(mA)}
  <p class="caption">{A_b['users']:,.0f} users who follow at least one entity ranked 81–100 in their category. Rel./user is the number of
  candidates a user follows. Dotted red marks popularity scores below the random ranking. p is from a Wilcoxon signed-rank test,
  social vs. popularity.</p></div>

  <h3>{ttitle('band_all')}</h3>
  <p>A robustness check. Each of the 11,153 users is evaluated in every category in which they follow at least one entity from each band.
  That gives {B_b['users']:,.0f} user–category pairs from {n_distinct_users:,} distinct users. Only {A_b['users']/B_b['users']:.0%} of these pairs
  come from users sampled for the category, so this is a largely independent check. The overall test pools all pairs.
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
    <span class="key"><i class="tick"></i>Median</span></div>
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
  candidate ({band_llm.loc['Overall','users']:,.0f} users; TV stations have only 198). Every listed entity of the
  target category is removed from the prompts. Without user information, the LLM is no better than popularity
  ({band_llm.loc['Overall','llm_baseline']:.3f} vs. {band_llm.loc['Overall','popularity']:.3f}). With user information it personalizes well:
  {band_llm.loc['Overall','llm_k12']:.3f} with 12 entities and {band_llm.loc['Overall','llm_k50']:.3f} with 50 accounts. Social similarity from all of
  the user's entities scores {band_llm.loc['Overall','socialvec']:.3f}. The two are complementary: with 12 entities the LLM is stronger
  for {word_list(band_llm_llm_better)}; social similarity from all entities for {word_list(band_llm_sv_better)}.</p>
  <p>For a like-for-like comparison, the <i>SocialVec k = 12</i> and <i>k = 25</i> columns build the user embedding from exactly the
  accounts listed in the LLM's k = 12 and k = 25 prompts ({band_llm_n[12]:.1f} and {band_llm_n[25]:.1f} on average, as some users have fewer).
  Given the same evidence, the LLM ranks better at both sizes:
  {band_llm.loc['Overall','llm_k12']:.3f} vs. {band_llm.loc['Overall','socialvec_k12']:.3f} with 12 accounts
  (paired Wilcoxon p {p_text(band_tests['llm_k12 vs socialvec_k12'])}; ahead in {band_llm_wins[12]} of 9 categories), and
  {band_llm.loc['Overall','llm_k25']:.3f} vs. {band_llm.loc['Overall','socialvec_k25']:.3f} with 25
  (p {p_text(band_tests['llm_k25 vs socialvec_k25'])}; ahead in {band_llm_wins[25]} of 9).</p>
  <div class="card">{band_llm_html()}
  <p class="caption">Bold marks the best method in each row. With 200 users per category, per-category MAP has a standard error of
  about 0.02. The first 30 users of each category are those of Section 5 and of the earlier 30-user version of this table. The baseline (no user information) gets the candidates in a random order per user, as in Section 5; the personalized prompts list
  them by follower count.</p></div>

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
  {band_llm.loc['Overall','users'] - top_llm.loc['Overall','users']:,.0f} users follow none of the top-20 candidates. Both bands use one random candidate order per user for the baseline
  and the follower-count order for the personalized prompts. p values are paired Wilcoxon tests over users.</p>
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
  <p class="caption">{llm_calls:,} model calls in total across all LLM experiments (including those in the backup page), costing ${llm_cost:.2f}.</p>
</section>

<section id="demographics">
  <h2><span class="num">5</span>Can demographics alone personalize the LLM?</h2>
  <p class="intro">A summary of <a href="LLM_DEMOGRAPHICS_REPORT.html">the demographics report</a>, which has the method,
  per-category tables and examples.</p>
  <p>For {demo['users']} of the users of {tnum('band_llm')} (its first 30 per category), GPT-4o first guessed each user's gender, age, ethnicity and education from the
  names of accounts they follow, and was told to rely on stereotypes rather than decline. {model_name} then ranked the candidates
  given only that description, both for the most popular candidates of each category (ranks 1–20, as in {tnum('llm')}) and for ranks 81–100. The final runs list
  the candidates in one random order per user, shared by every condition. For comparison, the same shuffled candidates were also
  ranked from the names of 12, 25 or 50 followed accounts.</p>

  <h3>Prompts</h3>
  <p>Both stages use dspy signatures, as in the original experiments. These are the exact messages dspy sends, with the inputs shown
  as placeholders in braces.</p>
  <div class="card"><h4 style="margin:0 0 6px">Stage 1 · Generating the demographic description (GPT-4o)</h4>
  <pre class="prompt" style="white-space:pre-wrap;font-size:12.5px;margin:0">{esc(demo_prompts['describe'])}</pre></div>
  <div class="card"><h4 style="margin:0 0 6px">Stage 2 · Ranking the candidates from the description ({model_name})</h4>
  <pre class="prompt" style="white-space:pre-wrap;font-size:12.5px;margin:0">{esc(demo_prompts['rank'])}</pre>
  <p class="caption">The model's answer is read from the <code>[[ ## ranked_ids ## ]]</code> field. The ranking prompts with account
  names and without user information are the authors' prompts used throughout this page.</p></div>

  <ul class="highlights">
    <li class="warn"><b>The descriptions are not always demographics only.</b> GPT-4o described all {demo['users']} users, but
    {demo['leak_4o']} of its descriptions also named the user's interests, against the instruction.</li>
    <li><b>Demographics help once the order is fair.</b> With shuffled candidates, GPT-4o's descriptions beat the no-information
    ranking: {dsh['81-100']['map']['ap_gpt-4o']:.3f} vs. {dsh['81-100']['map']['ap_baseline']:.3f} (p {p_text(dsh['81-100']['tests']['4o_vs_baseline'])}).
    Its demographics-only descriptions do too ({dsh['81-100']['clean_4o']['map']:.3f} vs. {dsh['81-100']['clean_4o']['baseline']:.3f}).</li>
    <li><b>The gain is smaller among the most popular candidates.</b> At ranks 1–20 the no-information ranking is already strong
    ({dsh['1-20']['map']['ap_baseline']:.3f}, above popularity at {dsh['1-20']['map']['popularity']:.3f}, p {p_text(dsh['1-20']['tests']['baseline_vs_popularity'])}).
    GPT-4o's descriptions still add to it ({dsh['1-20']['map']['ap_gpt-4o']:.3f}, p {p_text(dsh['1-20']['tests']['4o_vs_baseline'])}), a gain of
    {dsh['1-20']['map']['ap_gpt-4o'] - dsh['1-20']['map']['ap_baseline']:+.3f} against {dsh['81-100']['map']['ap_gpt-4o'] - dsh['81-100']['map']['ap_baseline']:+.3f} at
    ranks 81–100, and come close to SocialVec's full profile ({dsh['1-20']['map']['socialvec']:.3f}, p {p_text(dsh['1-20']['tests']['4o_vs_socialvec'])}).</li>
    <li><b>Sorted candidates hid the gain.</b> With candidates listed by popularity, the no-information ranking scored
    {demo['order_effect']['baseline']['sorted']:.3f}; shuffled, it drops to {demo['order_effect']['baseline']['shuffled']:.3f}
    (p {p_text(demo['order_effect']['baseline']['p'])}). The rankings that use user information barely change.</li>
    <li><b>Account names say more than demographics.</b> 12 names score {dsh['81-100']['map']['ap_k12']:.3f}, above GPT-4o's
    descriptions (p {p_text(dsh['81-100']['tests']['k12_vs_4o'])}).</li>
    <li><b>50 names beat SocialVec's full profile.</b> {dsh['81-100']['map']['ap_k50']:.3f} vs. {dsh['81-100']['map']['socialvec']:.3f}
    (p {p_text(dsh['81-100']['tests']['k50_vs_socialvec'])}).</li>
  </ul>

  <div class="card">
    <h4 style="margin:0 0 6px">Figure 4 · Ranking from demographics vs. account names, ranks 1–20 and 81–100</h4>
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s2)"></i>Demographics by GPT-4o</span>
    <span class="key"><i class="sq" style="background:var(--s3)"></i>Followed account names</span></div>
    <h4 style="margin:10px 0 2px">Ranks 1–20 · {dsh['1-20']['users']} users</h4>
    <div style="max-width:620px">{demographics_chart('1-20')}</div>
    <h4 style="margin:10px 0 2px">Ranks 81–100 · {dsh['81-100']['users']} users</h4>
    <div style="max-width:620px">{demographics_chart('81-100')}</div>
    <p class="caption">Overall MAP (mean over categories) with the candidates in random order. Each band uses the users who follow at
    least one of its candidates.</p>
  </div>
  <div class="card"><h4 style="margin:0 0 6px">{ttitle('demo_bands')}</h4>{demographics_table()}
  <p class="caption">Overall MAP is the mean over categories; p values are paired Wilcoxon tests over users.</p></div>

  <h3>What GPT-4o stated about the users</h3>
  <p>Each description is parsed with fixed rules for the first value it states for each attribute. A description that hedges, such as
  "in his 30s or 40s" or "Caucasian or mixed", counts under its first value; {tnum('demo_attrs')} also counts how many
  descriptions give more than one. "Middle-aged" is kept as stated, without assigning it an age range.</p>
  <ul class="highlights">
    <li><b>The guesses are concentrated.</b> {demo_attrs['Ethnicity']['counts'].get('Caucasian / White', 0)} of {demo['users']} descriptions say
    Caucasian or white and {demo_attrs['Education']['counts'].get('Higher education, level not stated', 0)} say higher education without a level.
    Gender is the most balanced attribute ({demo_attrs['Gender']['counts'].get('Male', 0)} male, {demo_attrs['Gender']['counts'].get('Female', 0)} female).</li>
    <li><b>Ages are vague.</b> {demo_attrs['Age']['multiple']} descriptions give an age range spanning two groups, and
    {demo_attrs['Age']['counts'].get('Middle-aged', 0)} say only "middle-aged".</li>
  </ul>
  <div class="card">
    <h4 style="margin:0 0 6px">Figure 5 · Demographic attributes stated in GPT-4o's descriptions · {demo['users']} users</h4>
    {demo_attrs_chart()}
    <p class="caption">Share of descriptions by the first value stated for each attribute. Hover a segment for the count.</p>
  </div>
  <h3>{ttitle('demo_attrs')}</h3>
  <div class="card">{demo_attrs_table()}
  <p class="caption">Education counts a specific level (high school, some higher education, a degree) when one is stated anywhere in the
  description, and "level not stated" otherwise.</p></div>
  <h3>{ttitle('demo_attrs_cat')}</h3>
  <div class="card">{demo_attrs_cat_table()}
  <p class="caption">Share of the category's 30 descriptions stating each value (first stated value). Columns for rarer values are left
  out; they are in {tnum('demo_attrs')}.</p></div>
  <p class="caption">The descriptions are stereotype-based guesses about real people. They are kept outside the repository and
  should not be read as findings about which groups like which entities.</p>
</section>

<section id="open">
  <h2><span class="num">6</span>Open items</h2>
  <div class="callouts">
    <div class="callout"><span class="tag">Scale</span><h4>LLM experiments on all users</h4>
    <p>The LLM results use samples of 50 users per category (30 for ranks 81–100), which keeps overall MAP within
    about ±0.01. A full-population run takes about 45K calls. The same command runs it with larger samples, and cached
    responses are reused.</p></div>
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
(R / "results.html").write_text(page, encoding="utf-8")
print("wrote", R / "results.html", f"{len(page)/1e6:.2f} MB")


# ---------------------------------------------------------------- results_backup.html: the extended category lists
backup = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Extended Lists Results</title><style>{CSS}</style></head>
<body><div id="tip" role="tooltip"></div>
<div class="wrap">
<header class="hero">
  <p class="eyebrow">Backup · results with the extended category lists</p>
  <h1>Results that use the extended category lists</h1>
  <p class="lede">These experiments use category lists of up to 1,000 accounts built from Wikidata and DBpedia, without an LLM and
  without manual validation (see <code>results/EXTENDED_CATEGORIES_REPORT.md</code>). They are kept here for reference and are not
  part of <a href="results.html">the main results</a>. Section and table numbers refer to this page unless they name the main page.</p>
  <nav class="toc"><a href="#tail2">B1 · Ranks 181–200</a><a href="#popularity">B2 · Global popularity of the bands</a>
  <a href="#demographics181">B3 · Demographics-only LLM, ranks 181–200</a><a href="#all200">B4 · Ranking all 200 candidates</a></nav>
</header>

<section id="tail2">
  <h2><span class="num">B1</span>Further down the category lists · Ranks 181–200</h2>
  <p class="intro">The original sheets stop at 100 entities per category. To go further, I repeated the ranks 81–100 experiments of the
  main page with the accounts ranked <b>181–200</b> in the extended lists.</p>
  <ul>
    <li><i>Categories:</i> the {len(mA2['band']) - 1} categories whose extended list reaches 200 accounts.</li>
    <li><i>Ranking:</i> the extended lists rank accounts by SocialVec popularity, while ranks 81–100 were by Twitter follower count.
    The median candidate is the {band2_sv_median:,}th most popular SocialVec entity, and {band2_below_10k:.0%} of the candidates lie
    beyond the top 10,000. Globally, these candidates are more popular than those of ranks 81–100 (Section B2): the lists
    differ, so a lower rank within a category does not mean a less popular account.</li>
    <li><i>Profiles:</i> user profiles exclude every account of the category's extended list (up to 1,000) and of its original
    sheet.</li>
  </ul>

  <ul class="highlights">
    <li><b>Popularity ranks worse than chance.</b> At ranks 181–200 it scores {A2_b['popularity']:.3f} against
    {A2_b['random']:.3f} for a random ranking, and falls below chance in {len(below2)} of {len(mA2['band']) - 1} categories.</li>
    <li><b>Social similarity still personalizes well.</b> It scores {A2_b['socialvec']:.3f}, {pct(A2_b['gain'])} over popularity,
    significant in every category (p &lt; 0.001), compared with {pct(A2_h['gain'])} on the top 20 for the same users.</li>
    <li><b>The same holds on all users.</b> On {B2_b['users']:,.0f} user–category pairs, the gain is {pct(B2_b['gain'])}, vs.
    {pct(B2_h['gain'])} on the top 20.</li>
    <li><b>The LLM pattern repeats</b> ({tnum('band2_llm')}).
    <ul>
      <li><i>No user information:</i> {model_name} stays close to chance ({band2_llm.loc['Overall','llm_baseline']:.3f} vs.
      {band2_llm.loc['Overall','random']:.3f} for a random ranking).</li>
      <li><i>12 or 25 account names:</i> it beats SocialVec on the same accounts
      ({band2_llm.loc['Overall','llm_k12']:.3f} vs. {band2_llm.loc['Overall','socialvec_k12']:.3f} and
      {band2_llm.loc['Overall','llm_k25']:.3f} vs. {band2_llm.loc['Overall','socialvec_k25']:.3f}).</li>
      <li><i>50 accounts:</i> it scores about as well as SocialVec's full profile ({band2_llm.loc['Overall','llm_k50']:.3f} vs.
      {band2_llm.loc['Overall','socialvec']:.3f}; p = {k50_vs_full:.2f}, not significant).</li>
    </ul></li>
  </ul>

  <h3>{ttitle('bands_overview')}</h3>
  <div class="card">{bands_overview()}
  <p class="caption">Each lower band is compared with the original top 20 on the same users (its matched population), so
  the two top-20 rows differ. Dotted red marks popularity below the random ranking.</p></div>

  <div class="card">
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s2)"></i>Popularity</span>
    <span class="key"><i class="sq" style="background:var(--s1)"></i>Social similarity</span>
    <span class="key"><i class="tick"></i>Random ranking (expected)</span></div>
    <div class="panels">
      <div class="panel"><h4>Original top 20</h4><p class="note">Same users and profiles as on the right</p>{bar_panel(mA2['head'], 'Ranks 1–20')}</div>
      <div class="panel"><h4>Ranks 181–200</h4><p class="note">Users sampled for each category</p>{bar_panel(mA2['band'], 'Ranks 181–200')}</div>
    </div>
    <p class="caption">MAP per category; Δ is the gain of social similarity over popularity. The values are in {tnum('band2_sampled')} below.</p>
  </div>

  <h3>{ttitle('band2_sampled')}</h3>
  <div class="card">{matched_html(mA2, 'Ranks 181–200')}
  <p class="caption">{A2_b['users']:,.0f} users who follow at least one entity ranked 181–200 in their category. Columns as in
  {tnum('band_sampled')} of the main page.</p></div>

  <h3>{ttitle('band2_all')}</h3>
  <div class="card">{matched_html(mB2, 'Ranks 181–200')}{entities_html(band2_entities, 'band2_entities')}</div>

  <h3>{ttitle('band2_llm')}</h3>
  <p>The protocol of {tnum('band_llm')} in the main page, for 30 random users per category, with every account of the category's
  extended list and original sheet removed from the prompts. The baseline (no user information) is the mean of 10 calls per category,
  each with the candidates in a different random order. The SocialVec k = 12 and k = 25 columns use exactly the accounts in the LLM's prompts ({band2_llm_n[12]:.1f} and
  {band2_llm_n[25]:.1f} on average). Given the same evidence, the LLM ranks better at both sizes:
  p {p_text(band2_tests['llm_k12 vs socialvec_k12'])} with 12 accounts, ahead in {band2_llm_wins[12]} of {len(band2_llm) - 1} categories;
  p {p_text(band2_tests['llm_k25 vs socialvec_k25'])} with 25, ahead in {band2_llm_wins[25]} of {len(band2_llm) - 1}.</p>
  <div class="card">{band_llm_html(band2_llm)}
  <p class="caption">Bold marks the best method in each row. With 30 users per category, per-category MAP has a standard error of
  about 0.05. The baseline uses shuffled candidates (<code>run_band_shuffled_baseline.py</code>); the personalized prompts list them in
  the extended list's order. These {model_name} runs cost ${band2_cost:.2f}.</p></div>

  <h3>Limitations</h3>
  <p>The extended lists are not manually validated. Known errors, such as people with acting credits listed as actors, team PR
  accounts and DBpedia mistypings, are described in the extended-lists report; they affect the candidate sets here. The
  LLM results use 30 users per category.</p>
</section>

<section id="popularity">
  <h2><span class="num">B2</span>Global popularity of the candidate bands</h2>
  <p class="intro">Each band's position is set within its own category list. To compare bands across lists, every candidate is placed
  in the popularity ranking of all {pop_n_vocab:,} SocialVec accounts (rank 1 is the most followed).</p>
  <ul class="highlights">
    <li class="warn"><b>Ranks 181–200 are not less popular than ranks 81–100.</b> Their median global rank is
    {pop_overall[POP_BANDS[2][0]]['median']:,.0f}, against {pop_overall[POP_BANDS[1][0]]['median']:,.0f} for ranks 81–100 (Mann–Whitney
    p {p_text(pop_mw_p)}), and their median follower count is {pop_overall[POP_BANDS[2][0]]['followers']:,.0f} against
    {pop_overall[POP_BANDS[1][0]]['followers']:,.0f}. They are globally more popular in {pop_more_popular_cats} of the {pop_both_cats}
    categories that have both bands.</li>
    <li><b>Why:</b> the original sheets hold 100 manually validated accounts per category, while the extended lists hold up to 1,000
    accounts typed by Wikidata and DBpedia. Many popular accounts are missing from the original sheets (only 44–85 of each sheet's 100
    fall in the extended top 200), so position 181 in an extended list can be more popular than position 81 in an original sheet.
    Because the extended lists are sorted by this same global rank, each category's 181–200 band is a narrow slice of it, either
    entirely inside or entirely outside the top 10,000.</li>
  </ul>
  <div class="card">
    <h4 style="margin:0 0 6px">Figure B1 · Share of candidates within the global top N</h4>
    <div class="legend">{''.join(f'<span class="key"><i class="sq" style="background:var(--{cls})"></i>{esc(b)}</span>' for b, cls in POP_BANDS)}</div>
    <div style="max-width:620px">{pop_cdf_chart()}</div>
    <p class="caption">Cumulative share of each band's candidates (all categories) at or above each global rank; candidates without a
    SocialVec embedding count as least popular.</p>
  </div>
  <h3>{ttitle('popularity_ext')}</h3>
  <div class="card">{pop_table([b for b, _ in POP_BANDS], pop_overall)}
  <p class="caption">Medians and shares per category; Overall pools all candidates of the band.</p></div>
</section>

<section id="demographics181">
  <h2><span class="num">B3</span>Demographics-only LLM ranking, ranks 181–200</h2>
  <p class="intro">The shuffled-candidate rankings of Section 5 of the main page, repeated on ranks 181–200 of the extended lists for the
  {dsh['181-200']['users']} users who follow at least one of those candidates, with GPT-4o's descriptions. The method is described in
  <a href="LLM_DEMOGRAPHICS_REPORT.html">the demographics report</a>, which covers ranks 81–100 only.</p>
  <ul class="highlights">
    <li><b>Demographics help:</b> GPT-4o's descriptions score {dsh['181-200']['map']['ap_gpt-4o']:.3f} against
    {dsh['181-200']['map']['ap_baseline']:.3f} without user information (p {p_text(dsh['181-200']['tests']['4o_vs_baseline'])}); its
    demographics-only descriptions {dsh['181-200']['clean_4o']['map']:.3f} vs. {dsh['181-200']['clean_4o']['baseline']:.3f}.</li>
    <li><b>Account names:</b> 12 names score {dsh['181-200']['map']['ap_k12']:.3f}, not significantly above GPT-4o's descriptions
    (p {p_text(dsh['181-200']['tests']['k12_vs_4o'])}). 50 names score {dsh['181-200']['map']['ap_k50']:.3f} against
    {dsh['181-200']['map']['socialvec']:.3f} for SocialVec's full profile (p {p_text(dsh['181-200']['tests']['k50_vs_socialvec'])}).</li>
  </ul>
  <div class="card">
    <h4 style="margin:0 0 6px">Figure B2 · Ranking from demographics vs. account names, ranks 181–200 · {dsh['181-200']['users']} users</h4>
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s2)"></i>Demographics by GPT-4o</span>
    <span class="key"><i class="sq" style="background:var(--s3)"></i>Followed account names</span></div>
    <div style="max-width:620px">{demographics_chart('181-200')}</div>
    <p class="caption">Overall MAP (mean over categories) with the candidates in random order.</p>
  </div>
</section>

<section id="all200">
  <h2><span class="num">B4</span>Ranking all 200 candidates</h2>
  <p class="intro">Does the LLM struggle with longer prompts? Here it must rank 200 candidates per category instead of 20.
  Worked examples are in <a href="LLM_200_EXAMPLES.html">a separate report</a>.</p>
  <ul>
    <li><i>Candidates:</i> ranks 1–200 of the extended category lists (by SocialVec popularity), for the
    {len(c200_cat)} categories that reach 200 accounts and have sampled users; {c200['n_candidates'].mean():.1f} candidates per
    category on average, as entities without a SocialVec embedding are dropped.</li>
    <li><i>Users:</i> {len(c200)} of the users of {tnum('band_llm')} in the main page (its first 30 per category), all of whom follow at least one candidate
    ({c200['n_relevant'].mean():.1f} on average).</li>
    <li><i>LLM:</i> {model_name} at reasoning effort <i>medium</i>, with the candidates in one random order per user shared by
    both prompts. <i>No user information</i> uses the baseline prompt. <i>Account names</i> lists up to 200 accounts the user
    follows ({c200['n_followed_listed'].mean():.0f} on average), sampled among accounts with a known name, with every account of
    the category's extended list and original sheet removed.</li>
    <li><i>A stronger prompt:</i> with the authors' prompts at effort <i>low</i>, a pilot returned only 20–50 of the 200 ids
    when given 200 names, cutting the list off itself. Both prompts therefore add an explicit instruction to return every
    candidate exactly once.</li>
    <li><i>Scoring incomplete answers:</i> only the candidates the model actually ranked count. A relevant candidate it left out
    is a miss: it adds nothing to AP but stays in the denominator. Ids that are not candidates are dropped.</li>
    <li><i>SocialVec:</i> on the same accounts as the LLM's prompt, and on all followed entities (category accounts removed).
    Popularity ranks by follower count.</li>
  </ul>

  <ul class="highlights">
    <li><b>With 200 candidates, the LLM keeps a small lead.</b> Given account names it scores {c200_o['llm_k200']:.3f},
    against {c200_o['socialvec_k200']:.3f} for SocialVec on the same accounts (p {p_text(c200_p['k200_vs_sv200'])}) and
    {c200_o['socialvec']:.3f} on all followed entities (p {p_text(c200_p['k200_vs_sv'])}). It is ahead in {c200_wins} of
    {len(c200_cat)} categories. Placing the left-out candidates at the end in random order instead would give it
    {c200_appended['k200']:.3f}, overstating the lead.</li>
    <li><b>Without user information it is no better than popularity</b> ({c200_o['llm_baseline']:.3f} vs. {c200_o['popularity']:.3f},
    p {p_text(c200_p['baseline_vs_popularity'])}), though far above chance ({c200_o['random']:.3f}).</li>
    <li class="warn"><b>Long lists are harder to complete</b> ({tnum('c200_stats')}).
    <ul>
      <li><i>Coverage:</i> even with the explicit instruction, only {(c200_frac['k200'] == 1).mean():.0%} of the name-based answers
      list every candidate, and {(c200_frac['k200'] >= 0.95).mean():.0%} list at least 95% of them
      ({(c200_frac['baseline'] == 1).mean():.0%} and {(c200_frac['baseline'] >= 0.95).mean():.0%} without user information). Across
      users, {1 - c200_rel_ranked['k200']:.0%} of the relevant candidates are missing from the name-based answers.</li>
      <li><i>Short answers fall behind:</i> for the {len(c200_short_list)} users whose answer stops before 60% of the candidates, the LLM
      scores {c200_short_list['llm_k200'].mean():.3f} against {c200_short_list['socialvec_k200'].mean():.3f} for SocialVec on the same
      accounts.</li>
      <li><i>Complete answers:</i> on the {len(c200_complete)} users whose two answers are both at least 95% complete, the lead is
      {c200_complete_o['llm_k200']:.3f} vs. {c200_complete_o['socialvec_k200']:.3f}, not significant (p {p_text(c200_complete_p)}).</li>
    </ul></li>
    <li><b>On the hardest candidates, the LLM does worse in the longer list</b> ({tnum('c200_subset')}). Cutting each 200-candidate
    ranking down to the ranks 181–200 candidates allows a comparison with the 20-candidate prompts of Section B3, on the
    {len(c200_sub)} users who follow one of them; both are scored with the rule above.
    <ul>
      <li><i>With names:</i> the LLM drops from {c200_sub_m['short_k50_181_200']:.3f} to {c200_sub_m['llm_k200_181_200']:.3f}
      (p {p_text(c200_sub_p['llm_k200_181_200'])}), although it now sees up to 200 names instead of 50.</li>
      <li><i>Without user information:</i> it drops from {c200_sub_m['short_baseline_181_200']:.3f} to
      {c200_sub_m['llm_baseline_181_200']:.3f} (p {p_text(c200_sub_p['llm_baseline_181_200'])}).</li>
      <li><i>SocialVec</i> does not depend on the list length ({c200_sub_m['socialvec_20']:.3f} vs.
      {c200_sub_m['socialvec_k200_181_200']:.3f}). The LLM's lead over it on these candidates shrinks from
      {signed_short(c200_sub_m['short_k50_181_200'] - c200_sub_m['socialvec_20'])} (p {p_text(c200_gap_p['short'])}) to
      {signed_short(c200_sub_m['llm_k200_181_200'] - c200_sub_m['socialvec_k200_181_200'])} (p {p_text(c200_gap_p['long'])}).</li>
    </ul></li>
  </ul>

  <div class="card">
    <h4 style="margin:0 0 6px">Figure B3 · Ranking all 200 candidates · {len(c200)} users</h4>
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s3)"></i>{model_name}, account names</span>
    <span class="key"><i class="sq" style="background:var(--s1)"></i>SocialVec</span></div>
    <div style="max-width:620px">{method_chart(c200_o, C200_METHODS, C200_COLORS, "200 candidates")}</div>
    <p class="caption">Overall MAP (mean over categories) when ranking all 200 candidates. Per-category values are in {tnum('c200_category')}.</p>
  </div>

  <h3>How many candidates did the LLM rank?</h3>
  <p>Answers often contain about 200 ids, but some are repeats, so fewer distinct candidates are ranked. Only the first position of
  a repeated id counts, and ids that are not candidates are dropped. The number ranked does not depend on the other prompt's answer
  for the same user (correlation {c200['returned_baseline'].corr(c200['returned_k200']):.2f}) or on the number of names listed
  ({c200['returned_k200'].corr(c200['n_followed_listed']):.2f}).</p>
  <div class="card">{c200_coverage_table()}
  <p class="caption">Counts are distinct candidate ids per answer. 95% means at least 190 of 200.</p></div>

  <h3>{ttitle('c200_subset')}</h3>
  <div class="card">{c200_subset_table()}
  <p class="caption">MAP on the same 20 candidates for the {len(c200_sub)} users who follow one of them (mean over
  categories). The 20-candidate prompts are those of Section B3 (authors' prompt, effort low). The two LLM runs also differ
  in prompt wording, reasoning effort and number of names, so the comparison is not a pure test of list length.</p></div>

  <h3>{ttitle('c200_category')}</h3>
  <div class="card">{c200_category_table()}
  <p class="caption">MAP per category over 200 candidates; bold marks the best method in each row. These {model_name} runs,
  including the pilots and a first run that listed fewer names, cost ${c200_cost:.2f}.</p></div>
</section>

<footer>
  <p>Build the extended lists with <code>python build_extended_categories.py</code>. For Section B1, run
  <code>python run_popularity_band.py --start 181 --end 200 --source extended</code> and
  <code>python run_llm_experiments.py --parts band --band-start 181 --band-end 200 --band-source extended</code>; Section B3 comes from
  <code>python run_llm_demographics_bands.py</code>; for Section B4, run
  <code>python run_popularity_band.py --start 1 --end 200 --source extended</code> and <code>python run_llm_200.py</code>.
  This page is written by <code>python make_results_page.py</code> together with <a href="results.html">results.html</a>.</p>
</footer>
</div>
<script>{JS}</script>
</body></html>
"""
(R / "results_backup.html").write_text(backup, encoding="utf-8")
print("wrote", R / "results_backup.html", f"{len(backup)/1e6:.2f} MB")
