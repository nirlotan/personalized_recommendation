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

import numpy as np
import pandas as pd

from reproduction import paper_results as paper
from reproduction.config import CATEGORY_LABELS, MODELS_DIR, PAPER_CATEGORY_ORDER, ROOT

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
per_user = pd.read_csv(R / "per_user_ap.csv.gz", usecols=["n_followed", "n_followed_entities", "n_closed_world", "n_relevant"])

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


def p_text(p):
    """A p-value for running text: '< 0.001' or '= 0.006'."""
    return pval(p) if p < 0.001 else f"= {p:.3f}"


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
            f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th><td>{ent}</td>"
            f"<td>{with_paper(o['popularity'], ref['popularity'])}</td>"
            f"<td>{with_paper(o['se1_all_legacy'], ref['all'])}</td><td{neg(o['se1_all_legacy_gain'])}>{pct(o['se1_all_legacy_gain'])}</td>"
            f"<td>{with_paper(o['se2_all'], ref['sampled'])}</td><td{neg(o['se2_all_gain'])}>{pct(o['se2_all_gain'])}</td>"
            f"<td><span class='num'>{o['se1_all']:.3f}</span></td><td{neg(o['se1_all_gain'])}>{pct(o['se1_all_gain'])}</td></tr>")
    return ("<div class='scroll'><table class='data'><thead><tr><th>Category</th><th>Entities / user</th><th>Popularity</th>"
            "<th>Social · All</th><th>Δ</th><th>Social · Sampled</th><th>Δ</th><th>All, target removed</th><th>Δ</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def matched_html(tables):
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
            "<tr class='super'><th></th><th></th><th colspan='5'>Ranks 1–20 (original candidates)</th><th colspan='6' class='grp'>Ranks 81–100</th></tr>"
            "<tr><th>Category</th><th>Users</th><th>Rel./user</th><th>Popularity</th><th>Social</th><th>Δ</th><th>p</th>"
            "<th class='grp'>Rel./user</th><th>Random</th><th>Popularity</th><th>Social</th><th>Δ</th><th>p</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


LLM_COLS = ["llm_baseline", "llm_k12", "llm_k25", "llm_k50"]
LLM_LABELS = ["Baseline", "k = 12", "k = 25", "k = 50"]


def table3_html():
    """GPT-6 Luna on the 700-user sample next to the paper's GPT-4o values."""
    rows = []
    for cat in PAPER_CATEGORY_ORDER + ["Overall"]:
        r = llm_t3.loc[cat]
        ref = paper.TABLE3.loc[cat] if cat != "Overall" else paper.TABLE3.mean()
        cls = " class='total'" if cat == "Overall" else ""
        rows.append(f"<tr{cls}><th>{esc(CATEGORY_LABELS.get(cat, 'Overall'))}</th><td>{r['popularity']:.3f}</td><td>{r['se1_legacy_k50']:.3f}</td>"
                    + "".join(f"<td{' class=grp' if i == 0 else ''}>{r[c]:.3f}</td>" for i, c in enumerate(LLM_COLS))
                    + "".join(f"<td class='muted{' grp' if i == 0 else ''}'>{ref[c]:.3f}</td>" for i, c in enumerate(LLM_COLS)) + "</tr>")
    gains = "".join(f"<td>{pct(llm_stats.loc[c, 'gain vs LLM baseline'])}</td>" for c in LLM_COLS[1:])
    paper_gains = "".join(f"<td class='muted'>{pct(paper.TABLE3[c].mean() / paper.TABLE3['llm_baseline'].mean() - 1)}</td>" for c in LLM_COLS[1:])
    rows.append(f"<tr><th>Δ vs. LLM baseline</th><td></td><td></td><td class='grp'></td>{gains}<td class='grp'></td>{paper_gains}</tr>")
    return ("<div class='scroll'><table class='data'><thead><tr class='super'><th></th><th colspan='2'>Same 700 users</th>"
            f"<th colspan='4' class='grp'>{esc(model_name)} (700 users)</th><th colspan='4' class='grp'>GPT-4o (paper, all users)</th></tr>"
            "<tr><th>Category</th><th>Popularity</th><th>Social, k = 50</th>"
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
        "<div class='panels'><div class='panel'><h4>Personalized, k = 25 (700 users)</h4><div class='scroll'><table class='data compact'>"
        "<thead><tr><th>Measure</th><th>Original order</th><th>Repeated</th><th>Shuffled</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div></div>"
        "<div class='panel'><h4>Non-personalized baseline (14 categories × 10 calls)</h4><div class='scroll'><table class='data compact'>"
        "<thead><tr><th>Measure</th><th>Original order</th><th>Shuffled</th></tr></thead>"
        f"<tbody>{bbody}</tbody></table></div></div></div>")


def band_llm_html():
    cols = ["random", "popularity", "socialvec", "socialvec_k12", "socialvec_k25"] + LLM_COLS
    group_start = {"socialvec", "llm_baseline"}
    rows = []
    for cat, r in band_llm.iterrows():
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
    return ("<details><summary>Data table</summary><div class='scroll'><table class='data compact'><thead><tr><th>k</th>"
            "<th>SE1 · paper setup</th><th>Paper</th><th>SE1 · target removed</th><th>SE2</th><th>Paper</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>")


def entities_html():
    rows = []
    for cat, g in band_entities.groupby("category", sort=False):
        names = ", ".join(esc(n) + ("<sup>*</sup>" if not v else "") for n, v in zip(g["twitter_screen_name"], g["in_socialvec"]))
        rng = f"{g['followers_count'].min()/1e3:,.0f}K – {g['followers_count'].max()/1e3:,.0f}K"
        rows.append(f"<tr><th>{esc(CATEGORY_LABELS[cat])}</th><td class='nowrap'>{rng}</td><td class='names'>{names}</td></tr>")
    return ("<details><summary>Candidate entities, ranks 81–100 (* no SocialVec embedding, excluded)</summary><div class='scroll'><table class='data compact'>"
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
llm_t3 = pd.read_csv(LLM / "table3_main.csv", index_col=0)
llm_stats = pd.read_csv(LLM / "table3_stats.csv", index_col=0)
order_st = pd.read_csv(LLM / "order_stats.csv", index_col=0)
base_st = pd.read_csv(LLM / "baseline_trial_stats.csv", index_col=0)
band_llm = pd.read_csv(LLM / "band_by_category.csv", index_col=0)
band_tests = pd.read_csv(LLM / "band_tests.csv", index_col=0).iloc[:, 0]
_band_users = pd.read_csv(LLM / "band_per_user.csv.gz", usecols=["n_k12_accounts", "n_k25_accounts"])
band_llm_n = {k: _band_users[f"n_k{k}_accounts"].mean() for k in (12, 25)}
band_llm_wins = {k: int((band_llm.drop("Overall")[f"llm_k{k}"] > band_llm.drop("Overall")[f"socialvec_k{k}"]).sum()) for k in (12, 25)}
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

CSS = """
:root{color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
 --good:#006300;--bad:#b42f2f;--accent-wash:#eef4fc;--warn-wash:#fdf3ea;--chip:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--card:#151514;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
 --good:#0ca30c;--bad:#e66767;--accent-wash:#14202f;--warn-wash:#2a1d14;--chip:#262624}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--card:#151514;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
 --good:#0ca30c;--bad:#e66767;--accent-wash:#14202f;--warn-wash:#2a1d14;--chip:#262624}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
header.hero{padding:56px 0 28px}
.eyebrow{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
h1{font-size:clamp(28px,4.4vw,42px);line-height:1.15;margin:0 0 14px;letter-spacing:-.01em;font-weight:650}
.lede{font-size:18px;color:var(--ink-2);max-width:760px;margin:0}
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:26px 0 0}
nav.toc a{font-size:14px;color:var(--ink-2);text-decoration:none;padding:5px 12px;border-radius:999px;background:var(--chip)}
nav.toc a:hover{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:34px 0 8px}
.tile{background:var(--card);border-radius:14px;padding:18px 18px 16px;box-shadow:0 0 0 1px var(--ring)}
.tile .label{font-size:13px;color:var(--ink-2)}
.tile .value{font-size:32px;font-weight:620;margin:4px 0 2px;letter-spacing:-.01em}
.tile .sub{font-size:13px;color:var(--muted)}
section{padding:44px 0 8px}
h2{font-size:26px;margin:0 0 6px;letter-spacing:-.005em;font-weight:640}
h2 .num{color:var(--muted);font-weight:500;margin-right:8px}
h3{font-size:18px;margin:34px 0 8px;font-weight:620}
p{max-width:780px}
.intro{color:var(--ink-2);margin-top:0}
.card{background:var(--card);border-radius:16px;padding:20px 20px 14px;box-shadow:0 0 0 1px var(--ring);margin:16px 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);margin:0 0 8px}
.key{display:inline-flex;align-items:center;gap:7px}
.key i{display:inline-block;width:18px;height:3px;border-radius:2px}
.key i.dot{width:10px;height:10px;border-radius:50%}
.key i.dia{width:9px;height:9px;transform:rotate(45deg);border:2px solid var(--ink);border-radius:1px;background:transparent}
.key i.tick{width:2px;height:14px;background:var(--ink-2)}
.key i.sq{width:12px;height:12px;border-radius:3px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .sep{stroke:var(--axis);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.chart .axis-title,.chart .col-head{fill:var(--muted);font-size:12px}
.chart .cat{fill:var(--ink);font-size:13px}
.chart .delta{fill:var(--ink-2);font-size:13px;font-variant-numeric:tabular-nums}
.chart .delta.neg{fill:var(--bad)}
.chart .ref{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 4}
.chart .ref-label{fill:var(--muted);font-size:12px}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .line.s1{stroke:var(--s1)}.chart .line.s2{stroke:var(--s2)}.chart .line.s3{stroke:var(--s3)}
.chart .dot{stroke:var(--card);stroke-width:2}
.chart .dot.s1{fill:var(--s1)}.chart .dot.s2{fill:var(--s2)}.chart .dot.s3{fill:var(--s3)}
.chart .paper-mark{fill:none;stroke:var(--ink);stroke-width:1.6}
.chart .bar.s1{fill:var(--s1)}.chart .bar.s2{fill:var(--s2)}
.chart .rand{stroke:var(--ink-2);stroke-width:2;stroke-linecap:round}
.chart .hit{fill:transparent;cursor:default}
.chart .hit:hover{fill:rgba(127,127,127,.10)}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel h4{margin:0 0 4px;font-size:15px;font-weight:620}
.panel .note{font-size:13px;color:var(--muted);margin:0 0 6px}
@media (max-width:820px){.panels{grid-template-columns:1fr}}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:12px 0}
table.data{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
table.data th,table.data td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--grid)}
table.data thead th{font-size:12px;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis);vertical-align:bottom}
table.data tr.super th{border-bottom:none;text-align:center;color:var(--muted);font-weight:500;padding-bottom:0}
table.data tbody th,table.data thead tr:last-child th:first-child{text-align:left}
table.data tbody th{font-weight:500}
table.data tr.total th,table.data tr.total td{font-weight:650;border-top:1px solid var(--axis);border-bottom:none}
table.data .grp{border-left:1px solid var(--axis)}
table.data td.neg{color:var(--bad)}
table.data td.below{text-decoration:underline dotted;text-underline-offset:3px;color:var(--bad)}
table.data td.p{color:var(--ink-2)}
table.data td.muted{color:var(--muted)}
table.data .paper{display:block;font-size:11px;color:var(--muted);line-height:1.2}
table.data.compact td,table.data.compact th{padding:6px 10px}
table.data td.names{white-space:normal;text-align:left;min-width:420px;color:var(--ink-2)}
table.data td.nowrap{white-space:nowrap}
.caption{font-size:13px;color:var(--muted);margin:6px 0 0;max-width:none}
.heats{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
figure.heat{margin:0}
figure.heat figcaption{font-size:14px;font-weight:600;margin:0 0 8px}
figure.heat figcaption small{display:block;font-weight:400;color:var(--muted)}
table.heatmap{border-collapse:separate;border-spacing:2px;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%}
table.heatmap td{padding:7px 0;text-align:center;border-radius:4px}
table.heatmap th{font-size:12px;font-weight:500;color:var(--muted);padding:2px 4px}
.callouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:14px}
.callout{background:var(--card);border-radius:14px;padding:18px;box-shadow:0 0 0 1px var(--ring)}
.callout h4{margin:0 0 6px;font-size:15px}
.callout p{margin:0;font-size:14.5px;color:var(--ink-2)}
.callout .tag{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:999px;background:var(--chip);color:var(--ink-2);margin-bottom:8px}
.highlights{list-style:none;padding:0;margin:16px 0;display:grid;gap:10px;max-width:820px}
.highlights li{padding:12px 16px 12px 16px;background:var(--accent-wash);border-radius:12px;font-size:15px}
.highlights li.warn{background:var(--warn-wash)}
details{margin:10px 0}
summary{cursor:pointer;color:var(--ink-2);font-size:14px}
img.figure{width:100%;height:auto;border-radius:12px;background:#fff;display:block;margin-top:10px}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--chip);padding:1px 6px;border-radius:6px}
pre{background:var(--card);box-shadow:0 0 0 1px var(--ring);border-radius:12px;padding:14px 16px;overflow-x:auto;font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
.status{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600}
.status.ok{color:var(--good)}.status.no{color:var(--bad)}.status.part{color:var(--ink-2)}
footer{padding:48px 0 64px;color:var(--muted);font-size:13px}
#tip{position:fixed;pointer-events:none;z-index:10;background:var(--card);color:var(--ink);box-shadow:0 0 0 1px var(--ring),0 6px 24px rgba(0,0,0,.12);
 border-radius:10px;padding:8px 11px;font-size:13px;line-height:1.45;opacity:0;transition:opacity .12s;max-width:260px}
"""

JS = """
const tip=document.getElementById('tip');
function show(el,e){tip.innerHTML=el.dataset.tip;tip.style.opacity=1;
  const x=Math.min(e.clientX+14,window.innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px';}
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mouseenter',e=>show(el,e));
  el.addEventListener('mousemove',e=>show(el,e));
  el.addEventListener('mouseleave',()=>{tip.style.opacity=0;});
});
"""

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
  <a href="#tail">3 · Less popular entities</a><a href="#order">4 · LLM candidate order</a><a href="#open">5 · Open items</a></nav>
  <div class="tiles">
    <div class="tile"><div class="label">Table 2 values matching the paper</div><div class="value">42 / 42</div><div class="sub">largest difference {max_t2:.4f}</div></div>
    <div class="tile"><div class="label">Overall MAP, social vs. popularity</div><div class="value">{ov['se1_all_legacy']:.3f}</div><div class="sub">vs. {ov['popularity']:.3f} · {pct(ov['se1_all_legacy_gain'])}, as in the paper</div></div>
    <div class="tile"><div class="label">Gain on entities ranked 81–100</div><div class="value">{pct(A_b['gain'])}</div><div class="sub">vs. {pct(A_h['gain'])} on the top 20, same users</div></div>
    <div class="tile"><div class="label">Popularity above chance, ranks 81–100</div><div class="value">+{A_b['popularity']-A_b['random']:.3f}</div><div class="sub">MAP; social similarity is +{A_b['socialvec']-A_b['random']:.3f}</div></div>
    <div class="tile"><div class="label">LLM gain from 12 favored entities</div><div class="value">{pct(llm_stats.loc['llm_k12','gain vs LLM baseline'])}</div><div class="sub">{model_name}; GPT-4o in the paper: +13%</div></div>
  </div>
</header>

<section id="reproduction">
  <h2><span class="num">1</span>Reproduction</h2>
  <p class="intro">{len(per_user):,} users who follow more than 10 of the {n_entities} candidate entities,
  14 categories, link-prediction evaluation with MAP. Users follow {per_user['n_followed'].mean():,.0f} accounts on average,
  of which {per_user['n_followed_entities'].mean():,.0f} are SocialVec entities.</p>

  <h3>Table 2 · Entity ranking (MAP)</h3>
  <p>Every per-category value matches the paper to its three published decimals. The last two columns show the
  protocol as the paper describes it, with the target category's candidates actually removed from the user profile
  (see section 2).</p>
  <div class="card">{table2_html()}
  <p class="caption">Reproduced values; the paper's value is shown beneath each one. Δ is the gain over popularity.
  Sampled is the closed-world setting, where profiles use only the experiment's entities from the other 13 categories.</p></div>

  <h3>Figure 1 · How much data is needed?</h3>
  <p>MAP rises quickly with the number of entities in the user representation and levels off beyond about 100 sampled
  accounts. With k = 50 sampled accounts, SE1 reaches {fig1.loc['Overall','se1_legacy_k50']/ov['se1_all_legacy']:.1%} of its full-profile
  score (paper: 98.4%). In the closed world, 10 entities already give {fig1.loc['Overall','se2_k10']/ov['se2_all']:.1%} of the full score.</p>
  <div class="card">
    <div class="legend"><span class="key"><i style="background:var(--s1)"></i>SE1 · paper setup</span>
    <span class="key"><i style="background:var(--s2)"></i>SE1 · target removed</span>
    <span class="key"><i style="background:var(--s3)"></i>SE2 · closed world</span>
    <span class="key"><i class="dia"></i>Value reported in the paper</span></div>
    {line_chart()}
    {table1d_html()}
    <details><summary>Per-category panels</summary><img class="figure" alt="Figure 1 per category" src="data:image/png;base64,{fig1_png}"></details>
  </div>

  <h3>Table 3 · LLM personalization</h3>
  <p>I re-ran the LLM experiment with {model_name} (reasoning effort <i>{llm_effort}</i>) and the authors' prompts, on a stratified
  random sample of 50 users per category. The paper's finding replicates: listing 12 favored entities lifts MAP from
  {lo['llm_baseline']:.3f} to {lo['llm_k12']:.3f} ({pct(llm_stats.loc['llm_k12','gain vs LLM baseline'])}), and 50 accounts lift it to
  {lo['llm_k50']:.3f} ({pct(llm_stats.loc['llm_k50','gain vs LLM baseline'])}). The paper reported +13% and +23% for GPT-4o. The social-embedding column (SE1, k = 50) matches
  the paper on all users: {fig1.loc[PAPER_CATEGORY_ORDER, 'se1_legacy_k50'].mean():.3f} vs. 0.562.</p>
  <div class="card">
    <div class="legend"><span class="key"><i class="sq" style="background:var(--s1)"></i>{model_name}, 700 sampled users</span>
    <span class="key"><i class="sq" style="background:var(--s2)"></i>GPT-4o, as reported in the paper</span></div>
    {llm_chart()}
    <details><summary>Per-category table</summary>{table3_html()}</details>
    <p class="caption">Overall MAP (mean over 14 categories). The dashed reference lines are computed on the same 700 users.
    {legacy_fail:.0%} of the personalized responses had an end-of-output marker appended after the id list, which the original
    parser scores as a failure. I read the id list itself. Details are in <code>results/LLM_REPORT.md</code>.</p>
  </div>

  <h3>Table 4 · Categories × entities per category</h3>
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
    <div class="callout"><span class="tag">Labels</span><h4>Two classifier outputs are flipped</h4>
    <p>The <code>socialvec</code> package labels the age and education outputs backwards. Read as shipped, 98% of
    <i>60 Minutes</i> followers would be under 25. The paper's figure, and this reproduction, use the corrected reading.</p></div>
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
  at positions <b>81–100</b> instead of the top 20. No category lists more, so ranks 100–120 were not possible.
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
    <p class="caption">MAP per category; Δ is the gain of social similarity over popularity. The values are in Table A below.</p>
  </div>

  <h3>Table A · Users sampled for each category</h3>
  <div class="card">{matched_html(mA)}
  <p class="caption">{A_b['users']:,.0f} users who follow at least one entity ranked 81–100 in their category. Rel./user is the number of
  candidates a user follows. Dotted red marks popularity scores below the random ranking. p is from a Wilcoxon signed-rank test,
  social vs. popularity.</p></div>

  <h3>Table B · Robustness: all experiment users</h3>
  <p>Each of the 11,153 users is evaluated in every category in which they follow at least one entity from each band. That gives
  {B_b['users']:,.0f} user–category pairs from {n_distinct_users:,} distinct users. Only {A_b['users']/B_b['users']:.0%} of these pairs come from users sampled for the category,
  so this is a largely independent check. The overall test pools all pairs.</p>
  <div class="card">{matched_html(mB)}{entities_html()}</div>

  <h3>Table C · LLM ranking of ranks 81–100</h3>
  <p>The Table 3 protocol applied to the less popular candidates, for 30 random users per category. Every listed entity of the
  target category is removed from the prompts. Without user information, the LLM is no better than popularity
  ({band_llm.loc['Overall','llm_baseline']:.3f} vs. {band_llm.loc['Overall','popularity']:.3f}). With user information it personalizes well:
  {band_llm.loc['Overall','llm_k12']:.3f} with 12 entities and {band_llm.loc['Overall','llm_k50']:.3f} with 50 accounts. Social similarity from all of
  the user's entities scores {band_llm.loc['Overall','socialvec']:.3f}. The two are complementary: the LLM is stronger for news outlets and
  TV shows, social similarity for musical artists, sports teams and actors.</p>
  <p>For a like-for-like comparison, the <i>SocialVec k = 12</i> and <i>k = 25</i> columns build the user embedding from exactly the
  accounts listed in the LLM's k = 12 and k = 25 prompts ({band_llm_n[12]:.1f} and {band_llm_n[25]:.1f} on average, as some users have fewer).
  Given the same evidence, the LLM ranks better at both sizes:
  {band_llm.loc['Overall','llm_k12']:.3f} vs. {band_llm.loc['Overall','socialvec_k12']:.3f} with 12 accounts
  (paired Wilcoxon p {p_text(band_tests['llm_k12 vs socialvec_k12'])}; ahead in {band_llm_wins[12]} of 9 categories), and
  {band_llm.loc['Overall','llm_k25']:.3f} vs. {band_llm.loc['Overall','socialvec_k25']:.3f} with 25
  (p {p_text(band_tests['llm_k25 vs socialvec_k25'])}; ahead in {band_llm_wins[25]} of 9).</p>
  <div class="card">{band_llm_html()}
  <p class="caption">Bold marks the best method in each row. With 30 users per category, per-category MAP has a standard error of about 0.05.</p></div>

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
  shuffled.</p>
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
  <p class="caption">{llm_calls:,} model calls in total across all LLM experiments, costing ${llm_cost:.2f}.</p>
</section>

<section id="open">
  <h2><span class="num">5</span>Open items</h2>
  <div class="callouts">
    <div class="callout"><span class="tag">Scale</span><h4>LLM experiments on all users</h4>
    <p>The LLM results use samples of 50 users per category (30 for ranks 81–100), which keeps overall MAP within about ±0.01.
    A full-population run takes about 45K calls. The same command runs it with larger samples, and cached responses are reused.</p></div>
    <div class="callout"><span class="tag">Next</span><h4>Description-based tail candidates</h4>
    <p>The requested variant selects 50–100 accounts per category by their <i>textual descriptions</i> and requires them to be
    below the top 10K/20K SocialVec entities. My band used the existing typed lists and follower counts: 20 candidates per
    category, of which only {n_below_10k} of {n_band_in_sv} fall below the top 10K. The ranking protocol carries over unchanged.</p></div>
  </div>
</section>

<footer>
  <p>Regenerate the underlying results with <code>python run_reproduction.py</code>,
  <code>python run_popularity_band.py --start 81 --end 100</code> and <code>python run_llm_experiments.py</code>, then
  rebuild this page with <code>python make_results_page.py</code>. Details are in <code>results/REPORT.md</code>,
  <code>results/POPULARITY_BAND_REPORT.md</code> and <code>results/LLM_REPORT.md</code>.</p>
</footer>
</div>
<script>{JS}</script>
</body></html>
"""
(R / "results.html").write_text(page, encoding="utf-8")
print("wrote", R / "results.html", f"{len(page)/1e6:.2f} MB")
