"""
Export results/results_v2.html as a LaTeX project for Overleaf: overleaf/main.tex with every
section, table, figure and prompt of the page, without the paper's published results (values,
columns, figures and sentences that report or compare against them), tables/ (one tabular per table) and figures/
(PDF redraws of the page's SVG charts with matplotlib, and the page's PNG figures).

    python make_overleaf.py        # results_v2.html -> overleaf/
    python make_overleaf.py v3     # results_v3.html -> overleaf_v3/

Zip the overleaf/ folder and upload it to Overleaf (New project > Upload project); it compiles
with the default pdfLaTeX. The script imports make_results_page_v2, which rebuilds results_v2.html
first, so the two always agree.
"""

import base64
import importlib
import sys
import re
import shutil
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bs4 import BeautifulSoup, NavigableString, Tag  # noqa: E402

from reproduction.config import CATEGORY_LABELS, ROOT  # noqa: E402

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
v2 = importlib.import_module(f"make_results_page_{VERSION}")  # builds results_<VERSION>.html; its data feeds the figures
PAGE = ROOT / "results" / f"results_{VERSION}.html"

OUT = ROOT / ("overleaf" if VERSION == "v2" else f"overleaf_{VERSION}")
FIG, TAB = OUT / "figures", OUT / "tables"
COLORS = {"s1": "#2f6fd6", "s2": "#e8693a", "s3": "#16a37b", "muted": "#bdbab0", "ink": "#222222", "grid": "#e4e2dc"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": COLORS["grid"], "axes.axisbelow": True, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- LaTeX text
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}"}
SYMBOLS = {"–": "--", "—": "---", "·": r"\textperiodcentered{}", "─": "-", "Δ": r"$\Delta$", "τ": r"$\tau$", "−": r"$-$",
           "×": r"$\times$", "±": r"$\pm$", "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$",
           "“": "``", "”": "''", "‘": "`", "’": "'", "…": r"\dots{}", " ": "~"}
SYMBOLS.update({"†": r"\textdagger{}", "­": ""})  # dagger (Figure 4 markers); soft hyphen


def tex(text):
    """Escape plain text for pdfLaTeX; Latin letters with accents pass through, other symbols are mapped or dropped."""
    out = []
    for ch in text:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in SYMBOLS:
            out.append(SYMBOLS[ch])
        elif ord(ch) < 128 or (unicodedata.category(ch).startswith("L") and ord(ch) <= 0x24F):
            out.append(ch)
    return "".join(out)


def ascii_text(text):
    """Text for verbatim blocks: the mapped symbols in plain ASCII, other non-ASCII characters dropped."""
    plain = {"–": "-", "—": "-", "·": ".", "─": "-", "−": "-", "×": "x", "→": "->", "“": '"', "”": '"', "‘": "'", "’": "'"}
    return "".join(plain.get(ch, ch) for ch in text if ord(ch) < 128 or ch in plain)


def classes(el):
    return el.get("class") or []


def inline(el):
    """LaTeX for the inline content of an element."""
    if isinstance(el, NavigableString):
        return tex(re.sub(r"\s+", " ", str(el)))
    if not isinstance(el, Tag) or el.name in ("script", "style", "svg"):
        return ""
    inner = "".join(inline(c) for c in el.children)
    if el.name in ("b", "strong"):
        return rf"\textbf{{{inner}}}"
    if el.name in ("i", "em"):
        return rf"\emph{{{inner}}}"
    if el.name == "code":
        return rf"\texttt{{{inner}}}"
    if el.name == "sup":
        return rf"\textsuperscript{{{inner}}}"
    if el.name == "br":
        return r"\newline "
    if el.name == "a" and str(el.get("href", "")).startswith("http"):
        return rf"\href{{{el['href']}}}{{{inner}}}"
    if "paper" in classes(el):  # the paper's value beneath ours
        return ""
    if "muted" in classes(el) or el.name == "small":
        return rf"{{\textcolor{{gray}}{{{inner}}}}}"
    if el.name in ("div", "p", "li"):
        return " " + inner + " "
    return inner


PAPER = re.compile(r"\bpaper\b(?! setup)|\bpublished\b|\bGPT-4o in the paper\b", re.I)
RESULT_WORDS = re.compile(r"\d|match|replicat|report|beneath|as in the|differ|decimal|reproduc|as published|conclusion", re.I)


def without_paper_results(text):
    """Drop the sentences that report or compare against the paper's results."""
    # a lead-in that only frames our own result as a replication: keep the result
    text = re.sub(r"The paper's finding replicates:\s*(\w)", lambda m: m.group(1).upper(), text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\\(])", text)
    kept = [x for x in sentences if not (PAPER.search(x) and RESULT_WORDS.search(x))]
    return " ".join(kept).strip()


def drop_paper_columns(table):
    """Remove the columns whose header refers to the paper (e.g. the paper's GPT-4o values)."""
    head = table.find("thead")
    if not head:
        return
    dropped = set()
    for tr in head.find_all("tr"):
        j = 0
        for c in tr.find_all(["th", "td"], recursive=False):
            span = int(c.get("colspan", 1))
            if PAPER.search(c.get_text(" ", strip=True)):
                dropped.update(range(j, j + span))
            j += span
    if not dropped:
        return
    for tr in table.find_all("tr"):
        j = 0
        for c in tr.find_all(["th", "td"], recursive=False):
            span = int(c.get("colspan", 1))
            if j in dropped:
                c.decompose()
            j += span


def heading_text(el):
    """A heading's text without the section number badge."""
    text = "".join(inline(c) for c in el.children if not (isinstance(c, Tag) and "num" in classes(c))).strip()
    return re.sub(r",\s*paper Table \d+", "", text)


# ---------------------------------------------------------------- tables
def table_tex(table):
    head = table.find("thead")
    head_rows = head.find_all("tr") if head else []
    body = table.find("tbody")
    body_rows = body.find_all("tr", recursive=False) if body else [r for r in table.find_all("tr") if r not in head_rows]
    cells = lambda tr: tr.find_all(["th", "td"], recursive=False)
    ncols = max(sum(int(c.get("colspan", 1)) for c in cells(tr)) for tr in head_rows + body_rows)
    long_cols = set()
    for tr in body_rows:
        j = 0
        for c in cells(tr):
            if "names" in classes(c) or len(c.get_text(" ", strip=True)) > 60:
                long_cols.add(j)
            j += int(c.get("colspan", 1))
    width = 0.75 / max(1, len(long_cols))
    spec = "".join(rf"p{{{width:.2f}\linewidth}}" if j in long_cols else ("l" if j == 0 else "r") for j in range(ncols))

    pending = {}  # column -> rows still covered by a rowspan cell above (left empty here)

    def row(tr, header=False):
        parts, j = [], 0
        for c in cells(tr):
            while pending.get(j):
                pending[j] -= 1
                parts.append("")
                j += 1
            if int(c.get("rowspan", 1)) > 1:
                pending[j] = int(c["rowspan"]) - 1
            j += int(c.get("colspan", 1))
            content = inline(c).strip()
            style = c.get("style", "")
            if m := re.search(r"background:\s*#([0-9a-fA-F]{6})", style):
                fg = re.search(r"color:\s*#([0-9a-fA-F]{6})", style)
                content = rf"\cellcolor[HTML]{{{m.group(1)}}}" + (rf"\textcolor[HTML]{{{fg.group(1)}}}{{{content}}}" if fg else content)
            span = int(c.get("colspan", 1))
            if span > 1 or header:
                content = rf"\multicolumn{{{span}}}{{c}}{{{content}}}" if span > 1 else content
            parts.append(content)
        while pending.get(j):
            pending[j] -= 1
            parts.append("")
            j += 1
        return " & ".join(parts) + r" \\"

    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    lines += [row(tr, header=True) for tr in head_rows]
    if head_rows:
        lines.append(r"\midrule")
    for tr in body_rows:
        if "total" in classes(tr):
            lines.append(r"\midrule")
        lines.append(row(tr))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures (matplotlib redraws of the page's SVG charts)
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    return f"figures/{name}.pdf"


def fig_line():
    if hasattr(v2, "f1b"):  # v3: both bands on the Table 6 pairs
        xs = v2.F1_KS[:-1] + ["All"]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for c, label, cls in v2.F1_BANDS:
            o = v2.f1b[c].loc["Overall"]
            ax.plot(xs, [o[f"k{k}"] for k in v2.F1_KS[:-1]] + [o["all"]], marker="o", ms=4, color=COLORS[cls], label=f"SocialVec, {label.lower()}")
            ax.axhline(o["popularity"], color=COLORS[cls], linestyle=":", linewidth=1.4, label=f"Popularity, {label.lower()} {o['popularity']:.3f}")
        ax.set_xlabel("Accounts sampled per user")
        ax.set_ylabel("MAP")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
        return save(fig, "figure1_overall")
    o = v2.fig1.loc["Overall"]
    xs = ["10", "20", "30", "40", "50", "80", "100", "300", "500", "All"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for cls, label, prefix, full in [("s1", "SE1 · paper setup", "se1_legacy_k", "se1_all_legacy")]:
        ax.plot(xs, [o[f"{prefix}{k}"] for k in xs[:-1]] + [o[full]], marker="o", ms=4, color=COLORS[cls], label=label)
    ax.axhline(o["popularity"], color="gray", linestyle="--", linewidth=1, label=f"Popularity {o['popularity']:.3f}")
    ax.set_xlabel("Accounts sampled per user")
    ax.set_ylabel("MAP")
    ax.legend(frameon=False, fontsize=8)
    return save(fig, "figure1_overall")


def fig_llm():
    o = v2.llm_t3.loc["Overall"]
    ours = [o[c] for c in v2.LLM_COLS]
    x = np.arange(len(v2.LLM_COLS))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(x, ours, 0.5, color=COLORS["s1"], label=f"{v2.model_name}, 700 sampled users")
    for name, v in [("Popularity", o["popularity"]), ("Social, k = 50", o["se1_legacy_k50"])]:
        ax.axhline(v, color="gray", linestyle="--", linewidth=1)
        ax.text(len(x) - 0.45, v + 0.003, f"{name} {v:.3f}", fontsize=7, color="gray")
    ax.set_xticks(x, v2.LLM_LABELS)
    ax.set_ylim(0.40, 0.65)
    ax.set_ylabel("MAP")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return save(fig, "llm_overall")


def fig_bars(table, name):
    rows = [c for c in table.index if c != "Overall"] + ["Overall"]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(3.6, 0.32 * len(rows) + 0.6))
    ax.barh(y + 0.18, table.loc[rows, "popularity"], 0.34, color=COLORS["s2"], label="Popularity")
    ax.barh(y - 0.18, table.loc[rows, "socialvec"], 0.34, color=COLORS["s1"], label="Social similarity")
    ax.scatter(table.loc[rows, "random"], y, marker="|", s=120, color=COLORS["ink"], label="Random (expected)", zorder=3)
    ax.set_yticks(y, [CATEGORY_LABELS.get(c, "Overall") for c in rows])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("MAP")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return save(fig, name)


def fig_strip():
    cats = v2.main_pop_cats
    colors = {band: COLORS[cls] for band, cls in v2.STRIP_BANDS}
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(cats) + 0.7))
    for i, cat in enumerate(cats):
        for j, (band, _) in enumerate(v2.STRIP_BANDS):
            g = v2.pop_ranked[(v2.pop_ranked["category"] == cat) & (v2.pop_ranked["band"] == band)]
            y = len(cats) - 1 - i + (0.12 if j == 0 else -0.12)
            ax.scatter(g["global_rank"], [y] * len(g), s=10, alpha=0.55, color=colors[band], label=band if i == 0 else None)
            if len(g):
                ax.plot([g["global_rank"].median()] * 2, [y - 0.1, y + 0.1], color=colors[band], linewidth=2.5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(cats)), [CATEGORY_LABELS[c] for c in cats][::-1])
    ax.set_xlabel("Global SocialVec popularity rank (log scale; left is more popular)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return save(fig, "figure3_global_popularity")


def fig_methods(band, summary=None, model=""):
    o = (v2.dsh if summary is None else summary)[band]["map"]
    methods = v2.DEMO_METHODS
    y = np.arange(len(methods))[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(methods) + 0.5))
    vals = [o[c] for c, _ in methods]
    ax.barh(y, vals, 0.6, color=[COLORS[v2.DEMO_COLORS.get(c, "muted")] for c, _ in methods])
    marks = v2.sig_marks("haiku" if model else "luna", band) if hasattr(v2, "sig_marks") else {}
    for yi, v, (c, _) in zip(y, vals, methods):
        ax.text(v + 0.008, yi, f"{v:.3f}" + (f"  {marks[c]}" if c in marks else ""), va="center", fontsize=8)
    ax.set_yticks(y, [label for _, label in methods])
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("MAP")
    suffix = "_" + re.sub(r"\W+", "_", model.lower()).strip("_") if model else ""
    return save(fig, f"figure4_ranks_{band.replace('-', '_')}{suffix}")


def fig_attrs():
    attrs, total = v2.demo_attrs, v2.demo["users"]
    fills = [COLORS["s1"], COLORS["s2"], COLORS["s3"], "#9a978e", "#55534d", "#d6d3cb"]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(attrs) + 0.4))
    for i, (attr, a) in enumerate(attrs.items()):
        y, x = len(attrs) - 1 - i, 0.0
        for j, value in enumerate(v for v in a["values"] if a["counts"].get(v)):
            share = a["counts"][value] / total
            ax.barh(y, share, left=x, color=fills[j % len(fills)], edgecolor="white", height=0.7)
            if share >= 0.12:
                ax.text(x + share / 2, y, f"{value} {share:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if j < 3 or j == 4 else COLORS["ink"])
            x += share
    ax.set_yticks(range(len(attrs)), list(attrs)[::-1])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(False)
    return save(fig, "figure5_attributes")


CHARTS = {
    "Overall MAP versus number of sampled entities": fig_line,
    "Overall MAP per LLM condition": fig_llm,
    "Ranks 1–20": lambda: fig_bars(v2.mB["head"], "section3_ranks_1_20"),
    "Ranks 81–100": lambda: fig_bars(v2.mB["band"], "section3_ranks_81_100"),
    "Global popularity rank of the candidates by category": fig_strip,
    "Overall MAP by method, ranks 1-20, shuffled candidates": lambda: fig_methods("1-20"),
    "Overall MAP by method, ranks 81-100, shuffled candidates": lambda: fig_methods("81-100"),
    f"Overall MAP by method, ranks 1-20, shuffled candidates, {v2.HAIKU}": lambda: fig_methods("1-20", v2.dhk, v2.HAIKU),
    f"Overall MAP by method, ranks 81-100, shuffled candidates, {v2.HAIKU}": lambda: fig_methods("81-100", v2.dhk, v2.HAIKU),
    "Demographic attributes in GPT-4o descriptions": fig_attrs,
}


# ---------------------------------------------------------------- page -> LaTeX
class Converter:
    def __init__(self):
        self.n_tables = 0
        self.table_names = set()
        self.n_images = 0

    def blocks(self, el):
        out = []
        for c in el.children:
            if isinstance(c, NavigableString):
                if c.strip():
                    out.append(inline(c))
                continue
            if isinstance(c, Tag):
                out += self.block(c)
        return out

    def block(self, el):
        cls = classes(el)
        if el.name in ("script", "style", "nav") or "legend" in cls or el.get("id") == "tip":
            return []
        if el.name == "h1":
            eyebrow = el.find_previous_sibling("p", class_="eyebrow")
            subtitle = rf"\\[4pt]\large {inline(eyebrow).strip()}" if eyebrow else ""
            return [rf"\title{{{inline(el).strip()}{subtitle}}}", r"\date{}", r"\maketitle"]
        if el.name == "h2":
            num = el.find(class_="num")
            if num and not num.get_text(strip=True).isdigit():  # e.g. the supplement, "S" on the page
                return [rf"\section*{{{num.get_text(strip=True)}\quad {heading_text(el)}}}"]
            return [rf"\section{{{heading_text(el)}}}"]
        if el.name == "h3":
            return [rf"\subsection*{{{heading_text(el)}}}"]
        if el.name in ("h4", "summary", "figcaption"):
            for small in el.find_all("small"):  # e.g. the heatmaps' "mean |difference|" from the paper's values
                if re.search(r"difference", small.get_text(), re.I):
                    small.decompose()
            return [rf"\subsubsection*{{{heading_text(el)}}}"]
        if el.name == "figure" and el.find("figcaption") and PAPER.search(el.find("figcaption").get_text()):
            return []  # the paper's heatmap
        if el.name == "p":
            text = without_paper_results(inline(el).strip())
            if not text:
                return []
            if "caption" in cls or "note" in cls:
                return [rf"{{\small {text}}}", ""]
            if "eyebrow" in cls:  # part of the title (see h1)
                return []
            return [text, ""]
        if el.name in ("ul", "ol"):
            items = [rf"  \item {t}" for li in el.find_all("li", recursive=False)
                     if (t := without_paper_results(inline(li).strip()))]
            return [r"\begin{itemize}", *items, r"\end{itemize}"] if items else []
        if "tiles" in cls:
            items = [rf"  \item {inline(t.select_one('.label')).strip()}: \textbf{{{inline(t.select_one('.value')).strip()}}} "
                     rf"({inline(t.select_one('.sub')).strip()})" for t in el.select(".tile") if not PAPER.search(t.get_text(" "))]
            return [r"\begin{itemize}", *items, r"\end{itemize}"]
        if "tag" in cls:
            return [rf"\noindent\textsc{{{inline(el).strip().lower()}}}\par"]
        if el.name == "table":
            drop_paper_columns(el)
            self.n_tables += 1
            # name the file after the page's table number (the nearest "Table N ·" heading above it), else table_extra_N
            name = None
            for h in el.find_all_previous(["h3", "h4", "summary"], limit=3):
                m = re.match(r"\s*Table (\d+) ·", h.get_text())
                if m and f"table_{int(m.group(1)):02d}" not in self.table_names:
                    name = f"table_{int(m.group(1)):02d}"
                    break
            if name is None:
                name = f"table_extra_{self.n_tables:02d}"
            self.table_names.add(name)
            (TAB / f"{name}.tex").write_text(table_tex(el), encoding="utf-8")
            return [r"\begin{center}\begin{adjustbox}{max width=\linewidth}", rf"\input{{tables/{name}}}",
                    r"\end{adjustbox}\end{center}", ""]
        if el.name == "svg":
            path = CHARTS[el.get("aria-label")]()
            return [r"\begin{center}", rf"\includegraphics[width=\linewidth,height=0.5\textheight,keepaspectratio]{{{path}}}",
                    r"\end{center}", ""]
        if el.name == "img":
            self.n_images += 1
            src = el["src"]
            ext = "png" if "image/png" in src else "jpg"
            name = f"image_{self.n_images:02d}.{ext}"
            (FIG / name).write_bytes(base64.b64decode(src.split(",", 1)[1]))
            return [r"\begin{center}", rf"\includegraphics[width=\linewidth]{{figures/{name}}}", r"\end{center}", ""]
        if el.name == "pre":
            return [r"\begin{Verbatim}[breaklines,breakanywhere,fontsize=\footnotesize]", ascii_text(el.get_text()).strip("\n"),
                    r"\end{Verbatim}", ""]
        if el.name == "footer":
            return [r"\section*{About these results}", *self.blocks(el)]
        return self.blocks(el)  # section, div, details, figure, header ...


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{array}
\usepackage[table]{xcolor}
\usepackage{fvextra}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\setlist{itemsep=2pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.15}
% Generated by make_overleaf.py from the results page; edit the generator, not this file.
"""


def main():
    page = BeautifulSoup(PAGE.read_text(encoding="utf-8"), "html.parser")
    shutil.rmtree(OUT, ignore_errors=True)
    FIG.mkdir(parents=True)
    TAB.mkdir(parents=True)
    conv = Converter()
    body = conv.blocks(page.select_one("div.wrap"))
    text = "\n".join([PREAMBLE, r"\begin{document}", *body, r"\end{document}", ""])
    text = re.sub(r"\n{3,}", "\n\n", text)
    (OUT / "main.tex").write_text(text, encoding="utf-8")
    (OUT / "README.md").write_text(
        f"LaTeX export of results/{PAGE.name}, written by `python make_overleaf.py{'' if VERSION == 'v2' else ' ' + VERSION}`.\n\n"
        "Zip this folder and upload it to Overleaf (New project > Upload project). main.tex compiles with the\n"
        "default pdfLaTeX compiler. Tables are in tables/, figures in figures/ (the charts are matplotlib\n"
        "redraws of the page's interactive charts; the other figures are the page's own images).\n", encoding="utf-8")
    print(f"wrote {OUT}: {conv.n_tables} tables, {len(list(FIG.glob('*.pdf')))} chart PDFs, {conv.n_images} images")


if __name__ == "__main__":
    main()
