"""
Markdown report comparing the reproduced numbers with the paper.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import paper_results as paper
from .config import CATEGORY_LABELS, PAPER_CATEGORY_ORDER


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", index_name: Optional[str] = None) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    header = [index_name if index_name is not None else (df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [cell(v) for v in row]) + " |")
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{x:+.0%}"


def table2_section(t2: pd.DataFrame) -> str:
    rows = []
    for topic in PAPER_CATEGORY_ORDER + ["Overall"]:
        ours = t2.loc[topic]
        ref = paper.TABLE2.loc[topic] if topic != "Overall" else pd.Series(
            {"n_relevant": np.nan, **paper.TABLE2_OVERALL})
        rows.append({
            "Category": CATEGORY_LABELS.get(topic, "**Overall**"),
            "Entities/user (paper)": ref["n_relevant"],
            "Entities/user": ours["n_relevant"],
            "Popularity (paper)": ref["popularity"],
            "Popularity": ours["popularity"],
            "All (paper)": ref["all"],
            "All": ours["se1_all_legacy"],
            "Δ": pct(ours["se1_all_legacy_gain"]),
            "Sampled (paper)": ref["sampled"],
            "Sampled": ours["se2_all"],
            "Δ ": pct(ours["se2_all_gain"]),
            "All, target removed": ours["se1_all"],
            "Δ  ": pct(ours["se1_all_gain"]),
        })
    df = pd.DataFrame(rows).set_index("Category")
    for col in ["Entities/user (paper)", "Entities/user"]:
        df[col] = df[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")

    ours_cols = t2.loc[PAPER_CATEGORY_ORDER, ["popularity", "se1_all_legacy", "se2_all"]].values
    ref_cols = paper.TABLE2.loc[PAPER_CATEGORY_ORDER, ["popularity", "all", "sampled"]].values
    max_diff = np.abs(ours_cols - ref_cols).max()
    return (
        "## Table 2: entity ranking (MAP)\n\n"
        + markdown_table(df)
        + f"\n\nLargest absolute difference from the paper over the 42 per-category MAP cells: "
          f"**{max_diff:.4f}** (the paper rounds to 3 decimals).\n\n"
          "*All, target removed*: the 'All' representation with the target category's candidates actually "
          "excluded from the user profile, as the paper describes (see the note on the legacy code below).\n"
    )


def figure1_section(fig1: pd.DataFrame) -> str:
    overall = fig1.loc["Overall"]
    rows = []
    for k, ref in paper.FIGURE1_SE1.items():
        rows.append({"Setting": f"SE1, k = {k}", "Paper": ref, "Reproduced": overall[f"se1_legacy_k{k}"],
                     "Target removed": overall[f"se1_k{k}"]})
    rows.append({"Setting": "SE1, All", "Paper": paper.TABLE2_OVERALL["all"],
                 "Reproduced": overall["se1_all_legacy"], "Target removed": overall["se1_all"]})
    for k, ref in paper.FIGURE1_SE2.items():
        rows.append({"Setting": f"SE2, k = {k}", "Paper": ref, "Reproduced": overall[f"se2_k{k}"],
                     "Target removed": np.nan})
    rows.append({"Setting": "SE2, All", "Paper": paper.TABLE2_OVERALL["sampled"],
                 "Reproduced": overall["se2_all"], "Target removed": np.nan})
    df = pd.DataFrame(rows).set_index("Setting")

    se1_cols = [c for c in fig1.columns if c.startswith("se1_legacy_k")]
    se2_cols = [c for c in fig1.columns if c.startswith("se2_k")]
    curve = pd.DataFrame({
        "SE1 (paper setup)": overall[se1_cols].values,
    }, index=[c.split("_k")[1] for c in se1_cols])
    curve["SE1 (target removed)"] = [overall[f"se1_k{k}"] for k in curve.index]
    curve["SE2"] = [overall.get(f"se2_k{k}", np.nan) for k in curve.index]
    curve.index.name = "k"
    return (
        "## Figure 1: MAP vs. number of sampled entities\n\n"
        + markdown_table(df)
        + "\n\nOverall MAP for every sample size:\n\n"
        + markdown_table(curve)
        + "\n\nRatios quoted in the paper: k = 10 → "
          f"{overall['se1_legacy_k10'] / overall['se1_all_legacy']:.1%} (paper 93.1%), k = 20 → "
          f"{overall['se1_legacy_k20'] / overall['se1_all_legacy']:.1%} (paper 96.0%), k = 50 → "
          f"{overall['se1_legacy_k50'] / overall['se1_all_legacy']:.1%} (paper 98.4%) of SE1-All; SE2 k = 10 → "
          f"{overall['se2_k10'] / overall['se2_all']:.1%} (paper 99.0%) of SE2-All.\n\n"
          "![Figure 1](figure1.png)\n"
    )


def table3_section(fig1: pd.DataFrame, llm_dir: Optional[Path]) -> str:
    """Table 3: the paper's GPT-4o columns and, if available, the sampled rerun from results/llm/."""
    llm_cols = ["llm_baseline", "llm_k12", "llm_k25", "llm_k50"]
    rerun = None
    if llm_dir is not None and (llm_dir / "table3_main.csv").exists():
        rerun = pd.read_csv(llm_dir / "table3_main.csv", index_col=0)
        runs = json.loads((llm_dir / "run_info.json").read_text()) if (llm_dir / "run_info.json").exists() else []
        model = runs[0]["model"] if runs else "LLM"
        model = {"gpt-6-luna": "GPT-6 Luna"}.get(model, model)
        effort = runs[0]["reasoning_effort"] if runs else "?"

    rows = []
    for topic in PAPER_CATEGORY_ORDER:
        ref = paper.TABLE3.loc[topic]
        row = {"Category": CATEGORY_LABELS[topic],
               "Social k=50 (paper)": ref["social_k50"], "Social k=50": fig1.loc[topic, "se1_legacy_k50"]}
        for col in llm_cols:
            row[f"GPT-4o {col[4:]} (paper)"] = ref[col]
        if rerun is not None:
            for col in llm_cols:
                row[f"{model} {col[4:]}"] = rerun.loc[topic, col]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Category")
    df.loc["**Average**"] = df.mean()
    text = "## Table 3: LLM personalization\n\n" + markdown_table(df) + "\n\n"
    text += "The 'Social k=50' column is the SE1 k = 50 setting of Figure 1, computed on all users. "
    if rerun is None:
        text += ("The LLM columns were not re-run; `python run_llm_experiments.py` produces them "
                 "(see README).\n")
    else:
        n = int(rerun.loc["Overall", "users"])
        text += (f"The {model} columns (reasoning effort '{effort}') were re-run on a stratified random sample "
                 f"of {n} users (50 per category), with the authors' prompts. See `results/LLM_REPORT.md` for "
                 "the analysis, including the candidate-order experiment.\n")
    return text


def table4_section(grid: pd.DataFrame, pooled: pd.DataFrame) -> str:
    def diff(t):
        return np.abs(t.values - paper.TABLE4.values).mean()

    return (
        "## Table 4: k categories × n entities (closed world)\n\n"
        "**Paper:**\n\n" + markdown_table(paper.TABLE4, index_name="k \\ n")
        + "\n\n**Reproduced, n entities from each of k categories** (the protocol as described): "
          f"mean absolute difference from the paper {diff(grid):.4f}\n\n"
        + markdown_table(grid, index_name="k \\ n")
        + "\n\n**Reproduced, n entities in total from the union of k categories**: "
          f"mean absolute difference {diff(pooled):.4f}\n\n"
        + markdown_table(pooled, index_name="k \\ n")
        + "\n\nThe code for Table 4 is not in the legacy notebooks. The first row and last column match "
          "under both readings. The paper's n = 1 column barely changes with k (0.471 → 0.477), which "
          "the pooled reading reproduces and the per-category reading does not. Both readings support the "
          "paper's conclusion: 3-5 categories with 4-5 entities each recover nearly all of SE2-All (0.522).\n"
    )


def figure2_section(radars: Dict[str, pd.DataFrame]) -> str:
    parts = ["## Figure 2: socio-demographic profiles of followers\n"]
    diffs = []
    for title, ours in radars.items():
        ref = pd.DataFrame(paper.FIGURE2[title], index=paper.RADAR_TRAITS).T.loc[ours.index]
        diffs.append(np.abs(ours.values - ref.values))
        table = pd.concat({"reproduced": ours, "notebook": ref}, axis=1)
        table.columns = [f"{t} ({src})" for src, t in table.columns]
        table = table[[f"{t} ({src})" for t in paper.RADAR_TRAITS for src in ("notebook", "reproduced")]]
        parts.append(f"### {title}\n\n" + markdown_table(table) + "\n")
    d = np.concatenate([x.ravel() for x in diffs])
    parts.append(
        f"Mean absolute difference from the values in the notebook (which produced the paper's figure): "
        f"**{d.mean():.3f}** (max {d.max():.3f}).\n\n![Figure 2](figure2.png)\n")
    return "\n".join(parts)


def dataset_section(per_user: pd.DataFrame, categories: Dict[str, pd.DataFrame]) -> str:
    n_entities = sum(len(c) for c in categories.values())
    return (
        "## Dataset\n\n"
        f"- Candidate entities: {n_entities} across {len(categories)} categories "
        "(the paper says 280 = 14 × 20; four candidates are marked unused in the spreadsheet).\n"
        f"- Users after keeping those who follow more than 10 candidate entities: {len(per_user):,} "
        "(of 13,800 sampled; the paper says 'roughly 12K').\n"
        f"- Accounts followed per user: {per_user['n_followed'].mean():,.0f} on average "
        f"(paper: 1,972), of which {per_user['n_followed_entities'].mean():,.0f} are SocialVec entities.\n"
        f"- Closed-world profile size: {per_user['n_closed_world'].mean():.1f} entities on average "
        f"({(per_user['n_closed_world'] + per_user['n_relevant']).mean():.1f} including the target "
        "category; the paper says 40).\n"
    )


NOTES = """## Notes on the legacy code and differences

1. **Target entities were not removed in the 'All' setting.** The notebook builds the 'All' user profile
   as `set(friend ids as strings) - set(user['topic_accounts_str'])`. But `topic_accounts_str` holds
   *screen names*, so the subtraction removes nothing, and the target category's candidates stay in the
   profile. The same applies to SE1 sampling and to the LLM k = 50 prompts. This reproduction reports
   both versions. Removing the target entities properly changes overall MAP from 0.571 to 0.567
   (+22% → +21% over popularity), so the paper's conclusions hold. The closed-world ('Sampled' / SE2)
   setting was implemented correctly.
2. **SE1 sampling order.** The legacy code sampled from a Python `set` of id *strings*, whose order
   depends on per-process hash randomization, so its SE1 samples cannot be redrawn exactly. We sample
   with the same seed (76) from the follow list in its stored order; overall values land within
   ±0.004 of the paper. SE2 samples come from a set of *integers*, which has a deterministic order, and
   match the notebook's output exactly.
3. **SE1 samples accounts, not entities.** Like the legacy code, SE1 draws k *followed accounts*. Only
   ~43% of them are SocialVec entities, so a k = 10 sample holds ~4 entities on average.
4. **Socio-demographic classifiers.** The `socialvec` package labels the age and education classifier
   outputs the opposite way round from the paper's Figure 2. We follow the figure. With the package's
   labels, ~98% of *60 Minutes* followers would be under 25.
5. **Table 4** — see the section above; the code behind it is not in the notebooks.
6. **Table 3** was re-run with a different model on a user sample (`run_llm_experiments.py`);
   GPT-4o itself was not re-run.
"""


def build_report(per_user: pd.DataFrame, categories, t2, fig1, grid, pooled,
                 radars: Optional[Dict[str, pd.DataFrame]], llm_dir: Optional[Path]) -> str:
    sections = [
        "# Reproduction report: *Social Knowledge for Cross-Domain User Preference Modeling*\n\n"
        "Generated by `run_reproduction.py`. 'Paper' columns are from arXiv:2603.10148v1; everything else "
        "was recomputed from `data/` with the SocialVec embeddings.\n",
        dataset_section(per_user, categories),
        table2_section(t2),
        figure1_section(fig1),
        table3_section(fig1, llm_dir),
        table4_section(grid, pooled),
    ]
    if radars is not None:
        sections.append(figure2_section(radars))
    sections.append(NOTES)
    return "\n".join(sections)
