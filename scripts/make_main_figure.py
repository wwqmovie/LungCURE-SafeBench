"""Create the LungCURE-SafeBench main schematic figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.4,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


COLORS = {
    "ink": "#172033",
    "muted": "#667085",
    "line": "#D7DEE8",
    "panel": "#F7F9FC",
    "blue": "#DCEBFF",
    "blue_edge": "#477DC2",
    "green": "#DFF3EA",
    "green_edge": "#3F8F72",
    "orange": "#FDEBD2",
    "orange_edge": "#C47A24",
    "red": "#FCE2DF",
    "red_edge": "#C75C54",
    "purple": "#EAE4FA",
    "purple_edge": "#7B66B5",
    "gray": "#EEF2F6",
    "gray_edge": "#8792A2",
    "dark": "#253247",
}


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body="",
    fc="#FFFFFF",
    ec=None,
    title_color=None,
    body_color=None,
    lw=1.15,
    radius=0.035,
    align="center",
    fontsize=7.2,
    title_size=8.0,
):
    ec = ec or COLORS["line"]
    title_color = title_color or COLORS["ink"]
    body_color = body_color or COLORS["ink"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    tx = x + (w / 2 if align == "center" else 0.035 * w)
    ha = "center" if align == "center" else "left"
    ax.text(
        tx,
        y + h - 0.26 * h,
        title,
        ha=ha,
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    if body:
        ax.text(
            tx,
            y + 0.38 * h,
            body,
            ha=ha,
            va="center",
            fontsize=fontsize,
            color=body_color,
            linespacing=1.2,
            zorder=3,
        )
    return patch


def add_arrow(ax, start, end, color=None, rad=0.0, lw=1.25, mutation_scale=10):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color or COLORS["gray_edge"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)
    return arr


def add_panel_label(ax, label, x, y):
    ax.text(
        x,
        y,
        label,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
    )


def add_section_title(ax, text, x, y, subtitle=None):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    if subtitle:
        ax.text(
            x,
            y - 0.025,
            subtitle,
            ha="left",
            va="top",
            fontsize=6.8,
            color=COLORS["muted"],
        )


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(7.35, 5.25), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.add_patch(Rectangle((0.018, 0.02), 0.964, 0.955, fc="white", ec="none"))

    ax.text(
        0.04,
        0.958,
        "LungCURE-SafeBench: safety- and counterfactual-enhanced evaluation for lung cancer CDS",
        ha="left",
        va="top",
        fontsize=11.8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.04,
        0.918,
        "Text-only English benchmark construction, task-first clinical reasoning, and rule-based safety protocol.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )

    # Panel a: data construction.
    add_panel_label(ax, "a", 0.04, 0.835)
    add_section_title(
        ax,
        "Benchmark construction",
        0.065,
        0.842,
        "1,000 original English text cases -> 4,000 safety stress-test cases",
    )
    add_box(
        ax,
        0.055,
        0.695,
        0.18,
        0.105,
        "LungCURE text cases",
        "EN clinical notes\nTNM ground truth\nCDS ground truth",
        fc=COLORS["gray"],
        ec=COLORS["gray_edge"],
        align="left",
    )
    add_box(
        ax,
        0.285,
        0.695,
        0.2,
        0.105,
        "SafeBench builder",
        "parse -> align -> perturb\ntraceable case_id\nnon-overwriting JSONL",
        fc=COLORS["blue"],
        ec=COLORS["blue_edge"],
        align="left",
    )
    add_arrow(ax, (0.237, 0.748), (0.282, 0.748), COLORS["blue_edge"])

    variants = [
        ("Missing", "mask biomarkers,\nECOG, treatment line", COLORS["blue"], COLORS["blue_edge"]),
        ("Uncertain", "suspected /\nindeterminate evidence", COLORS["orange"], COLORS["orange_edge"]),
        ("Counterfactual", "flip guideline\nconditions", COLORS["purple"], COLORS["purple_edge"]),
        ("Harm", "screen risky\nrecommendations", COLORS["red"], COLORS["red_edge"]),
    ]
    x0s = [0.055, 0.185, 0.315, 0.445]
    for (title, body, fc, ec), x in zip(variants, x0s):
        add_box(
            ax,
            x,
            0.548,
            0.112,
            0.103,
            title,
            f"{body}\nn=1,000",
            fc=fc,
            ec=ec,
            fontsize=6.15,
            title_size=6.9,
            radius=0.025,
        )
        add_arrow(ax, (0.385, 0.693), (x + 0.056, 0.653), ec, rad=0.07, lw=0.9, mutation_scale=8)

    # Panel b: methods and model evaluation.
    add_panel_label(ax, "b", 0.04, 0.463)
    add_section_title(
        ax,
        "Evaluation setting",
        0.065,
        0.47,
        "Same SafeBench samples are submitted to each method and model.",
    )
    method_boxes = [
        ("Direct prompting", "single clinical\nrecommendation prompt", COLORS["gray"], COLORS["gray_edge"]),
        ("LCAgent-compatible", "TNM decomposition\n+ treatment routing", COLORS["blue"], COLORS["blue_edge"]),
        ("Safe-LCAgent\n(Task-first)", "answer task first\nthen apply safety gates", COLORS["green"], COLORS["green_edge"]),
        ("Controls", "plain Safe\nlong safety prompt\nmodule ablations", COLORS["orange"], COLORS["orange_edge"]),
    ]
    for i, (title, body, fc, ec) in enumerate(method_boxes):
        add_box(
            ax,
            0.055 + i * 0.128,
            0.318,
            0.112,
            0.11,
            title,
            body,
            fc=fc,
            ec=ec,
            fontsize=6.2,
            title_size=6.8,
            radius=0.024,
        )
    add_box(
        ax,
        0.105,
        0.205,
        0.39,
        0.067,
        "Multi-LLM comparison",
        "completed models enter the main table; unfinished providers remain progress-only",
        fc="#FFFFFF",
        ec=COLORS["line"],
        fontsize=6.7,
        title_size=7.2,
    )
    for x in [0.111, 0.239, 0.367, 0.495]:
        add_arrow(ax, (x, 0.318), (0.30, 0.273), COLORS["gray_edge"], rad=0.02, lw=0.8, mutation_scale=7)

    # Panel c: task-first Safe-LCAgent mechanics.
    add_panel_label(ax, "c", 0.61, 0.835)
    add_section_title(
        ax,
        "Task-first Safe-LCAgent",
        0.635,
        0.852,
        "Preserve clinical utility before safety critique.",
    )
    add_box(
        ax,
        0.625,
        0.715,
        0.14,
        0.065,
        "Clinical task",
        "stage + recommend",
        fc=COLORS["green"],
        ec=COLORS["green_edge"],
        fontsize=6.6,
        title_size=7.1,
    )
    add_box(
        ax,
        0.805,
        0.715,
        0.14,
        0.065,
        "Initial answer",
        "structured CDS output",
        fc="#FFFFFF",
        ec=COLORS["line"],
        fontsize=6.6,
        title_size=7.1,
    )
    add_arrow(ax, (0.768, 0.748), (0.802, 0.748), COLORS["green_edge"])

    gate_data = [
        ("MIG", "missing information gate", COLORS["blue"], COLORS["blue_edge"]),
        ("UEF", "uncertainty evidence filter", COLORS["orange"], COLORS["orange_edge"]),
        ("GCV", "guideline condition verifier", COLORS["purple"], COLORS["purple_edge"]),
        ("HRC", "harmful recommendation critic", COLORS["red"], COLORS["red_edge"]),
    ]
    gate_y = [0.625, 0.54, 0.455, 0.37]
    for (abbr, desc, fc, ec), y in zip(gate_data, gate_y):
        add_box(
            ax,
            0.64,
            y,
            0.27,
            0.055,
            abbr,
            desc,
            fc=fc,
            ec=ec,
            fontsize=6.4,
            title_size=7.2,
            align="left",
            radius=0.02,
        )
    add_arrow(ax, (0.875, 0.715), (0.875, 0.708), COLORS["gray_edge"], lw=0.9, mutation_scale=7)
    for y in gate_y:
        add_arrow(ax, (0.875, 0.71), (0.875, y + 0.058), COLORS["gray_edge"], lw=0.8, mutation_scale=6)
    add_box(
        ax,
        0.64,
        0.255,
        0.27,
        0.065,
        "Safety-aware final CDS",
        "ask for missing data, avoid overcalling uncertainty,\nverify prerequisites, and flag harm",
        fc="#FFFFFF",
        ec=COLORS["green_edge"],
        fontsize=6.2,
        title_size=7.2,
        radius=0.022,
    )
    add_arrow(ax, (0.775, 0.368), (0.775, 0.322), COLORS["green_edge"], lw=1.0, mutation_scale=8)

    # Panel d: evaluation protocol.
    add_panel_label(ax, "d", 0.61, 0.206)
    add_section_title(
        ax,
        "Safety evaluation protocol",
        0.635,
        0.202,
        "Metrics are benchmark-level endpoints, not internal method scores.",
    )
    metric_specs = [
        ("MIR", "missing-info\nrecognition", COLORS["blue_edge"]),
        ("UER", "uncertainty\nevidence restraint", COLORS["orange_edge"]),
        ("CGC", "condition-guideline\nconsistency", COLORS["purple_edge"]),
        ("HRS", "harmful-recommendation\nsafety", COLORS["red_edge"]),
    ]
    for i, (abbr, desc, edge) in enumerate(metric_specs):
        x = 0.635 + (i % 2) * 0.155
        y = 0.088 - (i // 2) * 0.062
        add_box(
            ax,
            x,
            y,
            0.135,
            0.055,
            abbr,
            desc,
            fc="#FFFFFF",
            ec=edge,
            fontsize=5.45,
            title_size=7.1,
            radius=0.018,
        )

    # Cross-panel arrows.
    add_arrow(ax, (0.49, 0.60), (0.62, 0.60), COLORS["gray_edge"], rad=-0.08, lw=1.0, mutation_scale=9)
    add_arrow(ax, (0.495, 0.238), (0.63, 0.156), COLORS["gray_edge"], rad=0.05, lw=0.9, mutation_scale=8)
    add_arrow(ax, (0.775, 0.253), (0.775, 0.158), COLORS["green_edge"], lw=1.0, mutation_scale=9)

    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUT / "fig1_lungcure_safebench_overview"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
