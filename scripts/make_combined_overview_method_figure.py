from __future__ import annotations

import html
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
BASE = OUT / "fig1_combined_overview_method"


COLORS = {
    "ink": "#1f2937",
    "muted": "#475569",
    "line": "#cbd5e1",
    "panel": "#f8fafc",
    "blue": "#dbeafe",
    "blue_line": "#60a5fa",
    "green": "#dcfce7",
    "green_line": "#22c55e",
    "amber": "#fef3c7",
    "amber_line": "#f59e0b",
    "red": "#fee2e2",
    "red_line": "#ef4444",
    "purple": "#ede9fe",
    "purple_line": "#8b5cf6",
    "cyan": "#e0f2fe",
    "cyan_line": "#06b6d4",
}


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )


def box(
    ax,
    x,
    y,
    w,
    h,
    text="",
    fc="#ffffff",
    ec=None,
    lw=0.8,
    radius=0.9,
    fontsize=7.0,
    weight="normal",
    color=None,
    ha="center",
    va="center",
    pad=0.08,
    z=2,
):
    ec = ec or COLORS["line"]
    color = color or COLORS["ink"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            linespacing=1.15,
            zorder=z + 1,
        )
    return patch


def label(ax, x, y, text, size=7, weight="normal", color=None, ha="left", va="center"):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        linespacing=1.12,
    )


def arrow(ax, x1, y1, x2, y2, color=None, rad=0.0, lw=0.9):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=lw,
        color=color or COLORS["muted"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(arr)
    return arr


def panel(ax, x, y, w, h, letter, title):
    box(ax, x, y, w, h, fc=COLORS["panel"], ec="#d9e2ec", lw=0.8, radius=1.2, z=0)
    label(ax, x + 1.2, y + h - 4.0, letter, size=10.5, weight="bold")
    label(ax, x + 4.3, y + h - 4.0, title, size=8.6, weight="bold")


def build_matplotlib() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 4.05), dpi=600)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    label(
        ax,
        3,
        96.2,
        "LungCURE-SafeBench links traceable safety stress tests to paired model evaluation",
        size=10.0,
        weight="bold",
    )
    label(
        ax,
        3,
        92.3,
        "One figure combines dataset construction, prompted model methods, Task-first safety gating, and final endpoints.",
        size=6.3,
        color=COLORS["muted"],
    )

    panel(ax, 3, 52, 30, 37, "a", "Benchmark construction")
    box(ax, 5, 74, 9.5, 9, "LungCURE\n1,000 text\ncases", COLORS["blue"], COLORS["blue_line"], fontsize=6.2, weight="bold")
    box(ax, 5, 62.5, 9.5, 7.8, "TNM + CDS\nlabels", "#ffffff", COLORS["line"], fontsize=6.1)
    arrow(ax, 14.9, 78.4, 18.0, 78.4)
    arrow(ax, 14.9, 66.5, 18.0, 66.5)
    box(ax, 18, 73.5, 12.5, 9.5, "SafeBench\n4 x 1,000 rows", COLORS["green"], COLORS["green_line"], fontsize=6.2, weight="bold")
    chips = [
        ("Missing info", COLORS["cyan"]),
        ("Uncertain evidence", COLORS["amber"]),
        ("Counterfactual", COLORS["purple"]),
        ("Harm screen", COLORS["red"]),
    ]
    for i, (txt, fc) in enumerate(chips):
        box(ax, 18.4, 68.4 - i * 2.9, 11.7, 2.1, txt, fc, COLORS["line"], fontsize=4.5, radius=0.35, pad=0.02)
    box(ax, 18, 53.5, 12.5, 4.5, "Original task\nclean utility", "#ffffff", COLORS["line"], fontsize=5.3)

    panel(ax, 36, 52, 29, 37, "b", "Fixed cohort evaluation")
    box(ax, 38.3, 76.5, 9.7, 7.4, "13 API\nmodels", COLORS["blue"], COLORS["blue_line"], fontsize=6.2, weight="bold")
    box(ax, 38.3, 63.0, 9.7, 8.8, "5 prompt\nmethods", COLORS["amber"], COLORS["amber_line"], fontsize=6.2, weight="bold")
    method_labels = ["Direct", "LCAgent", "Plain Safe", "Task-first", "Long prompt"]
    for i, txt in enumerate(method_labels):
        box(ax, 50.1, 80.8 - i * 4.0, 11.8, 2.7, txt, "#ffffff", COLORS["line"], fontsize=5.1, radius=0.45, pad=0.02)
    arrow(ax, 48.3, 80.2, 49.8, 80.2)
    arrow(ax, 48.3, 67.4, 49.8, 67.4)
    box(ax, 48.7, 55.5, 13.0, 5.8, "Prediction JSONL\nresumable by case", COLORS["green"], COLORS["green_line"], fontsize=5.5)

    panel(ax, 68, 52, 29, 37, "c", "Paired endpoints")
    for i, (txt, fc) in enumerate([("MIR", COLORS["cyan"]), ("UER", COLORS["amber"]), ("CGC", COLORS["purple"]), ("HRS", COLORS["red"])]):
        box(ax, 70.2 + i * 5.5, 76.5, 4.6, 4.4, txt, fc, COLORS["line"], fontsize=5.7, weight="bold", radius=0.45)
    arrow(ax, 81.5, 74.8, 81.5, 70.5)
    box(ax, 73.2, 65.5, 16.5, 5.8, "SCSS\nstrict geometric mean", COLORS["red"], COLORS["red_line"], fontsize=5.7, weight="bold")
    box(ax, 71.5, 56.0, 8.7, 5.7, "Clean utility\nTNM", COLORS["blue"], COLORS["blue_line"], fontsize=5.4)
    box(ax, 81.1, 56.0, 8.7, 5.7, "Treatment\nF1", COLORS["blue"], COLORS["blue_line"], fontsize=5.4)
    arrow(ax, 89.8, 65.5, 92.0, 64.2)
    arrow(ax, 89.8, 58.8, 92.0, 61.1)
    box(ax, 90.5, 58.5, 5.2, 9.5, "Evidence\npackage", COLORS["green"], COLORS["green_line"], fontsize=4.7, weight="bold")

    panel(ax, 3, 9, 94, 36, "d", "Task-first Safe-LCAgent scaffold")
    xs = [7.0, 27.8, 50.0, 72.2]
    ws = [14.5, 16.0, 17.4, 17.2]
    ys = 23.2
    hs = 9.7
    box(ax, xs[0], ys, ws[0], hs, "Case text\n+ perturbation\nmetadata", COLORS["blue"], COLORS["blue_line"], fontsize=6.3, weight="bold")
    box(ax, xs[1], ys, ws[1], hs, "Stage 1\nstructured clinical\nanswer", "#ffffff", COLORS["line"], fontsize=6.3)
    gate_box = box(ax, xs[2], ys, ws[2], hs, "Safety gate audit", COLORS["purple"], COLORS["purple_line"], fontsize=6.5, weight="bold")
    for i, txt in enumerate(["MIG", "UEF", "GCV", "HRC"]):
        box(ax, xs[2] + 1.3 + i * 3.8, ys + 1.3, 3.1, 2.7, txt, "#ffffff", COLORS["line"], fontsize=5.0, radius=0.35, pad=0.02)
    box(ax, xs[3], ys, ws[3], hs, "Revised answer\nor explicit\nclinical caution", COLORS["green"], COLORS["green_line"], fontsize=6.2, weight="bold")
    for i in range(3):
        arrow(ax, xs[i] + ws[i] + 1.1, ys + hs / 2, xs[i + 1] - 1.1, ys + hs / 2)
    box(ax, 16.0, 13.2, 68.0, 5.4, "Prompted two-stage baseline; no model training.", "#ffffff", COLORS["line"], fontsize=6.2)

    fig.savefig(BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(BASE.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


class DrawioBuilder:
    def __init__(self):
        self.next_id = 2
        self.cells = ['        <mxCell id="0" />', '        <mxCell id="1" parent="0" />']

    def _id(self):
        val = str(self.next_id)
        self.next_id += 1
        return val

    @staticmethod
    def _geom(x, y, w, h):
        sx, sy = 22.0, 11.5
        return x * sx, (100 - y - h) * sy, w * sx, h * sy

    def rect(self, x, y, w, h, text, fill, stroke, font_size=16, bold=False, rounded=True):
        cid = self._id()
        dx, dy, dw, dh = self._geom(x, y, w, h)
        style = (
            f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;arcSize=8;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
            f"fontFamily=Helvetica;fontSize={font_size};fontColor={COLORS['ink']};"
            f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;"
        )
        value = escape(text).replace("\n", "&#xa;")
        self.cells.append(
            f'        <mxCell id="{cid}" value="{value}" style="{style}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{dx:.1f}" y="{dy:.1f}" width="{dw:.1f}" height="{dh:.1f}" as="geometry" />\n'
            f"        </mxCell>"
        )
        return cid

    def text(self, x, y, w, h, text, font_size=18, bold=False, color=None, align="left"):
        cid = self._id()
        dx, dy, dw, dh = self._geom(x, y, w, h)
        style = (
            "text;html=1;strokeColor=none;fillColor=none;verticalAlign=middle;"
            f"align={align};fontFamily=Helvetica;fontSize={font_size};"
            f"fontColor={color or COLORS['ink']};fontStyle={1 if bold else 0};"
        )
        value = escape(text).replace("\n", "&#xa;")
        self.cells.append(
            f'        <mxCell id="{cid}" value="{value}" style="{style}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{dx:.1f}" y="{dy:.1f}" width="{dw:.1f}" height="{dh:.1f}" as="geometry" />\n'
            f"        </mxCell>"
        )
        return cid

    def edge(self, source, target):
        cid = self._id()
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
            "jettySize=auto;html=1;strokeColor=#475569;strokeWidth=1.3;"
            "endArrow=block;endFill=1;"
        )
        self.cells.append(
            f'        <mxCell id="{cid}" value="" style="{style}" edge="1" parent="1" source="{source}" target="{target}">\n'
            '          <mxGeometry relative="1" as="geometry" />\n'
            "        </mxCell>"
        )
        return cid

    def xml(self):
        cells = "\n".join(self.cells)
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="drawio" version="26.0.0">
  <diagram name="Combined overview and method">
    <mxGraphModel dx="2200" dy="1150" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2200" pageHeight="1150" math="0" shadow="0">
      <root>
{cells}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''


def build_drawio() -> None:
    b = DrawioBuilder()
    b.text(3, 94, 92, 5, "LungCURE-SafeBench links traceable safety stress tests to paired model evaluation", 24, True)
    b.text(3, 90.5, 92, 4, "Dataset construction, prompted methods, Task-first safety gating, and final endpoints in one view.", 15, False, COLORS["muted"])

    a = b.rect(3, 52, 30, 37, "", COLORS["panel"], "#d9e2ec", 1)
    b.text(4.2, 84.5, 3, 4, "a", 24, True)
    b.text(7.3, 84.5, 20, 4, "Benchmark construction", 19, True)
    source = b.rect(5, 74, 9.5, 9, "LungCURE\n1,000 text\ncases", COLORS["blue"], COLORS["blue_line"], 14, True)
    labels = b.rect(5, 62.5, 9.5, 7.8, "TNM + CDS\nlabels", "#ffffff", COLORS["line"], 13)
    safe = b.rect(18, 69.8, 12.5, 13.6, "SafeBench\n4 x 1,000 rows", COLORS["green"], COLORS["green_line"], 14, True)
    orig = b.rect(18, 54.5, 12.5, 5.5, "Original task\nclean utility", "#ffffff", COLORS["line"], 13)
    b.edge(source, safe)
    b.edge(labels, safe)
    b.edge(labels, orig)

    b.rect(36, 52, 29, 37, "", COLORS["panel"], "#d9e2ec", 1)
    b.text(37.2, 84.5, 3, 4, "b", 24, True)
    b.text(40.3, 84.5, 20, 4, "Fixed cohort evaluation", 19, True)
    models = b.rect(38.3, 76.5, 9.7, 7.4, "13 API\nmodels", COLORS["blue"], COLORS["blue_line"], 14, True)
    methods = b.rect(38.3, 63.0, 9.7, 8.8, "5 prompt\nmethods", COLORS["amber"], COLORS["amber_line"], 14, True)
    pred = b.rect(48.7, 55.5, 13.0, 5.8, "Prediction JSONL\nresumable by case", COLORS["green"], COLORS["green_line"], 13)
    b.edge(models, pred)
    b.edge(methods, pred)

    b.rect(68, 52, 29, 37, "", COLORS["panel"], "#d9e2ec", 1)
    b.text(69.2, 84.5, 3, 4, "c", 24, True)
    b.text(72.3, 84.5, 20, 4, "Paired endpoints", 19, True)
    strict = b.rect(73.2, 65.5, 16.5, 5.8, "SCSS\nstrict geometric mean", COLORS["red"], COLORS["red_line"], 13, True)
    util = b.rect(71.5, 56.0, 18.3, 5.7, "Clean utility\nTNM + treatment F1", COLORS["blue"], COLORS["blue_line"], 13)
    evidence = b.rect(90.5, 58.5, 5.2, 9.5, "Evidence\npackage", COLORS["green"], COLORS["green_line"], 11, True)
    b.edge(strict, evidence)
    b.edge(util, evidence)

    b.rect(3, 9, 94, 36, "", COLORS["panel"], "#d9e2ec", 1)
    b.text(4.2, 40.5, 3, 4, "d", 24, True)
    b.text(7.3, 40.5, 38, 4, "Task-first Safe-LCAgent scaffold", 19, True)
    case = b.rect(7.0, 23.2, 14.5, 9.7, "Case text\n+ perturbation\nmetadata", COLORS["blue"], COLORS["blue_line"], 14, True)
    ans = b.rect(27.8, 23.2, 16.0, 9.7, "Stage 1\nstructured clinical\nanswer", "#ffffff", COLORS["line"], 14)
    gates = b.rect(50.0, 23.2, 17.4, 9.7, "Safety gate audit\nMIG / UEF / GCV / HRC", COLORS["purple"], COLORS["purple_line"], 14, True)
    rev = b.rect(72.2, 23.2, 17.2, 9.7, "Revised answer\nor explicit\nclinical caution", COLORS["green"], COLORS["green_line"], 14, True)
    b.edge(case, ans)
    b.edge(ans, gates)
    b.edge(gates, rev)
    b.rect(16.0, 13.2, 68.0, 5.4, "Prompted two-stage baseline; no model training.", "#ffffff", COLORS["line"], 13)

    BASE.with_suffix(".drawio").write_text(b.xml(), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_matplotlib()
    build_drawio()
    print(f"Wrote {BASE}.{{png,pdf,svg,tiff,drawio}}")


if __name__ == "__main__":
    main()
