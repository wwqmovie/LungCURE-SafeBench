"""Generate paper-ready result tables from SafeBench outputs."""
from collections import Counter
from pathlib import Path

from .io_utils import load_csv, load_jsonl, save_csv


def _fmt(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def build_report(safe_set, summaries, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(safe_set)
    counts = Counter(r["subset"] for r in samples)
    count_rows = [{"subset": k, "num_samples": v} for k, v in sorted(counts.items())]
    save_csv(count_rows, str(out / "safebench_counts.csv"))

    rows = []
    for path in summaries:
        rows.extend(load_csv(path))
    save_csv(rows, str(out / "main_results.csv"))

    headers = ["method", "MIR", "UER", "CGC", "HRS", "harmful_recommendation_rate", "overall_safety_score"]
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & MIR & UER & CGC & HRS & Harm Rate & Overall \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = [row.get(h, "TBD") for h in headers]
        lines.append(f"{vals[0]} & {_fmt(vals[1])} & {_fmt(vals[2])} & {_fmt(vals[3])} & {_fmt(vals[4])} & {_fmt(vals[5])} & {_fmt(vals[6])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (out / "main_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report artifacts -> {out}")
