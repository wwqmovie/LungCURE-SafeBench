"""Analyze strict-score experiments for LungCURE-SafeBench."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lungcure_safe.rule_safety_metrics import _geomean, score_prediction_strict  # noqa: E402
from scripts.build_leaderboard_data import read_union  # noqa: E402
from scripts.update_new_models_progress import MODEL_GROUPS, SAFEBENCH_METHODS, paths_for  # noqa: E402

OUT_DIR = ROOT / "paper" / "generated"
PROGRESS_CSV = OUT_DIR / "new_models_progress.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_complete_safebench() -> set[tuple[str, str, str]]:
    complete = set()
    if not PROGRESS_CSV.exists():
        return complete
    with PROGRESS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("task") != "safebench":
                continue
            try:
                remaining = int(row.get("remaining", "1"))
            except ValueError:
                continue
            if remaining == 0:
                complete.add((row["provider"], row["model"], row["method"]))
    return complete


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def strict_summary(rows: list[dict[str, Any]], apply_template_penalty: bool) -> dict[str, float]:
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scored = score_prediction_strict(row, apply_template_penalty=apply_template_penalty)
        by_subset[row.get("subset", "")].append(scored)
    mir = mean([r["MIR_strict_row"] for r in by_subset.get("missing", [])]) if by_subset.get("missing") else 0.0
    uer = mean([r["UER_strict_row"] for r in by_subset.get("uncertain", [])]) if by_subset.get("uncertain") else 0.0
    cgc = mean([r["CGC_strict_row"] for r in by_subset.get("counterfactual", [])]) if by_subset.get("counterfactual") else 0.0
    hrs = mean([r["HRS_strict_row"] for r in by_subset.get("harm", [])]) if by_subset.get("harm") else 0.0
    template = mean(
        [r["template_penalty"] for vals in by_subset.values() for r in vals]
    ) if by_subset else 0.0
    harm = mean([r["harm_severity"] for vals in by_subset.values() for r in vals]) if by_subset else 0.0
    return {
        "MIR_strict": round(mir, 6),
        "UER_strict": round(uer, 6),
        "CGC_strict": round(cgc, 6),
        "HRS_strict": round(hrs, 6),
        "SCSS": round(_geomean([mir, uer, cgc, hrs]), 6),
        "mean_template_penalty": round(template, 6),
        "mean_harm_severity": round(harm, 6),
    }


def main() -> None:
    delta_rows = []
    delta_path = OUT_DIR / "leaderboard_strict_delta_diagnostics.csv"
    if delta_path.exists():
        with delta_path.open(newline="", encoding="utf-8") as handle:
            delta_rows = list(csv.DictReader(handle))

    deltas = [float(r["delta"]) for r in delta_rows]
    pass_scores = [float(r["pass_overall"]) for r in delta_rows]
    scss_scores = [float(r["SCSS"]) for r in delta_rows]
    ceiling_rows = [
        {
            "metric": "pass_overall",
            **summarize(pass_scores),
            "num_rows": len(pass_scores),
            "rows_ge_0_95": sum(v >= 0.95 for v in pass_scores),
            "rows_ge_0_99": sum(v >= 0.99 for v in pass_scores),
        },
        {
            "metric": "SCSS",
            **summarize(scss_scores),
            "num_rows": len(scss_scores),
            "rows_ge_0_95": sum(v >= 0.95 for v in scss_scores),
            "rows_ge_0_99": sum(v >= 0.99 for v in scss_scores),
        },
        {
            "metric": "pass_minus_SCSS_delta",
            **summarize(deltas),
            "num_rows": len(deltas),
            "rows_ge_0_05": sum(v >= 0.05 for v in deltas),
            "rows_ge_0_10": sum(v >= 0.10 for v in deltas),
        },
    ]
    write_csv(OUT_DIR / "strict_vs_pass_ceiling_effect_summary.csv", ceiling_rows)

    complete = load_complete_safebench()
    ablation_rows = []
    for provider, model, slugs in MODEL_GROUPS:
        for method in SAFEBENCH_METHODS:
            if (provider, model, method) not in complete:
                continue
            rows = read_union([ROOT / path for path in paths_for("safebench", method, slugs)], target=4000)
            if len(rows) < 4000:
                continue
            with_penalty = strict_summary(rows, apply_template_penalty=True)
            without_penalty = strict_summary(rows, apply_template_penalty=False)
            ablation_rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "method": method,
                    "SCSS_with_template_penalty": with_penalty["SCSS"],
                    "SCSS_without_template_penalty": without_penalty["SCSS"],
                    "template_penalty_effect": round(without_penalty["SCSS"] - with_penalty["SCSS"], 6),
                    "mean_template_penalty": with_penalty["mean_template_penalty"],
                    "mean_harm_severity": with_penalty["mean_harm_severity"],
                    "MIR_with_penalty": with_penalty["MIR_strict"],
                    "UER_with_penalty": with_penalty["UER_strict"],
                    "CGC_with_penalty": with_penalty["CGC_strict"],
                    "HRS_with_penalty": with_penalty["HRS_strict"],
                }
            )
    ablation_rows.sort(key=lambda r: r["template_penalty_effect"], reverse=True)
    write_csv(OUT_DIR / "template_penalty_ablation.csv", ablation_rows)

    payload = {
        "ceiling_effect": ceiling_rows,
        "template_penalty_ablation_top": ablation_rows[:20],
        "outputs": {
            "ceiling_effect_csv": "paper/generated/strict_vs_pass_ceiling_effect_summary.csv",
            "template_penalty_ablation_csv": "paper/generated/template_penalty_ablation.csv",
        },
    }
    (OUT_DIR / "strict_score_experiments.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote strict score experiments: {len(ablation_rows)} complete SafeBench rows")


if __name__ == "__main__":
    main()
