"""Reproduce the main numerical claims from the released aggregate tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper" / "generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce key LungCURE-SafeBench claims from aggregate CSV files.")
    parser.add_argument("--out", type=Path, default=GENERATED / "reproduced_claims.json")
    args = parser.parse_args()

    combined = read_csv(GENERATED / "leaderboard_whitelist12_combined.csv")
    complete = [r for r in combined if r.get("complete") == "1"]
    method_means = read_csv(GENERATED / "paper_method_mean_summary.csv")
    best_su = read_csv(GENERATED / "whitelist12_best_su_by_model.csv")
    best_utility = read_csv(GENERATED / "whitelist12_best_utility_by_model.csv")
    taskfirst = read_csv(GENERATED / "whitelist12_taskfirst_delta_all_models.csv")
    counts = read_csv(GENERATED / "safebench_counts.csv")

    top_su = max(best_su, key=lambda r: f(r, "S--U"))
    top_clean = max(best_utility, key=lambda r: f(r, "Utility"))
    top_tnm = max(best_utility, key=lambda r: f(r, "TNM"))
    mean_delta = next(r for r in taskfirst if r["Model"] == "Mean")

    claims = {
        "dataset": {
            "safebench_rows": line_count(ROOT / "data_safe" / "lungcure_safebench_en_full.jsonl"),
            "original_rows": line_count(ROOT / "data_safe" / "lungcure_original_en.jsonl"),
            "subset_counts": counts,
        },
        "evaluated_configurations": len(complete),
        "mean_permissive_safety": round(mean(f(r, "overall_safety_score") for r in complete), 3),
        "mean_scss": round(mean(f(r, "SCSS") for r in complete), 3),
        "best_safety_utility_row": top_su,
        "best_clean_utility_row": top_clean,
        "best_tnm_row": top_tnm,
        "method_means": method_means,
        "taskfirst_mean_delta": mean_delta,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(claims, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
