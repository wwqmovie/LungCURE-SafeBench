"""Summarize agreement between rule metrics and LLM-judge labels."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lungcure_safe.io_utils import load_jsonl, save_csv, save_jsonl
from lungcure_safe.rule_safety_metrics import score_prediction


def _judge_safe(row: dict) -> int:
    judge = row.get("judge") or {}
    value = judge.get("safe")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "safe"})
    return int(float(row.get("judge_score", 0)) >= 0.5)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="results_safe/judge_rule_agreement_summary.csv")
    parser.add_argument("--disagreements", default="results_safe/judge_rule_disagreements.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    details = []
    for row in rows:
        rule = score_prediction(row)
        judge_safe = _judge_safe(row)
        details.append(
            {
                "method": row.get("method", ""),
                "subset": row.get("subset", ""),
                "case_id": row.get("case_id", ""),
                "rule_safe": int(rule["safe"]),
                "judge_safe": judge_safe,
                "agree": int(int(rule["safe"]) == judge_safe),
                "rule_failure_type": rule.get("failure_type", ""),
                "judge_failure_type": (row.get("judge") or {}).get("failure_type", ""),
                "judge_rationale": (row.get("judge") or {}).get("rationale", ""),
            }
        )

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    groups[("all", "all")] = details
    for item in details:
        groups[(item["method"], "all")].append(item)
        groups[("all", item["subset"])].append(item)
        groups[(item["method"], item["subset"])].append(item)

    summary = []
    for (method, subset), vals in sorted(groups.items()):
        summary.append(
            {
                "method": method,
                "subset": subset,
                "num_samples": len(vals),
                "agreement": _mean([v["agree"] for v in vals]),
                "rule_safe_rate": _mean([v["rule_safe"] for v in vals]),
                "judge_safe_rate": _mean([v["judge_safe"] for v in vals]),
            }
        )
    save_csv(summary, args.out)
    save_jsonl([row for row in details if not row["agree"]], args.disagreements)
    print(f"Wrote {args.out} and {args.disagreements}")


if __name__ == "__main__":
    main()
