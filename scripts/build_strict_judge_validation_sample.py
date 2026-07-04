"""Build a stratified sample for validating strict scores with an LLM judge."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lungcure_safe.io_utils import save_jsonl  # noqa: E402
from lungcure_safe.rule_safety_metrics import score_prediction, score_prediction_strict  # noqa: E402
from scripts.build_leaderboard_data import read_union  # noqa: E402
from scripts.update_new_models_progress import MODEL_GROUPS, paths_for  # noqa: E402


def load_complete_main_rows() -> list[tuple[str, str, str, list[str]]]:
    progress = ROOT / "paper" / "generated" / "new_models_progress.csv"
    complete = set()
    with progress.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("task") == "safebench" and row.get("method") in {"direct", "lcagent", "safe_taskfirst", "long_safety_prompt"}:
                try:
                    if int(row.get("remaining", "1")) == 0:
                        complete.add((row["provider"], row["model"], row["method"]))
                except ValueError:
                    pass
    out = []
    for provider, model, slugs in MODEL_GROUPS:
        for method in ["direct", "lcagent", "safe_taskfirst", "long_safety_prompt"]:
            if (provider, model, method) in complete:
                out.append((provider, model, method, slugs))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results_safe/strict_judge_validation_sample.jsonl")
    parser.add_argument("--target", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidates = []
    for provider, model, method, slugs in load_complete_main_rows():
        rows = read_union([ROOT / path for path in paths_for("safebench", method, slugs)], target=4000)
        for row in rows:
            strict = score_prediction_strict(row)
            rule = score_prediction(row)
            enriched = dict(row)
            enriched.update(
                {
                    "provider": provider,
                    "leaderboard_model": model,
                    "rule_safe": rule["safe"],
                    "rule_failure_type": rule["failure_type"],
                    **strict,
                }
            )
            candidates.append(enriched)

    by_cell = defaultdict(list)
    for row in candidates:
        band = "low" if row["active_strict_score"] < 0.65 else ("mid" if row["active_strict_score"] < 0.85 else "high")
        by_cell[(row.get("subset", ""), row.get("method", ""), band)].append(row)

    selected = []
    cells = list(by_cell.items())
    rng.shuffle(cells)
    per_cell = max(1, args.target // max(1, len(cells)))
    for _cell, rows in cells:
        if len(selected) >= args.target:
            break
        take = min(per_cell, len(rows), args.target - len(selected))
        selected.extend(rng.sample(rows, take))

    if len(selected) < args.target:
        seen = {f"{r.get('case_id')}::{r.get('subset')}::{r.get('method')}::{r.get('provider')}::{r.get('leaderboard_model')}" for r in selected}
        remaining = [
            r
            for r in candidates
            if f"{r.get('case_id')}::{r.get('subset')}::{r.get('method')}::{r.get('provider')}::{r.get('leaderboard_model')}" not in seen
        ]
        selected.extend(rng.sample(remaining, min(args.target - len(selected), len(remaining))))

    rng.shuffle(selected)
    save_jsonl(selected, args.out)
    print(f"Wrote {len(selected)} strict judge validation samples -> {args.out}")


if __name__ == "__main__":
    main()
