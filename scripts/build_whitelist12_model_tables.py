"""Build all-model paper tables for the included completed models."""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper" / "generated"
COMBINED = GEN / "leaderboard_whitelist12_combined.csv"

MODEL_ORDER = [
    ("agnes", "agnes-2.0-flash", "Agnes-2.0-Flash"),
    ("dmxapi", "deepseek-v4-flash", "DeepSeek-v4-Flash"),
    ("mimo", "mimo-v2.5", "MiMo-v2.5"),
    ("nvidia", "openai/gpt-oss-120b", "GPT-OSS-120B"),
    ("nvidia", "llama-4-maverick-17b-128e-instruct", "Llama-4-Maverick"),
    ("stepfun", "step-3.5-flash", "Step-3.5-Flash"),
    ("stepfun+avi", "step-3.7-flash", "Step-3.7-Flash"),
    ("xfyun", "Qwen3.5-2B", "Qwen3.5-2B"),
    ("xfyun", "Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B"),
    ("xfyun", "Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B"),
    ("siliconflow", "GLM-Z1-9B-0414", "GLM-Z1-9B"),
    ("siliconflow", "GLM-4-9B-0414", "GLM-4-9B"),
    ("siliconflow", "DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-Qwen3-8B"),
]

METHOD_LABELS = {
    "direct": "Direct",
    "lcagent": "LCAgent",
    "safe": "Plain Safe",
    "safe_taskfirst": "Task-first",
    "long_safety_prompt": "Long prompt",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row["provider"], row["model"]


def main() -> None:
    rows = [row for row in read_csv(COMBINED) if row["complete"] == "1"]
    by_model: dict[tuple[str, str], list[dict[str, str]]] = {}
    by_model_method: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        by_model.setdefault(row_key(row), []).append(row)
        by_model_method[(row["provider"], row["model"], row["method"])] = row

    best_su_rows: list[dict[str, str]] = []
    utility_rows: list[dict[str, str]] = []
    delta_rows: list[dict[str, str]] = []
    delta_values: list[dict[str, float]] = []

    for provider, model, display in MODEL_ORDER:
        model_rows = by_model[(provider, model)]
        best_su = max(model_rows, key=lambda row: f(row, "paper_ready_score"))
        best_utility = max(model_rows, key=lambda row: f(row, "clean_utility_score"))

        best_su_rows.append(
            {
                "Model": display,
                "Best method": method_label(best_su["method"]),
                "S--U": fmt(f(best_su, "paper_ready_score")),
                "SCSS": fmt(f(best_su, "SCSS")),
                "Utility": fmt(f(best_su, "clean_utility_score")),
                "Pass": fmt(f(best_su, "overall_safety_score")),
            }
        )

        utility_rows.append(
            {
                "Model": display,
                "Best utility method": method_label(best_utility["method"]),
                "Utility": fmt(f(best_utility, "clean_utility_score")),
                "TNM": fmt(f(best_utility, "TNM_acc")),
                "T": fmt(f(best_utility, "T_acc")),
                "N": fmt(f(best_utility, "N_acc")),
                "M": fmt(f(best_utility, "M_acc")),
                "Treat F1": fmt(f(best_utility, "treatment_token_f1")),
            }
        )

        direct = by_model_method[(provider, model, "direct")]
        taskfirst = by_model_method[(provider, model, "safe_taskfirst")]
        deltas = {
            "delta_su": f(taskfirst, "paper_ready_score") - f(direct, "paper_ready_score"),
            "delta_scss": f(taskfirst, "SCSS") - f(direct, "SCSS"),
            "delta_utility": f(taskfirst, "clean_utility_score") - f(direct, "clean_utility_score"),
        }
        delta_values.append(deltas)
        delta_rows.append(
            {
                "Model": display,
                "Delta S--U": fmt_delta(deltas["delta_su"]),
                "Delta SCSS": fmt_delta(deltas["delta_scss"]),
                "Delta Utility": fmt_delta(deltas["delta_utility"]),
            }
        )

    delta_rows.append(
        {
            "Model": "Mean",
            "Delta S--U": fmt_delta(mean(v["delta_su"] for v in delta_values)),
            "Delta SCSS": fmt_delta(mean(v["delta_scss"] for v in delta_values)),
            "Delta Utility": fmt_delta(mean(v["delta_utility"] for v in delta_values)),
        }
    )

    write_csv(GEN / "whitelist12_best_su_by_model.csv", best_su_rows)
    write_csv(GEN / "whitelist12_best_utility_by_model.csv", utility_rows)
    write_csv(GEN / "whitelist12_taskfirst_delta_all_models.csv", delta_rows)
    print(f"Wrote all-model tables to {GEN}")


if __name__ == "__main__":
    main()
