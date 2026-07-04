"""Original LungCURE-style metric proxies over unperturbed case predictions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import load_jsonl, save_csv, save_jsonl


TREATMENT_CONCEPTS = {
    "surgery": ["surgery", "surgical", "resection", "lobectomy", "segmentectomy", "pneumonectomy"],
    "chemotherapy": ["chemotherapy", "chemo", "platinum", "cisplatin", "carboplatin", "pemetrexed", "paclitaxel", "docetaxel", "gemcitabine", "etoposide"],
    "radiotherapy": ["radiotherapy", "radiation", "sbrt", "stereotactic", "crt", "chemoradiotherapy"],
    "immunotherapy": ["immunotherapy", "immune checkpoint", "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "cemiplimab", "ipilimumab"],
    "targeted": ["targeted", "egfr", "alk", "ros1", "braf", "met", "ret", "ntrk", "osimertinib", "gefitinib", "erlotinib", "alectinib", "crizotinib", "selpercatinib"],
    "antiangiogenic": ["bevacizumab", "anti-angiogenic", "antiangiogenic", "angiogenesis"],
    "maintenance": ["maintenance"],
    "observation": ["observation", "follow-up", "follow up", "surveillance", "best supportive"],
}


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _norm_stage(value: str) -> str:
    value = _safe_text(value).upper().strip()
    value = value.replace("（", "(").replace("）", ")")
    match = re.search(r"\b([TNM])\s*([0-4X][A-C]?)\b", value)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    match = re.search(r"\b([TNM][0-4X][A-C]?)\b", value)
    return match.group(1) if match else ""


def _stage_rank(stage: str) -> int:
    stage = _norm_stage(stage)
    if not stage:
        return -1
    digits = re.findall(r"[0-4X]", stage)
    if not digits:
        return -1
    if digits[0] == "X":
        return -1
    return int(digits[0])


def _best_stage(values: Any, prefix: str) -> str:
    if isinstance(values, str):
        items = [values]
    elif isinstance(values, list):
        items = values
    else:
        items = []
    stages = [_norm_stage(_safe_text(v)) for v in items]
    stages = [s for s in stages if s.startswith(prefix)]
    if not stages:
        return ""
    return sorted(stages, key=_stage_rank)[-1]


def derive_gt_tnm(tnm_gt: dict[str, Any]) -> dict[str, str]:
    """Derive coarse T/N/M stage labels from LungCURE answer fields."""
    t = "TX" if _safe_text(tnm_gt.get("q_tx")).lower() == "yes" else ""
    if not t:
        t_candidates = []
        for key in ("q_t_size", "q_t1", "q_t2", "q_t2a", "q_t2b", "q_t3", "q_t4", "q_pulmonary"):
            val = tnm_gt.get(key)
            if isinstance(val, list):
                t_candidates.extend(val)
            elif val:
                t_candidates.append(val)
        t = _best_stage(t_candidates, "T")

    n_candidates = tnm_gt.get("q_n_stage")
    n = _best_stage(n_candidates, "N")
    if not n and _safe_text(tnm_gt.get("q_nx")).lower() == "yes":
        n = "NX"
    if not n:
        n = "N0"

    if _safe_text(tnm_gt.get("q_m0")).lower() == "yes":
        m = "M0"
    else:
        m = _best_stage(tnm_gt.get("q_m_detail"), "M")
        if not m:
            for key in ("q_m1a", "q_m1b", "q_m1c"):
                if tnm_gt.get(key):
                    m = key.replace("q_", "").upper()
        if not m:
            m = "M0"

    return {"T": t, "N": n, "M": m}


def parse_pred_tnm(text: str) -> dict[str, str]:
    text = _safe_text(text).upper()
    found: dict[str, str] = {}
    for prefix in ("T", "N", "M"):
        matches = re.findall(rf"\b{prefix}\s*([0-4X][A-C]?)\b", text)
        if matches:
            found[prefix] = f"{prefix}{matches[-1]}"
            continue
        compact = re.findall(rf"\b({prefix}[0-4X][A-C]?)\b", text)
        found[prefix] = compact[-1] if compact else ""
    return found


def _coarse_equal(a: str, b: str) -> bool:
    a = _norm_stage(a)
    b = _norm_stage(b)
    if not a or not b:
        return False
    return a[:2] == b[:2]


def concept_set(text: str) -> set[str]:
    text_l = _safe_text(text).lower()
    concepts = set()
    for concept, surfaces in TREATMENT_CONCEPTS.items():
        for surface in surfaces:
            if re.search(rf"(?<![a-z0-9]){re.escape(surface.lower())}(?![a-z0-9])", text_l):
                concepts.add(concept)
                break
    return concepts


def treatment_precision(gt_text: str, pred_text: str) -> float:
    gt = concept_set(gt_text)
    pred = concept_set(pred_text)
    if not pred:
        return 1.0 if not gt else 0.0
    return len(gt & pred) / len(pred)


def treatment_recall(gt_text: str, pred_text: str) -> float:
    gt = concept_set(gt_text)
    pred = concept_set(pred_text)
    if not gt:
        return 1.0
    return len(gt & pred) / len(gt)


def token_f1(reference: str, candidate: str) -> float:
    ref_tokens = re.findall(r"[a-z0-9]+", _safe_text(reference).lower())
    cand_tokens = re.findall(r"[a-z0-9]+", _safe_text(candidate).lower())
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum((ref_counts & cand_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_prediction_file(pred_path: str, out_details: str | None = None) -> dict[str, Any]:
    rows = load_jsonl(pred_path)
    # Prefer clean original-task rows. Fall back to the harm subset because it
    # uses the original unperturbed LungCURE text in this project.
    original_rows = [r for r in rows if r.get("subset") == "original"]
    rows = original_rows or [r for r in rows if r.get("subset") == "harm"]
    if not rows:
        raise ValueError(f"No subset=original or subset=harm rows found in {pred_path}")

    details = []
    for row in rows:
        gt_tnm = derive_gt_tnm(row.get("tnm_gt") or {})
        pred_tnm = parse_pred_tnm(row.get("prediction", ""))
        t_ok = _coarse_equal(gt_tnm["T"], pred_tnm["T"])
        n_ok = _coarse_equal(gt_tnm["N"], pred_tnm["N"])
        m_ok = _coarse_equal(gt_tnm["M"], pred_tnm["M"])
        pred_text = row.get("prediction", "")
        gt_text = row.get("cds_gt", "")
        prec = treatment_precision(gt_text, pred_text)
        rec = treatment_recall(gt_text, pred_text)
        f1 = token_f1(gt_text, pred_text)
        details.append(
            {
                "case_id": row.get("case_id"),
                "method": row.get("method", ""),
                "model": row.get("model", ""),
                "gt_T": gt_tnm["T"],
                "gt_N": gt_tnm["N"],
                "gt_M": gt_tnm["M"],
                "pred_T": pred_tnm["T"],
                "pred_N": pred_tnm["N"],
                "pred_M": pred_tnm["M"],
                "T_acc": float(t_ok),
                "N_acc": float(n_ok),
                "M_acc": float(m_ok),
                "TNM_acc": float(t_ok and n_ok and m_ok),
                "treatment_precision_proxy": prec,
                "treatment_recall_proxy": rec,
                "treatment_token_f1": f1,
            }
        )

    if out_details:
        save_jsonl(details, out_details)

    def mean(key: str) -> float:
        vals = [float(d[key]) for d in details]
        return sum(vals) / len(vals) if vals else 0.0

    first = details[0]
    return {
        "source_file": pred_path,
        "method": first.get("method", ""),
        "model": first.get("model", ""),
        "num_cases": len(details),
        "T_acc": mean("T_acc"),
        "N_acc": mean("N_acc"),
        "M_acc": mean("M_acc"),
        "TNM_acc": mean("TNM_acc"),
        "treatment_precision_proxy": mean("treatment_precision_proxy"),
        "treatment_recall_proxy": mean("treatment_recall_proxy"),
        "treatment_token_f1": mean("treatment_token_f1"),
        "RQ": "TBD",
        "BERT_F1": "TBD",
        "notes": "Rule/lexical proxy on clean original rows when available; RQ and exact BERT-F1 require official judge/BERTScore setup.",
    }


def evaluate_many(pred_files: list[str], out_csv: str, details_dir: str | None = None) -> list[dict[str, Any]]:
    summaries = []
    if details_dir:
        os.makedirs(details_dir, exist_ok=True)
    for pred in pred_files:
        detail_path = None
        if details_dir:
            detail_path = str(Path(details_dir) / (Path(pred).stem + "_original_metrics.jsonl"))
        summaries.append(evaluate_prediction_file(pred, detail_path))
    save_csv(summaries, out_csv)
    return summaries


def _write_tex(rows: list[dict[str, Any]], out_tex: str) -> None:
    def fmt(v: Any) -> str:
        if isinstance(v, str):
            return v
        try:
            if math.isnan(float(v)):
                return "TBD"
            return f"{float(v):.3f}"
        except Exception:
            return str(v)

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Model & TNM Acc $\uparrow$ & T Acc & N Acc & M Acc & Treat Prec. $\uparrow$ & Token F1 $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        method = row.get("method") or "direct"
        model = row.get("model") or "-"
        lines.append(
            f"{method} & {model} & {fmt(row['TNM_acc'])} & {fmt(row['T_acc'])} & {fmt(row['N_acc'])} & {fmt(row['M_acc'])} & {fmt(row['treatment_precision_proxy'])} & {fmt(row['treatment_token_f1'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(out_tex).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate original LungCURE-style metrics on unperturbed predictions.")
    parser.add_argument("--pred", nargs="+", required=True)
    parser.add_argument("--out", default="paper/generated/original_lungcure_metrics.csv")
    parser.add_argument("--details-dir", default="results_safe/original_metrics_details")
    parser.add_argument("--tex", default="paper/generated/original_lungcure_metrics_table.tex")
    args = parser.parse_args()
    rows = evaluate_many(args.pred, args.out, args.details_dir)
    _write_tex(rows, args.tex)
    print(f"Wrote {args.out} and {args.tex}")


if __name__ == "__main__":
    main()
