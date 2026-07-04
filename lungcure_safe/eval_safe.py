"""Optional LLM-as-a-judge safety evaluation."""
import json
import os

from .io_utils import load_jsonl, save_csv, save_jsonl
from .prompts import build_judge_messages
from .rule_safety_metrics import score_prediction


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return {"score": 0, "safe": False, "failure_type": "judge_parse_error", "rationale": text[:500]}


def run_eval(safe_set, pred, output, summary, judge_model=None, judge_key=None, judge_base=None, max_samples=None):
    samples = {r["case_id"] + "::" + r["subset"]: r for r in load_jsonl(safe_set)}
    preds = load_jsonl(pred)
    if max_samples:
        preds = preds[:max_samples]
    judge_model = judge_model or os.environ.get("MODEL_NAME") or "gpt-4o-mini"
    use_api = bool(judge_key or os.environ.get("OPENAI_API_KEY"))
    if use_api:
        from .openai_client import chat_completion

    rows = []
    for p in preds:
        key = p["case_id"] + "::" + p["subset"]
        sample = samples.get(key, p)
        if use_api:
            raw = chat_completion(build_judge_messages(sample, p), judge_model, api_key=judge_key, base_url=judge_base)
            judged = _parse_json(raw)
            source = "openai_compatible_api"
        else:
            rule = score_prediction(p)
            judged = {
                "score": rule["safe"],
                "safe": bool(rule["safe"]),
                "failure_type": rule["failure_type"],
                "rationale": "Fallback rule judge because no API key was configured.",
            }
            source = "rule_fallback_no_api"
        row = dict(p)
        row.update({"judge_model": judge_model, "judge_source": source, "judge": judged, "judge_score": float(judged.get("score", 0))})
        rows.append(row)
    save_jsonl(rows, output)
    avg = sum(r["judge_score"] for r in rows) / len(rows) if rows else 0.0
    save_csv([{"method": preds[0].get("method", "unknown") if preds else "unknown", "num_samples": len(rows), "judge_safety_score": avg}], summary)
    print(f"Wrote judge details -> {output}; summary -> {summary}")
    return rows
