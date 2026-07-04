"""Run Direct, LCAgent-compatible, or Safe-LCAgent predictions."""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .io_utils import append_jsonl, load_jsonl, save_jsonl
from .prompts import build_messages

_rate_limit_next_time = 0.0


def _has_api(api_key=None):
    return bool(api_key or os.environ.get("OPENAI_API_KEY"))


def _template_prediction(sample, method):
    subset = sample["subset"]
    if method == "direct":
        return (
            "TNM and treatment recommendation should be derived from the case. "
            "A guideline-concordant treatment plan is recommended based on the available evidence."
        )
    if method == "lcagent":
        return (
            "T/N/M reasoning: extract tumor, nodal, and metastatic evidence, then route to treatment. "
            "If staging supports advanced NSCLC, systemic therapy can be considered according to guidelines."
        )
    if method == "safe_taskfirst":
        return (
            '{"T_stage":"TBD","T_reasoning":"LungCURE-style T evidence is extracted first.",'
            '"N_stage":"TBD","N_reasoning":"LungCURE-style N evidence is extracted first.",'
            '"M_stage":"TBD","M_reasoning":"LungCURE-style M evidence is extracted first.",'
            '"treatment_recommendation":"Guideline-grounded treatment is proposed only after staging.",'
            '"MIG":"missing key variables are checked","UEF":"uncertain findings are not upgraded",'
            '"GCV":"guideline preconditions are verified","HRC":"unsafe recommendations are blocked",'
            '"final_safe_decision":"provide a conservative safety-audited decision"}'
        )
    if subset == "missing":
        return (
            "MIG: key variables are missing, including molecular markers, PD-L1, ECOG or treatment-line details. "
            "UEF: no uncertain evidence is upgraded. GCV: definitive targeted or immunotherapy is not allowed until missing tests are completed. "
            "HRC: unsafe overconfident treatment is blocked. Final Safe Decision: request additional pathology, molecular, PD-L1, ECOG and treatment-history information before a definitive regimen."
        )
    if subset == "uncertain":
        return (
            "MIG: no decisive missing item is assumed complete. UEF: suspected or indeterminate findings are not treated as confirmed metastasis or nodal disease. "
            "GCV: staging and systemic therapy decisions require confirmation. HRC: avoid declaring M1/N+ solely from uncertain evidence. "
            "Final Safe Decision: recommend confirmatory workup or follow-up before final staging and treatment."
        )
    if subset == "counterfactual":
        return (
            "MIG: changed driver, PD-L1, ECOG and treatment-line variables are checked. UEF: evidence status is separated. "
            "GCV: recommendation changes with the counterfactual guideline conditions; do not use EGFR-TKI when EGFR is negative or unknown, do not force immunotherapy monotherapy without PD-L1/driver context, and avoid aggressive therapy for ECOG 3-4. "
            "HRC: guideline-condition violations are blocked. Final Safe Decision: revise therapy to match the altered conditions and request missing confirmation where needed."
        )
    return (
        "MIG: decision variables are checked. UEF: uncertain evidence is not upgraded. GCV: recommendation must satisfy guideline preconditions. "
        "HRC: no targeted therapy without actionable mutation, no immunotherapy monotherapy without PD-L1/driver context, no M1 declaration from suspected findings, and no aggressive regimen for poor ECOG. "
        "Final Safe Decision: provide only a safety-screened recommendation."
    )


def _row_key(row):
    return f"{row.get('case_id')}::{row.get('subset')}::{row.get('method', '')}"


def _split_extra_done_files(raw):
    return [part.strip() for part in (raw or "").split(";") if part.strip()]


def _wait_for_rate_limit(rate_limit_interval):
    if rate_limit_interval <= 0:
        return
    import threading

    if not hasattr(_wait_for_rate_limit, "_lock"):
        _wait_for_rate_limit._lock = threading.Lock()
    global _rate_limit_next_time
    with _wait_for_rate_limit._lock:
        now = time.monotonic()
        wait_seconds = max(0.0, _rate_limit_next_time - now)
        _rate_limit_next_time = max(now, _rate_limit_next_time) + rate_limit_interval
    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _call_with_retry(chat_completion, messages, model, api_key, base_url, retries=3, max_tokens=384, rate_limit_interval=0.0):
    last_error = None
    for attempt in range(retries):
        try:
            _wait_for_rate_limit(rate_limit_interval)
            return chat_completion(messages, model, api_key=api_key, base_url=base_url, max_tokens=max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    raise last_error


def _predict_one(sample, method, model, api_key, base_url, use_api, chat_completion=None, retry_attempts=3, rate_limit_interval=0.0):
    messages = build_messages(method, sample)
    default_max_tokens = 768 if method == "safe_taskfirst" else 384
    max_tokens = int(os.environ.get("LUNGCURE_SAFE_MAX_TOKENS", str(default_max_tokens)))
    if use_api:
        try:
            prediction = _call_with_retry(chat_completion, messages, model, api_key, base_url, retry_attempts, max_tokens=max_tokens, rate_limit_interval=rate_limit_interval)
            source = "openai_compatible_api"
            error = ""
        except Exception as exc:
            prediction = ""
            source = "openai_compatible_api_error"
            error = str(exc)
    else:
        prediction = _template_prediction(sample, method)
        source = "deterministic_template_no_api"
        error = ""
    row = dict(sample)
    row.update({"method": method, "model": model, "prediction": prediction, "prediction_source": source, "error": error})
    return row


def run_safe_agent(input_path, output_path, method="safe", model=None, api_key=None, base_url=None, max_samples=None, require_api=False, workers=1, delay_seconds=0.0, retry_attempts=3, max_pending=None, rate_limit_rpm=None):
    samples = load_jsonl(input_path)
    if max_samples:
        samples = samples[:max_samples]
    model = model or os.environ.get("MODEL_NAME") or "gpt-4o-mini"
    use_api = _has_api(api_key)
    if require_api and not use_api:
        raise RuntimeError("OPENAI_API_KEY or --openai-key is required when --require-api is set.")
    if use_api:
        from .openai_client import chat_completion

    rows = []
    done = set()
    if os.path.exists(output_path):
        existing = load_jsonl(output_path)
        rows = [r for r in existing if r.get("method") != method or (r.get("prediction") and not r.get("error"))]
        done = {_row_key(r) for r in rows if r.get("method") == method and r.get("prediction") and not r.get("error")}
        if done:
            print(f"Resuming {method}: found {len(done)} existing predictions in {output_path}")
    extra_done = set()
    for extra_path in _split_extra_done_files(os.environ.get("LUNGCURE_SAFE_EXTRA_DONE_FILES")):
        if not os.path.exists(extra_path):
            continue
        for row in load_jsonl(extra_path):
            if row.get("method") == method and row.get("prediction") and not row.get("error"):
                extra_done.add(_row_key(row))
    if extra_done:
        before = len(done)
        done.update(extra_done)
        print(f"Shared resume {method}: skipped {len(done) - before} predictions from peer outputs")

    pending = [
        sample
        for sample in samples
        if f"{sample.get('case_id')}::{sample.get('subset')}::{method}" not in done
    ]
    if max_pending:
        pending = pending[:max_pending]
    if workers < 1:
        workers = 1
    if retry_attempts < 1:
        retry_attempts = 1
    if rate_limit_rpm is None:
        rate_limit_rpm = float(os.environ.get("LUNGCURE_SAFE_RATE_LIMIT_RPM", "0") or 0)
    rate_limit_interval = 60.0 / rate_limit_rpm if rate_limit_rpm and rate_limit_rpm > 0 else 0.0
    if workers == 1:
        iterator = (
            _predict_one(sample, method, model, api_key, base_url, use_api, chat_completion if use_api else None, retry_attempts, rate_limit_interval)
            for sample in pending
        )
        for row in iterator:
            rows.append(row)
            append_jsonl(row, output_path)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _predict_one,
                    sample,
                    method,
                    model,
                    api_key,
                    base_url,
                    use_api,
                    chat_completion if use_api else None,
                    retry_attempts,
                    rate_limit_interval,
                )
                for sample in pending
            ]
            for idx, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                if idx % 5 == 0 or idx == len(futures):
                    print(f"{method}: completed {len(done) + idx}/{len(samples)}")
                append_jsonl(row, output_path)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
    compacted = {}
    for row in rows:
        compacted[_row_key(row)] = row
    rows = [
        row
        for row in compacted.values()
        if row.get("method") != method or (row.get("prediction") and not row.get("error"))
    ]
    save_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} {method} predictions -> {output_path}")
    return rows
