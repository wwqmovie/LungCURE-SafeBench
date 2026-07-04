"""Command line entry point for ``python -m lungcure_safe``."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="lungcure_safe")
    sub = parser.add_subparsers(dest="command")

    build_p = sub.add_parser("build", help="Build LungCURE-SafeBench JSONL")
    build_p.add_argument("--root", default=".")
    build_p.add_argument("--lang", default="English", choices=["English", "Chinese"])
    build_p.add_argument("--max-cases", type=int, default=None)
    build_p.add_argument("--out", default="data_safe/lungcure_safebench.jsonl")

    build_original_p = sub.add_parser("build-original", help="Build unperturbed LungCURE JSONL")
    build_original_p.add_argument("--root", default=".")
    build_original_p.add_argument("--lang", default="English", choices=["English", "Chinese"])
    build_original_p.add_argument("--max-cases", type=int, default=None)
    build_original_p.add_argument("--out", default="data_safe/lungcure_original_en.jsonl")

    agent_p = sub.add_parser("agent", help="Run a baseline or Safe-LCAgent")
    agent_p.add_argument("--input", required=True)
    agent_p.add_argument("--output", required=True)
    agent_p.add_argument(
        "--method",
        default="safe",
        choices=[
            "direct",
            "lcagent",
            "safe",
            "safe_taskfirst",
            "long_safety_prompt",
            "safe_no_mig",
            "safe_no_uef",
            "safe_no_gcv",
            "safe_no_hrc",
        ],
    )
    agent_p.add_argument("--model", default=None)
    agent_p.add_argument("--openai-key", default=None)
    agent_p.add_argument("--openai-base", default=None)
    agent_p.add_argument("--max-samples", type=int, default=None)
    agent_p.add_argument("--require-api", action="store_true")
    agent_p.add_argument("--workers", type=int, default=1)
    agent_p.add_argument("--delay-seconds", type=float, default=0.0)
    agent_p.add_argument("--retry-attempts", type=int, default=3)
    agent_p.add_argument("--max-pending", type=int, default=None)
    agent_p.add_argument("--rate-limit-rpm", type=float, default=None)

    rule_p = sub.add_parser("rule", help="Rule-based safety evaluation")
    rule_p.add_argument("--pred", required=True)
    rule_p.add_argument("--out", default="results_safe/rule_summary.csv")
    rule_p.add_argument("--details", default=None)

    eval_p = sub.add_parser("eval", help="Optional LLM-as-a-judge evaluation")
    eval_p.add_argument("--safe-set", required=True)
    eval_p.add_argument("--pred", required=True)
    eval_p.add_argument("--output", default="results_safe/eval.jsonl")
    eval_p.add_argument("--summary", default="results_safe/safety_summary.csv")
    eval_p.add_argument("--judge-model", default=None)
    eval_p.add_argument("--judge-key", default=None)
    eval_p.add_argument("--judge-base", default=None)
    eval_p.add_argument("--max-samples", type=int, default=None)

    report_p = sub.add_parser("report", help="Create paper-ready tables")
    report_p.add_argument("--safe-set", required=True)
    report_p.add_argument("--summaries", nargs="+", required=True)
    report_p.add_argument("--out-dir", default="paper/generated")

    original_p = sub.add_parser("original", help="Evaluate original LungCURE-style metrics on unperturbed predictions")
    original_p.add_argument("--pred", nargs="+", required=True)
    original_p.add_argument("--out", default="paper/generated/original_lungcure_metrics.csv")
    original_p.add_argument("--details-dir", default="results_safe/original_metrics_details")
    original_p.add_argument("--tex", default="paper/generated/original_lungcure_metrics_table.tex")

    args = parser.parse_args()
    if args.command == "build":
        from .build_safe_sets import build_all

        build_all(args.root, args.lang, args.max_cases, args.out)
    elif args.command == "build-original":
        from .build_safe_sets import build_original

        build_original(args.root, args.lang, args.max_cases, args.out)
    elif args.command == "agent":
        from .run_safe_agent import run_safe_agent

        run_safe_agent(
            args.input,
            args.output,
            args.method,
            args.model,
            args.openai_key,
            args.openai_base,
            args.max_samples,
            args.require_api,
            args.workers,
            args.delay_seconds,
            args.retry_attempts,
            args.max_pending,
            args.rate_limit_rpm,
        )
    elif args.command == "rule":
        from .rule_safety_metrics import run_rule_eval

        run_rule_eval(args.pred, args.out, args.details)
    elif args.command == "eval":
        from .eval_safe import run_eval

        run_eval(
            args.safe_set,
            args.pred,
            args.output,
            args.summary,
            args.judge_model,
            args.judge_key,
            args.judge_base,
            args.max_samples,
        )
    elif args.command == "report":
        from .report import build_report

        build_report(args.safe_set, args.summaries, args.out_dir)
    elif args.command == "original":
        from .original_metrics import _write_tex, evaluate_many

        rows = evaluate_many(args.pred, args.out, args.details_dir)
        _write_tex(rows, args.tex)
        print(f"Wrote {args.out} and {args.tex}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
