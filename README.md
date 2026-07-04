# LungCURE-SafeBench

This repository contains the reproducibility package for **LungCURE-SafeBench: A Disclosed-Audit Benchmark for Safety--Utility Evaluation in Guideline-Grounded Lung Cancer Clinical Decision Support**.

The package includes the released SafeBench JSONL files, the original-task JSONL file used for clean utility scoring, model-prompt evaluation code, rule-based safety metrics, aggregate result tables, figure-generation scripts, and the LaTeX source of the submitted manuscript.

## Repository Layout

```text
lungcure_safe/      Benchmark builders, prompts, API runner, and scoring code
data_safe/          Released SafeBench and original-task JSONL files
scripts/            One-command API evaluation and result reproduction scripts
paper/main.tex      Manuscript source
paper/figures/      Final manuscript figures
paper/generated/    Aggregate CSV/JSON files used by the manuscript
configs/            Example environment configuration
```

Large raw API prediction logs are not included in the repository. They can be regenerated with the provided API runner. The included aggregate files are sufficient to reproduce the manuscript tables, figures, and numerical claims.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce Main Numerical Claims

```bash
python scripts/reproduce_claims.py
```

This reads `paper/generated/*.csv` and writes `paper/generated/reproduced_claims.json`. It verifies the released row counts, number of evaluated configurations, mean permissive safety, mean SCSS, best safety-utility row, best clean-utility row, and Task-first mean deltas.

## Run a Smoke Test Without an API Key

The runner falls back to deterministic template outputs when no API key is supplied, which is useful for checking the pipeline.

```bash
python -m lungcure_safe agent \
  --method safe_taskfirst \
  --input data_safe/lungcure_safebench_smoke.jsonl \
  --output result/smoke_safe_taskfirst.jsonl \
  --max-samples 10

python -m lungcure_safe rule \
  --pred result/smoke_safe_taskfirst.jsonl \
  --out result/smoke_safe_taskfirst_summary.csv
```

## Run API-Backed Evaluation

Set an OpenAI-compatible endpoint and model name:

```bash
export OPENAI_API_KEY=replace_with_your_api_key
export OPENAI_BASE_URL=https://api.example.com/v1
export MODEL_NAME=your-model-name
```

Then run one prompt design on a bounded sample:

```bash
python scripts/one_command_api_eval.py \
  --url "$OPENAI_BASE_URL" \
  --model "$MODEL_NAME" \
  --prompt safe_taskfirst \
  --task both \
  --n 100 \
  --workers 1
```

The sample count must be between 10 and 1000. Outputs are written under `result/`, including JSONL records, per-sample text bundles, and metric summaries.

For full reproduction, run the five main prompt designs on the released full files:

```bash
python -m lungcure_safe agent --require-api --method direct --input data_safe/lungcure_safebench_en_full.jsonl --output results_safe/direct_MODEL_safebench.jsonl --model "$MODEL_NAME"
python -m lungcure_safe agent --require-api --method lcagent --input data_safe/lungcure_safebench_en_full.jsonl --output results_safe/lcagent_MODEL_safebench.jsonl --model "$MODEL_NAME"
python -m lungcure_safe agent --require-api --method safe --input data_safe/lungcure_safebench_en_full.jsonl --output results_safe/safe_MODEL_safebench.jsonl --model "$MODEL_NAME"
python -m lungcure_safe agent --require-api --method safe_taskfirst --input data_safe/lungcure_safebench_en_full.jsonl --output results_safe/safe_taskfirst_MODEL_safebench.jsonl --model "$MODEL_NAME"
python -m lungcure_safe agent --require-api --method long_safety_prompt --input data_safe/lungcure_safebench_en_full.jsonl --output results_safe/long_safety_prompt_MODEL_safebench.jsonl --model "$MODEL_NAME"
```

Evaluate a prediction file:

```bash
python -m lungcure_safe rule \
  --pred results_safe/safe_taskfirst_MODEL_safebench.jsonl \
  --out results_safe/safe_taskfirst_MODEL_safety_summary.csv \
  --details results_safe/safe_taskfirst_MODEL_safety_details.jsonl
```

Clean original-task utility can be evaluated on outputs generated from `data_safe/lungcure_original_en.jsonl`:

```bash
python -m lungcure_safe original \
  --pred results_safe/original_safe_taskfirst_MODEL.jsonl \
  --out paper/generated/original_MODEL_metrics.csv
```

## Rebuild the Manuscript PDF

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Notes

The benchmark is intended for offline evaluation of text-only clinical decision support behavior. It is not intended for patient-level clinical decision making.
