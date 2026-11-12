# FrontierOR evaluation

This directory runs the one-shot protocol of
[FrontierOR](https://github.com/Minw913/FrontierOR) (180 large-scale optimization
tasks; [paper](https://arxiv.org/abs/2605.25246)) against a self-hosted model served
through an OpenAI-compatible endpoint such as vLLM. FrontierOR's code and data are
not copied here; this directory only adds a small patch, a model registry, the
SDRL prompt used for FrontierOR, and a run script.

| File | Purpose |
| --- | --- |
| `frontieror_selfhosted.patch` | Patch for FrontierOR commit `bc6b598` (see below) |
| `oneshot_selfhosted.yaml` | Model registry for self-hosted endpoints |
| `sdrl_frontieror.system.txt` | SDRL strategy-routing system prompt with a runtime-requirements section |
| `run_frontieror.sh` | Runs `one_shot_eval.py` for one model |

## Setup

```bash
git clone https://github.com/Minw913/FrontierOR.git && cd FrontierOR
git checkout bc6b598
git apply /path/to/LLMMetaSolver/frontieror_eval/frontieror_selfhosted.patch
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
hf download frontieror/FrontierOR --repo-type dataset --local-dir frontier-or
```

A Gurobi license is required (`GRB_LICENSE_FILE`): FrontierOR's checkers and most
generated programs use `gurobipy`.

The patch only adds opt-in environment variables; with none of them set, FrontierOR
behaves as upstream:

- `FRONTIER_OR_LLM_BASE_URL`, `FRONTIER_OR_LLM_API_KEY`, `FRONTIER_OR_LLM_TIMEOUT`:
  send chat completions to any OpenAI-compatible endpoint instead of OpenRouter.
- `FRONTIER_OR_ONESHOT_CONFIG`: use another model registry instead of `configs/oneshot.yaml`.
- `FRONTIER_OR_SYSTEM_PROMPT_FILE`, `FRONTIER_OR_USER_SUFFIX`: prepend a system prompt
  and append a suffix (e.g. `/no_think`) to user messages.
- `FRONTIER_OR_CANDIDATE_TIME_RESERVE`: tell the generated program
  `time_limit - reserve` seconds while the hard kill stays at `time_limit`.
- `FRONTIER_OR_SAVE_RESPONSES` (default on): save every raw LLM exchange to
  `responses.jsonl` next to the generated code.
- `repetition_penalty`, `presence_penalty`, `frequency_penalty` are accepted in the
  registry's `model_generation`; vLLM's `prompt_tokens_details: null` is handled.

## Run

Serve the model (one model per endpoint; the served name must match the registry):

```bash
vllm serve tokentideAI/SDRL-Qwen3-32B --served-model-name sdrl-qwen3-32b \
  --max-model-len 32768 --port 8000
```

Then, on a CPU machine that can reach the endpoint:

```bash
FRONTIEROR_DIR=/path/to/FrontierOR GRB_LICENSE_FILE=/path/to/gurobi.lic \
  MODEL=sdrl-qwen3-32b BASE_URL=http://localhost:8000/v1 PROMPT_MODE=sdrl \
  bash frontieror_eval/run_frontieror.sh
```

- `PROMPT_MODE=sdrl` (default): `sdrl_frontieror.system.txt` as the system prompt,
  FrontierOR's own user prompt, and `/no_think` appended. `PROMPT_MODE=native`:
  the unmodified FrontierOR prompt.
- Defaults are the paper protocol: 180 tasks, a tiny-instance gate followed by
  `large_1`..`large_5`, 3600 s per run, up to 5 debug retries, and one CPU core per
  candidate program. Override with `INSTANCES`, `TIME_LIMIT`, `DEBUG_RETRIES`,
  `PAPER_WORKERS`, `INSTANCE_WORKERS`.
- `RESERVE=120` reproduces our time-reserve variant (program told 3480 s, killed at
  3600 s). It is not the paper protocol; report it as such.
- Outputs go to `OUT` (default `$FRONTIEROR_DIR/eval/<model>_<prompt>_reserve<R>/`):
  `eval_results_<model>.csv`, per-task code and logs under `eval_tasks/`, and `eval.log`.
  Large instances can produce multi-GB solution/log files; keep several hundred GB free.
- The run is resumable: rerun with the same `OUT` and finished pairs are skipped.
  To refill rows missing from the CSV for papers whose code already exists, run with
  `SKIP_EXISTING= EXTRA_ARGS='--paper_id <paper> ...'`.

## Metrics

Use FrontierOR's own aggregation script, which reports execution rate, feasibility,
solution quality and QTE on the full set (180) and the Hard set (50) in the format of
the paper's Table 2:

```bash
cd /path/to/FrontierOR
python scripts/compute_benchmark_main_metrics.py \
  --model-csv eval/sdrl-qwen3-32b_sdrl_reserve0/eval_results_sdrl-qwen3-32b.csv \
  --model-name SDRL-Qwen3-32B \
  --hard-set frontier-or/metadata/hardset/hardset.json --data-dir frontier-or \
  --output-dir eval/metrics
```

The table's execution rate counts programs that run after debug retries; the
first-attempt rate is reported separately as `first_execution_rate`.
