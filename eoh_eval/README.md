# EoH gap evaluation

This directory evaluates the 18 converted EoH tasks (six each for online bin
packing, traveling salesman, and flow shop scheduling). Questions and reference
objectives are in `../test_data/eoh_eval.jsonl`; the 30 input CSVs are in
`../test_data/eoh_instances/<instance>/data/`. No extra data restoration is needed.

Use Python 3.11 with a CUDA-compatible PyTorch/vLLM installation for inference.
The judge needs NumPy and tqdm; generated programs may also need pandas, SciPy,
and Gurobi (with a working license):

```bash
python -m pip install 'numpy>=1.24,<2' tqdm pandas scipy gurobipy
```

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_NAME="$PWD/models/eval-model" \
  OUTPUT_PATH=outputs/eoh TENSOR_PARALLEL_SIZE=1 \
  SEED=1 NUM_SAMPLES=1 TEMPERATURE=0.5 bash eoh_eval/inference.sh

INPUT_PATH=outputs/eoh bash eoh_eval/judge.sh
```

Inference reuses `eval/inference.py` and its default `sdrl_prompt`, selecting only
`eoh_eval.jsonl`. For multiple samples, set `NUM_SAMPLES=8 TEMPERATURE=1.0` and use
a separate output directory. Seed 1 is an example, not a historical experiment
seed. Judge defaults: 3600 seconds per sample, two workers, one Gurobi thread, and
a 1000 MB address-space limit where supported. Override with `EXEC_TIMEOUT`,
`EOH_WORKERS`, `ORSTAR_GUROBI_THREADS`, and `ORSTAR_EXEC_MAX_MEM_MB`.

The standalone gap judge runs each generated program from its own instance
folder. It accepts `Result: <number>` or the shared evaluator's objective output
marker. `instance.json` and `original.json`, which contain reference answers,
are excluded from those folders. Set `EOH_INSTANCE_DIR` to relocate the CSV root;
use `--benchmark path/to/eoh_eval.jsonl` to relocate the reference file. Reference
objectives are read from the benchmark, not copied from response metadata.

## Metric and output

All bundled tasks are minimization problems with positive reference objectives:

```text
gap_pct = 100 * (computed_objective - reference_objective) / abs(reference_objective)
```

Lower is better. An objective of 110 against a reference of 100 gives 10%; 100
gives 0%; 95 gives -5%. Negative gaps are retained, not clipped or made absolute.
They indicate a reported objective below the supplied reference and warrant
checking the solution and reference. No correctness labels or Pass@K are used.
No variable-type retry is applied. A solution need not be proven optimal to
receive a gap. Gurobi models created through `gurobipy.Model` receive a time
limit based on the remaining execution budget, reserving up to 10 seconds to
return their incumbent. Existing shorter solver limits are preserved. An
incumbent is read only when `SolCount > 0`; no solution is invented on failure.

Programs run with unbuffered output. If a heuristic reaches the process timeout,
the lowest positive finite objective it has already printed as `Result: <value>`
is retained; programs should print a current feasible solution as they improve
it. Without a complete-program output, the most recent Gurobi incumbent is used.
Timeout results retained this way have `termination="timeout"`,
`used_partial_solution=true`, and `optimality="not_proven"`. They contribute to
gap statistics. Normal results use `optimality="not_verified"`; matching the
reference is not treated as proof of optimality.

Each input response file gets three outputs under `eoh_judged/`:

- `*_gap.jsonl`: per-sample objective, reference, gap, execution status and logs.
- `*_gap_summary.json`: per-instance, per-family and overall statistics.
- `*_gap_summary.txt`: compact human-readable statistics.

For multiple samples, first average gaps within an instance, then give each
instance equal weight in family and overall means. Best valid sample gaps are
also listed per instance. Timeout without a reported solution, execution failure, missing code, and
non-finite or non-positive objectives receive a null gap, never zero. The main
`mean_gap_pct` is null if an instance is missing or any sample fails; conditional
means over valid samples/instances and their coverage are shown separately.
Do not report a conditional mean without its failure and coverage counts.
Each response file is summarized separately, so different models are not mixed.
Existing outputs require `--overwrite` to rerun.

This evaluates supplied objectives on the converted tasks; it does not reproduce
the original EoH algorithm-evolution loop. The judge does not independently
verify decision-variable feasibility or online arrival-order compliance. The
CSV-only working directory is not a filesystem security sandbox.
