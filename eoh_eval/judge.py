#!/usr/bin/env python3
"""Execute the converted EoH tasks and report signed reference gaps in percent."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("evaluation_helpers", ROOT / "eval/judge.py")
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)

GUROBI_LIMIT_PREFIX = r'''
import time as _eoh_time
import os as _eoh_os
import gurobipy as _eoh_gp
_eoh_budget = float(_eoh_os.environ["EOH_EXEC_TIMEOUT"])
_eoh_deadline = _eoh_time.monotonic() + _eoh_budget - min(10.0, _eoh_budget * 0.1)
_eoh_original_model = _eoh_gp.Model
class _EohLimitedModel(_eoh_original_model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setParam("Threads", int(_eoh_os.environ.get("ORSTAR_GUROBI_THREADS", "1")))
        self.setParam("OutputFlag", 0)
    def optimize(self, *args, **kwargs):
        remaining = max(0.0, _eoh_deadline - _eoh_time.monotonic())
        self.setParam("TimeLimit", min(float(self.Params.TimeLimit), remaining))
        result = super().optimize(*args, **kwargs)
        if self.SolCount > 0:
            print("EOH incumbent:", self.ObjVal, flush=True)
        return result
_eoh_gp.Model = _EohLimitedModel
'''


def reported_objective(stdout, *, interrupted=False):
    """Read explicit objective outputs, including the last Gurobi incumbent."""
    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else (stdout or "")
    values, incumbents = [], []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(Result:|Just print the best solution:|EOH incumbent:)\s*(" + helpers.NUMBER_PATTERN + r")\s*", line, re.IGNORECASE)
        if match:
            value = float(match.group(2))
            if math.isfinite(value) and value > 0:
                (incumbents if match.group(1).lower() == "eoh incumbent:" else values).append(value)
    # Complete program outputs take precedence over intermediate solver models.
    if values:
        return min(values) if interrupted else values[-1]
    return incumbents[-1] if incumbents else None


def relative_gap_percent(objective, reference):
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("EoH reference objectives must be finite and positive")
    if not math.isfinite(objective) or objective <= 0:
        raise ValueError("EoH computed objectives must be finite and positive")
    gap = 100.0 * (objective - reference) / abs(reference)
    if not math.isfinite(gap):
        raise ValueError("Non-finite relative gap")
    return gap


def instance_cwd(item, data_root):
    root = Path(data_root).resolve()
    name = item.get("instance_dir")
    if not isinstance(name, str) or not name:
        raise ValueError("Missing EoH instance_dir")
    cwd = (root / name).resolve()
    if cwd == root or not cwd.is_relative_to(root):
        raise ValueError("Instance path escapes the EoH data directory")
    if not (cwd / "data").is_dir():
        raise FileNotFoundError(f"Instance CSV directory not found: {cwd / 'data'}")
    if any((cwd / name).exists() for name in ("instance.json", "original.json")):
        raise ValueError("Remove answer-bearing JSON files from the instance directory")
    return cwd


def evaluate(row, reference_row, data_root, timeout):
    reference = float(reference_row["en_answer"])
    result = {
        "instance_dir": reference_row["instance_dir"],
        "family": reference_row["eoh_problem_family"],
        "reference_objective": reference,
        "objective": None, "gap_pct": None,
        "status": "execution_error", "stdout": "", "stderr": "",
        "termination": "completed", "optimality": "not_verified",
    }
    try:
        cwd = instance_cwd(reference_row, data_root)
        response = row.get("response")
        code = helpers.extract_code_block(response, "python_algo") if isinstance(response, str) else None
        if not code:
            result["status"] = "missing_code"
            return result
        if helpers.is_gurobi_code(code):
            code = GUROBI_LIMIT_PREFIX + "\n" + code
            env = helpers.build_execution_env(use_solver_safe_env=True)
        else:
            env = helpers.build_execution_env()
        env["EOH_EXEC_TIMEOUT"] = str(timeout)
        completed = subprocess.run(
            [sys.executable, "-u", "-c", helpers.EXECUTION_WRAPPER], input=code,
            capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout,
        )
        result.update(returncode=completed.returncode,
                      stdout=helpers.truncate_exec_output(completed.stdout),
                      stderr=helpers.truncate_exec_output(completed.stderr))
        if completed.returncode:
            return result
        # Inspect complete output before truncating diagnostic logs.
        value = reported_objective(completed.stdout)
        if value is None or not math.isfinite(value) or value <= 0:
            result["status"] = "invalid_objective"
            return result
        result.update(objective=value, gap_pct=relative_gap_percent(value, reference), status="ok")
    except subprocess.TimeoutExpired as exc:
        result.update(status="timeout", termination="timeout", stdout=helpers.truncate_exec_output(exc.stdout),
                      stderr=helpers.truncate_exec_output(exc.stderr))
        value = reported_objective(exc.stdout, interrupted=True)
        if value is not None:
            result.update(status="ok", objective=value,
                          gap_pct=relative_gap_percent(value, reference),
                          optimality="not_proven", used_partial_solution=True)
    except Exception as exc:
        result["stderr"] = f"{type(exc).__name__}: {exc}"
    return result


def summarize(results, references):
    instances = []
    for ref in references:
        samples = [r for r in results if r["instance_dir"] == ref["instance_dir"]]
        gaps = [r["gap_pct"] for r in samples if r["status"] == "ok"]
        mean = statistics.mean(gaps) if gaps else None
        instances.append({
            "instance_dir": ref["instance_dir"], "family": ref["eoh_problem_family"],
            "samples": len(samples), "valid_samples": len(gaps),
            "failed_samples": len(samples) - len(gaps),
            "mean_gap_pct": mean if gaps and len(gaps) == len(samples) else None,
            "mean_gap_pct_valid_samples": mean,
            "best_gap_pct_valid_samples": min(gaps) if gaps else None,
        })

    def aggregate(group):
        valid = [r["mean_gap_pct_valid_samples"] for r in group if r["valid_samples"]]
        complete = bool(group) and all(r["mean_gap_pct"] is not None for r in group)
        return {
            "instances": len(group), "instances_with_valid_samples": len(valid),
            "missing_instances": sum(r["samples"] == 0 for r in group),
            "samples": sum(r["samples"] for r in group),
            "valid_samples": sum(r["valid_samples"] for r in group),
            "failed_samples": sum(r["failed_samples"] for r in group),
            "mean_gap_pct": statistics.mean(valid) if complete else None,
            "mean_gap_pct_valid_instances": statistics.mean(valid) if valid else None,
        }
    return {
        "metric": "signed_relative_gap_percent",
        "formula": "100 * (objective - reference) / abs(reference)",
        "direction": "minimize", "lower_is_better": True,
        "aggregation": "mean across samples within each instance, then equal-weight mean across instances",
        "failure_policy": "Null gaps on failure; main mean is null if any instance is missing or any sample fails. Conditional means and coverage are reported separately.",
        "feasibility_verified": False,
        "status_counts": dict(Counter(r["status"] for r in results)),
        "timeout_with_solution": sum(r.get("used_partial_solution", False) for r in results),
        "overall": aggregate(instances),
        "by_family": {family: aggregate([r for r in instances if r["family"] == family])
                      for family in sorted({r["family"] for r in instances})},
        "per_instance": instances,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "test_data/eoh_eval.jsonl")
    parser.add_argument("--instance_dir", type=Path, default=Path(os.environ.get("EOH_INSTANCE_DIR", ROOT / "test_data/eoh_instances")))
    parser.add_argument("--exec_timeout", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.exec_timeout <= 0 or args.workers <= 0:
        parser.error("timeout and workers must be positive")
    references = helpers.read_jsonl(args.benchmark)
    by_name = {r["instance_dir"]: r for r in references}
    if not references or len(by_name) != len(references):
        raise ValueError("Benchmark must contain unique instances")
    for ref in references:
        relative_gap_percent(float(ref["en_answer"]), float(ref["en_answer"]))
        instance_cwd(ref, args.instance_dir)
    source = args.input_path.resolve()
    files = [source] if source.is_file() and source.name.endswith("_response.jsonl") else sorted(source.rglob("*_response.jsonl")) if source.is_dir() else []
    if not files:
        raise ValueError("Provide an EoH response JSONL or a directory containing *_response.jsonl")
    output = args.output_path or (source.parent if source.is_file() else source) / "eoh_judged"
    # Preflight every input before executing or writing any results.
    jobs = []
    for path in files:
        rows = helpers.read_jsonl(path)
        if not rows or any(r.get("instance_dir") not in by_name or
                           not (r.get("benchmark") == "EoH" or r.get("data_source") == "EoH") for r in rows):
            raise ValueError(f"Expected only EoH response rows in {path.name}; select a dedicated EoH run directory")
        stem = path.name.removesuffix("_response.jsonl")
        folder = output / path.relative_to(source).parent if source.is_dir() else output
        targets = [folder / f"{stem}_gap.jsonl", folder / f"{stem}_gap_summary.json", folder / f"{stem}_gap_summary.txt"]
        if not args.overwrite and any(p.exists() for p in targets):
            raise FileExistsError(f"Results already exist for {path.name}; use --overwrite to rerun")
        jobs.append((path, rows, targets))
    for path, rows, targets in jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda row: evaluate(row, by_name[row["instance_dir"]], args.instance_dir, args.exec_timeout), rows))
        for index, result in enumerate(results):
            result.update(response_file=path.name, response_row_index=index,
                          sample_index=(rows[index].get("_eval3") or {}).get("sample_index"))
        summary = summarize(results, references)
        summary.update(response_file=path.name, exec_timeout=args.exec_timeout,
                       workers=args.workers, variable_type_retry=False)
        targets[0].parent.mkdir(parents=True, exist_ok=True)
        helpers.write_jsonl_atomic(targets[0], results)
        targets[1].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        lines = ["Signed relative gap (%); lower is better."]
        for label, stats in [("overall", summary["overall"]), *summary["by_family"].items()]:
            lines.append(f"{label}: mean gap={stats['mean_gap_pct']}; conditional mean={stats['mean_gap_pct_valid_instances']}; valid samples={stats['valid_samples']}/{stats['samples']}; missing instances={stats['missing_instances']}")
        targets[2].write_text("\n".join(lines) + "\n")
        print(f"{path.name}: " + lines[1])
        print(f"Saved: {targets[1]}")


if __name__ == "__main__":
    main()
