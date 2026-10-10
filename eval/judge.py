#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

MAX_EXEC_OUTPUT_CHARS = 20000
DEFAULT_EXEC_TIMEOUT = 300
DEFAULT_MIPLIB_NL_EXEC_TIMEOUT = 300
NUMBER_PATTERN = r'[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf(?:inity)?)'
INFEASIBLE_ANSWERS = {"infeasible", "impossible", "no feasible solution", "no solution"}
UNKNOWN_ANSWERS = {"no best solution"}
EXECUTION_WRAPPER = r"""
import os
import sys

source = sys.stdin.read()
limit_raw = os.environ.get("ORSTAR_EXEC_MAX_MEM_MB")
if limit_raw:
    try:
        limit_mb = int(limit_raw)
    except ValueError:
        limit_mb = 0
    if limit_mb > 0:
        try:
            import resource
            limit_bytes = limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            pass

exec(compile(source, "<stdin>", "exec"), {"__name__": "__main__"})
"""
SAFE_GUROBI_PREFIX = r'''
try:
    import gurobipy as gp
    _ORSTAR_ORIG_MODEL = gp.Model

    def _orstar_safe_model(*args, **kwargs):
        model = _ORSTAR_ORIG_MODEL(*args, **kwargs)
        import os
        model.setParam("Threads", int(os.environ.get("ORSTAR_GUROBI_THREADS", "0")))
        model.setParam("OutputFlag", int(os.environ.get("ORSTAR_GUROBI_OUTPUT_FLAG", "0")))
        return model

    gp.Model = _orstar_safe_model
except Exception:
    pass
'''

DEFAULT_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
DEFAULT_LEGACY_MIPLIB_NL_DATASET_DIR = DEFAULT_TEST_DATA_DIR / "MIPLIB-NL" / "dataset"
DEFAULT_FIXED_MIPLIB_NL_DATASET_DIR = DEFAULT_TEST_DATA_DIR / "MIPLIB-NL-Fixed" / "dataset"
DEFAULT_MIPLIB_NL_DATASET_DIR = (
    DEFAULT_FIXED_MIPLIB_NL_DATASET_DIR
    if DEFAULT_FIXED_MIPLIB_NL_DATASET_DIR.is_dir()
    else DEFAULT_LEGACY_MIPLIB_NL_DATASET_DIR
)


def load_jsonl(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}")
                    print(f"Error details: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []
    return data


def load_data(filepath):
    if filepath.endswith(".jsonl"):
        return load_jsonl(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return load_jsonl(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    print(f"Unsupported data format in {filepath}: {type(data).__name__}")
    return []


def _indent_width(line: str) -> int:
    return len(re.match(r'^[ \t]*', line).group(0).expandtabs(4))


def _line_index_from_pos(text: str, pos: int) -> int:
    return text[:pos].count('\n')


def replace_gurobi_solve_tail(code: str, optimize_match) -> str:
    lines = code.splitlines()
    optimize_line_index = _line_index_from_pos(code, optimize_match.start())
    indent = optimize_match.group(1)
    model_name = optimize_match.group(2)
    optimize_indent_width = _indent_width(lines[optimize_line_index])
    block_end = len(lines)

    if optimize_indent_width > 0:
        for index in range(optimize_line_index + 1, len(lines)):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _indent_width(lines[index]) < optimize_indent_width:
                block_end = index
                break

    replacement = [
        f"{indent}{model_name}.optimize()",
        f'{indent}if getattr({model_name}, "SolCount", 0) > 0:',
        f'{indent}    print(f"Just print the best solution: {{{model_name}.ObjVal}}")',
        f"{indent}else:",
        f'{indent}    print("infeasible")',
    ]
    return "\n".join(lines[:optimize_line_index] + replacement + lines[block_end:])


def is_gurobi_code(code: str) -> bool:
    return bool(re.search(r'(^|\n)\s*(import\s+gurobipy|from\s+gurobipy\s+import)\b', code))


def build_execution_env(use_solver_safe_env=False):
    env = os.environ.copy()
    if not use_solver_safe_env:
        return env

    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env.pop("CUDA_VISIBLE_DEVICES", None)

    ld_paths = []
    for path in env.get("LD_LIBRARY_PATH", "").split(":"):
        lowered = path.lower()
        if "cuda" in lowered or "nvidia" in lowered:
            continue
        if path:
            ld_paths.append(path)
    if ld_paths:
        env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def parse_memory_limit_mb() -> int | None:
    raw_value = os.environ.get("ORSTAR_EXEC_MAX_MEM_MB")
    if raw_value is None or raw_value == "":
        return None
    try:
        limit_mb = int(raw_value)
    except ValueError:
        return None
    if limit_mb <= 0:
        return None
    return limit_mb


def prepare_execution_code(code):
    if is_gurobi_code(code):
        return SAFE_GUROBI_PREFIX + "\n" + code, build_execution_env(use_solver_safe_env=True)
    return code, build_execution_env(use_solver_safe_env=False)


def resolve_execution_cwd(item):
    if item.get("benchmark") != "MIPLIB-NL" or not item.get("instance_dir"):
        return None

    dataset_dir = Path(os.environ.get("ORSTAR_MIPLIB_NL_DATASET_DIR", DEFAULT_MIPLIB_NL_DATASET_DIR))
    instance_cwd = dataset_dir / item["instance_dir"]
    if instance_cwd.is_dir():
        return str(instance_cwd)
    return None


def resolve_execution_timeout(item, execution_timeout=None):
    if execution_timeout is not None:
        return int(execution_timeout)

    env_timeout = os.environ.get("ORSTAR_EXEC_TIMEOUT")
    if env_timeout:
        return int(env_timeout)

    if item.get("benchmark") == "MIPLIB-NL":
        return int(os.environ.get("ORSTAR_MIPLIB_NL_EXEC_TIMEOUT", DEFAULT_MIPLIB_NL_EXEC_TIMEOUT))

    return DEFAULT_EXEC_TIMEOUT


def insert_print(code: str, solver_name: str) -> str:
    model_pattern = r'^(\s*)(\w+)\.(optimize|solve)\(\)'
    model_matches = list(re.finditer(model_pattern, code, re.M))
    if not model_matches:
        return code

    if solver_name == "gurobi" or is_gurobi_code(code):
        optimize_matches = [match for match in model_matches if match.group(3) == "optimize"]
        if not optimize_matches:
            return code
        return replace_gurobi_solve_tail(code, optimize_matches[-1])

    if solver_name == "copt":
        model_match = next((match for match in model_matches if match.group(3) == "solve"), None)
        if model_match is None:
            return code
        indent = model_match.group(1)
        model_name = model_match.group(2)
        pattern = r'^(\s*)(' + model_name + r'\.solve\(\))'
        return re.sub(pattern, rf'\1\2\n{indent}print(f"Just print the best solution: {{{model_name}.ObjVal}}")', code, flags=re.M)

    return code


def extract_fenced_code(text: str) -> str:
    pattern = r'```\s*(?:python|py)\s*\n(.*?)(?:\n```|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern = r'```\s*\n(.*?)(?:\n```|$)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def extract_code_block(llm_output: str, solver_name) -> str:
    pattern = r'<python>(.*?)(?:</python>|$)'
    match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        if '```' in code:
            code = extract_fenced_code(code) or code
        code = insert_print(code, solver_name)
        return code

    code = extract_fenced_code(llm_output)
    if code:
        code = insert_print(code,solver_name)
        return code
    return None


def extract_obj(str_log):
    if 'Just print the best solution:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best solution:' in i)
        result = re.findall(NUMBER_PATTERN, item, re.IGNORECASE)
        return float(result[0]) if result else None

    if 'Result:' in str_log:
        for line in str_log.split('\n'):
            if 'Result:' in line:
                match = re.search(r'Result:\s*(' + NUMBER_PATTERN + r')', line, re.IGNORECASE)
                if match:
                    val_str = match.group(1).lower()
                    if 'inf' in val_str:
                        return float('inf')
                    return float(val_str)
    return None


def parse_expected_answer(value):
    text = str(value).strip()
    normalized = text.lower()
    if normalized in UNKNOWN_ANSWERS or "-9999" in normalized:
        return None, None
    if normalized in INFEASIBLE_ANSWERS:
        return None, "infeasible"
    try:
        return float(text), None
    except (TypeError, ValueError):
        return None, f"unparseable answer: {text}"


def output_indicates_infeasible(stdout, stderr=""):
    text = f"{stdout}\n{stderr}".lower()
    return any(
        marker in text
        for marker in (
            "infeasible",
            "impossible",
            "no feasible",
            "not feasible",
            "no solution",
            "no feasible solution",
        )
    )


def result_indicates_no_finite_solution(solver_result, stdout="", stderr=""):
    if output_indicates_infeasible(stdout, stderr):
        return True
    if solver_result is None:
        return True
    if isinstance(solver_result, (int, float, np.floating)) and not np.isfinite(solver_result):
        return True
    return False


def enforce_integer_variables(code):
    """Add ``vtype=GRB.INTEGER`` to addVar/addVars calls without a vtype.

    This intentionally mirrors the fallback used by SIRL's public evaluator.
    It is only used for the optional second execution attempt.
    """
    pattern = r'(\w+\s*=\s*\w+\.addVar[s]?)\(([\s\S]*?)(\)\n)'

    def replacer(match):
        variable_assignment = match.group(1)
        parameters = match.group(2).rstrip()
        closing = match.group(3)
        if re.search(r'\bvtype\s*=', parameters, re.IGNORECASE):
            return match.group(0)
        if parameters:
            if not parameters.endswith(','):
                parameters += ','
            parameters = f"{parameters} vtype=GRB.INTEGER"
        else:
            parameters = "vtype=GRB.INTEGER"
        return f"{variable_assignment}({parameters}{closing}"

    return re.sub(pattern, replacer, code, flags=re.MULTILINE)


def change_variable_types(code):
    """Return a SIRL-style alternative variable typing and its action name.

    INTEGER is changed globally to CONTINUOUS, CONTINUOUS is changed globally
    to INTEGER, and code with no explicit vtype receives INTEGER on addVar(s).
    BINARY-only code is left unchanged, matching the original SIRL behavior.
    """
    if re.search(r'\bvtype\s*=', code, re.IGNORECASE):
        if "INTEGER" in code:
            changed = code.replace("INTEGER", "CONTINUOUS")
            return (changed, "integer_to_continuous") if changed != code else (None, None)
        if "CONTINUOUS" in code:
            changed = code.replace("CONTINUOUS", "INTEGER")
            return (changed, "continuous_to_integer") if changed != code else (None, None)
        return None, None

    changed = enforce_integer_variables(code)
    if changed != code:
        return changed, "implicit_continuous_to_integer"
    return None, None


def variable_type_retry_reason(detail, expected_answer):
    """Return why an otherwise executable answer should receive one retry."""
    if expected_answer is None or detail.get("timeout") or detail.get("returncode") != 0:
        return None
    if output_indicates_infeasible(detail.get("stdout", ""), detail.get("stderr", "")):
        return "infeasible_output_for_numeric_answer"
    solver_result = detail.get("solver_result")
    if solver_result is not None and np.abs(solver_result - expected_answer) >= 0.01:
        return "numeric_mismatch"
    return None


def attempt_snapshot(detail):
    """Keep both attempts auditable without changing the legacy top-level keys."""
    return {
        key: detail.get(key)
        for key in (
            "error_type",
            "extracted_code",
            "returncode",
            "stdout",
            "stderr",
            "timeout",
            "solver_result",
        )
    }


def truncate_exec_output(value, max_chars=MAX_EXEC_OUTPUT_CHARS):
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value)
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return value[:max_chars] + f"\n...[truncated {omitted} chars]"


def check_result_detail(
    result_str,
    item,
    solver_name,
    execution_timeout=None,
    variable_type_retry=True,
):
    detail = {
        "error_type": None,
        "extracted_code": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timeout": False,
        "solver_result": None,
        "cwd": None,
        "execution_timeout": None,
        "expected_status": None,
        "answer_parse_error": None,
        "variable_type_retry": {
            "enabled": bool(variable_type_retry),
            "attempted": False,
            "reason": None,
            "transformation": None,
            "accepted": False,
        },
    }

    sub_answer, expected_status = parse_expected_answer(item.get('en_answer'))
    if expected_status and expected_status != "infeasible":
        detail["answer_parse_error"] = expected_status
        detail["error_type"] = 4
        return detail
    detail["expected_status"] = expected_status
    expected_no_finite_solution = sub_answer is None and expected_status in (None, "infeasible")
    code_snippet = extract_code_block(result_str, solver_name)
    detail["extracted_code"] = code_snippet
    if code_snippet is None:
        detail["error_type"] = 2
        return detail
    execution_code, execution_env = prepare_execution_code(code_snippet)
    execution_cwd = resolve_execution_cwd(item)
    timeout_seconds = resolve_execution_timeout(item, execution_timeout)
    detail["cwd"] = execution_cwd
    detail["execution_timeout"] = timeout_seconds
    detail["memory_limit_mb"] = parse_memory_limit_mb()
    try:
        result = subprocess.run(
            [sys.executable, '-c', EXECUTION_WRAPPER],
            input=execution_code,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=execution_env,
            cwd=execution_cwd,
        )
    except subprocess.TimeoutExpired as e:
        detail["timeout"] = True
        detail["stdout"] = truncate_exec_output(getattr(e, "stdout", None) or getattr(e, "output", None))
        detail["stderr"] = truncate_exec_output(getattr(e, "stderr", None))
        if expected_no_finite_solution:
            detail["error_type"] = 1
        else:
            detail["error_type"] = 0
        return detail
    except OSError as e:
        detail["returncode"] = None
        detail["stderr"] = truncate_exec_output(
            f"Failed to start execution subprocess: {type(e).__name__}: {e}"
        )
        detail["error_type"] = 3
        return detail
    detail["returncode"] = result.returncode
    detail["stdout"] = truncate_exec_output(result.stdout)
    detail["stderr"] = truncate_exec_output(result.stderr)
    if result.returncode != 0:
        detail["error_type"] = 3
        return detail
    solver_result = extract_obj(result.stdout)
    detail["solver_result"] = solver_result

    if expected_no_finite_solution:
        detail["error_type"] = int(result_indicates_no_finite_solution(solver_result, result.stdout, result.stderr))
    elif sub_answer is not None and solver_result is not None:
        detail["error_type"] = int(np.abs(solver_result-sub_answer)<0.01)
    elif sub_answer == solver_result:
        detail["error_type"] = 1
    elif output_indicates_infeasible(result.stdout, result.stderr):
        if sub_answer is None:
            detail["error_type"] = 1
        else:
            detail["error_type"] = 0
    else:
        detail["error_type"] = 4

    retry_reason = variable_type_retry_reason(detail, sub_answer) if variable_type_retry else None
    retry_code, retry_action = change_variable_types(code_snippet) if retry_reason else (None, None)
    if retry_code:
        retry_response = f"```python\n{retry_code}\n```"
        retry_detail = check_result_detail(
            retry_response,
            item,
            solver_name,
            execution_timeout,
            variable_type_retry=False,
        )
        retry_metadata = {
            "enabled": True,
            "attempted": True,
            "reason": retry_reason,
            "transformation": retry_action,
            "accepted": retry_detail.get("error_type") == 1,
            "first_attempt": attempt_snapshot(detail),
            "retry_attempt": attempt_snapshot(retry_detail),
        }
        if retry_metadata["accepted"]:
            detail = retry_detail
        detail["variable_type_retry"] = retry_metadata
    else:
        detail["variable_type_retry"] = {
            "enabled": bool(variable_type_retry),
            "attempted": False,
            "reason": retry_reason,
            "transformation": retry_action,
            "accepted": False,
        }
    return detail


def check_result(result_str, item,solver_name):
    return check_result_detail(result_str, item, solver_name)["error_type"]

"""Stage 2: execute saved ORStar responses and score them on a licensed dev machine."""

import argparse
import json
import math
import os
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
JUDGEMENT_PROTOCOL = "orstar-eval3-judgement-v2"
RESULT_KEY = {
    0: "wrong",
    1: "correct",
    2: "code formulation failed",
    3: "execution error",
    4: "other error",
}



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def response_files_from_manifest(path: Path) -> list[Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "orstar-eval3-response-v1":
        raise ValueError(f"Unsupported manifest protocol in {path}: {manifest.get('protocol')}")
    files = []
    for item in manifest.get("files", []):
        response_file = Path(item["response_file"])
        if not response_file.is_absolute():
            response_file = path.parent / response_file
        files.append(response_file)
    return files


def row_resume_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    metadata = row.get("_eval3") or {}
    dataset = metadata.get("dataset")
    item_index = metadata.get("item_index")
    sample_index = metadata.get("sample_index")
    if isinstance(dataset, str) and isinstance(item_index, int) and isinstance(sample_index, int):
        return dataset, item_index, sample_index
    if isinstance(dataset, str) and isinstance(item_index, int):
        return dataset, item_index
    return None


def normalize_judged_row_for_resume(row: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(row)
    for key in ("error_type", "error_type_name", "evaluation"):
        normalized.pop(key, None)
    metadata = normalized.get("_eval3")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        for key in ("judge_stage", "judge_protocol", "variable_type_retry", "judged_item_index"):
            metadata.pop(key, None)
        normalized["_eval3"] = metadata
    return normalized


def load_resumable_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    good_offset = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                good_offset = handle.tell()
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            if not isinstance(row, dict):
                break
            rows.append(row)
            good_offset = handle.tell()
    file_size = path.stat().st_size
    if good_offset < file_size:
        with path.open("r+b") as handle:
            handle.truncate(good_offset)
    return rows, good_offset


def append_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def discover_response_files(input_path: Path) -> tuple[list[Path], Path]:
    if input_path.is_file() and input_path.name == "inference_manifest.json":
        return response_files_from_manifest(input_path), input_path.parent
    if input_path.is_file():
        if not input_path.name.endswith("_response.jsonl"):
            raise ValueError("Input JSONL filename must end with _response.jsonl")
        return [input_path], input_path.parent
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    manifests = sorted(input_path.rglob("inference_manifest.json"))
    if manifests:
        files = []
        for manifest in manifests:
            files.extend(response_files_from_manifest(manifest))
    else:
        files = sorted(input_path.rglob("*_response.jsonl"))
    unique_files = list(dict.fromkeys(path.resolve() for path in files))
    if not unique_files:
        raise FileNotFoundError(f"No response JSONL files found under {input_path}")
    return unique_files, input_path.resolve()


def output_paths(response_path: Path, input_root: Path, output_root: Path) -> tuple[Path, Path]:
    try:
        relative_parent = response_path.resolve().parent.relative_to(input_root.resolve())
    except ValueError:
        relative_parent = Path(response_path.parent.name)
    stem = response_path.name.removesuffix("_response.jsonl")
    parent = output_root / relative_parent
    return parent / f"{stem}_judged.jsonl", parent / f"{stem}_execution_errors.jsonl"


def valid_existing_judgement(
    path: Path,
    expected_rows: int,
    variable_type_retry: bool,
) -> tuple[bool, Counter[int]]:
    if not path.is_file():
        return False, Counter()
    try:
        rows = read_jsonl(path)
    except Exception:
        return False, Counter()
    if len(rows) != expected_rows:
        return False, Counter()
    counts: Counter[int] = Counter()
    for row in rows:
        metadata = row.get("_eval3") or {}
        if metadata.get("judge_protocol") != JUDGEMENT_PROTOCOL:
            return False, Counter()
        if metadata.get("variable_type_retry") is not variable_type_retry:
            return False, Counter()
        error_type = row.get("error_type")
        if not isinstance(error_type, int) or error_type not in RESULT_KEY:
            return False, Counter()
        counts[error_type] += 1
    return True, counts


def build_response_key_map(rows: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], int], bool]:
    key_map: dict[tuple[Any, ...], int] = {}
    for index, row in enumerate(rows):
        key = row_resume_key(row)
        if key is None:
            return {}, False
        if key in key_map:
            raise ValueError(f"Duplicate response key at row {index}: {key}")
        key_map[key] = index
    return key_map, True


def validate_resume_rows(
    existing_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    existing_key_map: dict[tuple[Any, ...], int],
    source_key_map: dict[tuple[Any, ...], int],
    variable_type_retry: bool,
    require_complete: bool,
) -> None:
    existing_keys = set(existing_key_map)
    source_keys = set(source_key_map)
    if not existing_keys <= source_keys:
        missing = []
        extra = sorted(set(existing_key_map) - set(source_key_map))
        raise ValueError(
            "Cannot resume because judged rows do not match current responses: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if require_complete and existing_keys != source_keys:
        missing = sorted(source_keys - existing_keys)
        extra = []
        raise ValueError(
            "Cannot resume because judged rows do not match current responses: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for key, existing_index in existing_key_map.items():
        source_index = source_key_map[key]
        existing_row = existing_rows[existing_index]
        metadata = existing_row.get("_eval3") or {}
        if isinstance(metadata, dict):
            if "judge_protocol" in metadata and metadata["judge_protocol"] != JUDGEMENT_PROTOCOL:
                raise ValueError(f"Cannot resume because judge protocol mismatch at key={key}")
            if "variable_type_retry" in metadata and metadata["variable_type_retry"] != variable_type_retry:
                raise ValueError(f"Cannot resume because variable_type_retry mismatch at key={key}")
        if normalize_judged_row_for_resume(existing_row) != source_rows[source_index]:
            raise ValueError(f"Cannot resume because judged row mismatch at key={key}")


def check_rows_stream(
    rows: list[dict[str, Any]],
    solver_name: str,
    exec_timeout: int | None,
    workers: int,
    variable_type_retry: bool,
):
    if workers <= 1:
        for index, row in enumerate(tqdm(rows, desc="Executing and judging")):
            yield index, check_one(row, solver_name, exec_timeout, variable_type_retry)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(check_one, row, solver_name, exec_timeout, variable_type_retry): index
            for index, row in enumerate(rows)
        }
        for future in tqdm(as_completed(future_to_index), total=len(rows), desc="Executing and judging"):
            yield future_to_index[future], future.result()


def compute_pass_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the standard pass@k estimator after grouping samples by problem."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        metadata = row.get("_eval3") or {}
        dataset = str(metadata.get("dataset", ""))
        item_index = metadata.get("item_index")
        if not isinstance(item_index, int):
            item_index = row_index
        groups[(dataset, item_index)].append(row)

    if not groups:
        return {
            "problems": 0,
            "samples": 0,
            "samples_per_problem": {},
            "pass_at_k": {},
            "_pass_at_k_totals": {},
        }

    sample_counts = [len(group_rows) for group_rows in groups.values()]
    sample_count_histogram = Counter(sample_counts)
    max_supported_k = min(sample_counts)
    pass_totals: dict[str, float] = {}
    pass_at_k: dict[str, float] = {}
    for k in range(1, max_supported_k + 1):
        total = 0.0
        for group_rows in groups.values():
            n = len(group_rows)
            correct = sum(int(row.get("error_type") == 1) for row in group_rows)
            if n - correct < k:
                estimate = 1.0
            else:
                estimate = 1.0 - math.comb(n - correct, k) / math.comb(n, k)
            total += estimate
        key = str(k)
        pass_totals[key] = total
        pass_at_k[key] = round(total / len(groups), 4)

    return {
        "problems": len(groups),
        "samples": len(rows),
        "samples_per_problem": {
            str(sample_count): int(problem_count)
            for sample_count, problem_count in sorted(sample_count_histogram.items())
        },
        "pass_at_k": pass_at_k,
        "_pass_at_k_totals": pass_totals,
    }


def check_one(
    row: dict[str, Any],
    solver_name: str,
    exec_timeout: int | None,
    variable_type_retry: bool,
) -> dict[str, Any]:
    response = row.get("response")
    if not isinstance(response, str):
        return {
            "error_type": 2,
            "extracted_code": None,
            "returncode": None,
            "stdout": "",
            "stderr": "response field is missing or not a string",
            "timeout": False,
            "solver_result": None,
        }
    try:
        return check_result_detail(
            response,
            row,
            solver_name,
            exec_timeout,
            variable_type_retry=variable_type_retry,
        )
    except Exception as exc:
        return {
            "error_type": 3,
            "extracted_code": None,
            "returncode": None,
            "stdout": "",
            "stderr": f"judge exception: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
            "timeout": False,
            "solver_result": None,
        }


def check_rows(
    rows: list[dict[str, Any]],
    solver_name: str,
    exec_timeout: int | None,
    workers: int,
    variable_type_retry: bool,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [
            check_one(row, solver_name, exec_timeout, variable_type_retry)
            for row in tqdm(rows, desc="Executing and judging")
        ]
    details: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(check_one, row, solver_name, exec_timeout, variable_type_retry): index
            for index, row in enumerate(rows)
        }
        for future in tqdm(as_completed(future_to_index), total=len(rows), desc="Executing and judging"):
            details[future_to_index[future]] = future.result()
    return [detail for detail in details if detail is not None]


def judge_file(
    response_path: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = read_jsonl(response_path)
    judged_path, errors_path = output_paths(response_path, input_root, output_root)
    existing_rows, _ = load_resumable_jsonl(judged_path)
    existing_key_map, can_resume_by_key = build_response_key_map(existing_rows) if existing_rows else ({}, True)
    response_key_map, response_has_keys = build_response_key_map(rows) if rows else ({}, True)

    if not args.overwrite and existing_rows:
        if not (response_has_keys and can_resume_by_key):
            raise ValueError(
                f"Cannot resume {judged_path}: missing resume keys in response or judged rows."
            )
        if len(existing_rows) == len(rows):
            validate_resume_rows(
                existing_rows,
                rows,
                existing_key_map,
                response_key_map,
                args.variable_type_retry,
                True,
            )
            valid, counts = valid_existing_judgement(
                judged_path,
                len(rows),
                args.variable_type_retry,
            )
            if valid:
                print(f"skip existing complete judgement: {judged_path}")
                judged_rows = read_jsonl(judged_path)
                return {
                    "response_file": str(response_path),
                    "judged_file": str(judged_path),
                    "execution_errors_file": str(errors_path),
                    "rows": len(rows),
                    "counts": {str(key): int(counts.get(key, 0)) for key in RESULT_KEY},
                    "pass_at_1": round(counts.get(1, 0) / len(rows), 4) if rows else 0.0,
                    "skipped_existing": True,
                **compute_pass_metrics(judged_rows),
                }
            raise ValueError(f"Existing judgement file is complete but invalid: {judged_path}")
        validate_resume_rows(
            existing_rows,
            rows,
            existing_key_map,
            response_key_map,
            args.variable_type_retry,
            False,
        )

    if existing_rows and response_has_keys and can_resume_by_key:
        judged_key_set = set(existing_key_map)
        remaining_rows = [row for row in rows if row_resume_key(row) not in judged_key_set]
        print(
            f"resume partial judgement: {judged_path} "
            f"({len(existing_rows)}/{len(rows)} already judged, {len(remaining_rows)} remaining)"
        )
    else:
        remaining_rows = rows

    judged_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with judged_path.open("a", encoding="utf-8") as judged_handle:
        for local_index, detail in check_rows_stream(
            remaining_rows,
            args.solver_name,
            args.exec_timeout,
            max(1, args.check_workers),
            args.variable_type_retry,
        ):
            detail = json_safe(detail)
            error_type = int(detail["error_type"])
            output_row = deepcopy(remaining_rows[local_index])
            output_row["error_type"] = error_type
            output_row["error_type_name"] = RESULT_KEY[error_type]
            output_row["evaluation"] = detail
            row_key = row_resume_key(output_row)
            metadata = dict(output_row.get("_eval3") or {})
            metadata.update(
                {
                    "judge_stage": "completed",
                    "judge_protocol": JUDGEMENT_PROTOCOL,
                    "variable_type_retry": args.variable_type_retry,
                    "judged_item_index": response_key_map.get(row_key, local_index) if row_key is not None else local_index,
                }
            )
            output_row["_eval3"] = metadata
            append_jsonl_row(judged_handle, output_row)

    judged_rows = read_jsonl(judged_path)
    counts: Counter[int] = Counter()
    for row in judged_rows:
        error_type = row.get("error_type")
        if isinstance(error_type, int):
            counts[error_type] += 1
    execution_errors = [row for row in judged_rows if row.get("error_type") == 3]
    write_jsonl_atomic(errors_path, execution_errors)
    accuracy = round(counts.get(1, 0) / len(rows), 4) if rows else 0.0
    pass_metrics = compute_pass_metrics(judged_rows)
    metric_text = ", ".join(
        f"pass@{k}={value}" for k, value in pass_metrics["pass_at_k"].items()
    )
    print(f"judged {response_path.name}: {metric_text}; counts={dict(counts)}")
    return {
        "response_file": str(response_path),
        "judged_file": str(judged_path),
        "execution_errors_file": str(errors_path),
        "rows": len(rows),
        "counts": {str(key): int(counts.get(key, 0)) for key in RESULT_KEY},
        "pass_at_1": accuracy,
        "skipped_existing": False,
        **pass_metrics,
    }


def write_summary(output_root: Path, file_results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    total_counts: Counter[int] = Counter()
    total_rows = 0
    total_problems = 0
    pass_totals: dict[str, float] = defaultdict(float)
    pass_problem_counts: Counter[str] = Counter()
    for result in file_results:
        total_rows += result["rows"]
        total_problems += result["problems"]
        for key, value in result["counts"].items():
            total_counts[int(key)] += int(value)
        for key, value in result["_pass_at_k_totals"].items():
            pass_totals[key] += float(value)
            pass_problem_counts[key] += int(result["problems"])
    overall_pass_at_k = {
        key: round(pass_totals[key] / pass_problem_counts[key], 4)
        for key in sorted(pass_totals, key=int)
    }
    overall = overall_pass_at_k.get("1", 0.0)
    public_file_results = [
        {key: value for key, value in result.items() if not key.startswith("_")}
        for result in file_results
    ]
    summary = {
        "protocol": JUDGEMENT_PROTOCOL,
        "stage": "judge",
        "solver_name": args.solver_name,
        "exec_timeout": args.exec_timeout,
        "check_workers": max(1, args.check_workers),
        "gurobi_threads": int(os.environ.get("ORSTAR_GUROBI_THREADS", "1")),
        "variable_type_retry": args.variable_type_retry,
        "result_key": {str(key): value for key, value in RESULT_KEY.items()},
        "files": public_file_results,
        "overall_rows": total_rows,
        "overall_problems": total_problems,
        "overall_counts": {str(key): int(total_counts.get(key, 0)) for key in RESULT_KEY},
        "overall_pass_at_1": overall,
        "overall_pass_at_k": overall_pass_at_k,
        "overall_problems_at_k": {
            key: int(pass_problem_counts[key]) for key in sorted(pass_problem_counts, key=int)
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "judgement_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path = output_root / "judgement_summary.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(f"overall pass@1: {overall}\n")
        for key, value in overall_pass_at_k.items():
            if key != "1":
                handle.write(f"overall pass@{key}: {value}\n")
        handle.write(f"overall problems: {total_problems}\n")
        handle.write(f"overall rows: {total_rows}\n")
        handle.write("overall counts:\n")
        for key, name in RESULT_KEY.items():
            handle.write(f"  {key}:{name}={total_counts.get(key, 0)}\n")
        handle.write("\nper-file results:\n")
        for result in public_file_results:
            metrics = ", ".join(
                f"pass@{key}={value}" for key, value in result["pass_at_k"].items()
            )
            handle.write(
                f"  {Path(result['response_file']).name}: {metrics}; "
                f"problems={result['problems']}; samples={result['samples']}\n"
            )
    print(f"saved summary: {json_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True, help="Response JSONL, inference manifest, or inference directory.")
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--solver_name", default="python_algo", choices=["gurobi", "copt", "text", "python_algo"])
    parser.add_argument("--exec_timeout", type=int, default=None)
    parser.add_argument("--check_workers", type=int, default=int(os.getenv("ORSTAR_CHECK_WORKERS", "8")))
    parser.add_argument("--gurobi_threads", type=int, default=int(os.getenv("ORSTAR_GUROBI_THREADS", "1")))
    parser.add_argument(
        "--variable-type-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry numeric mismatches once after SIRL-style INTEGER/CONTINUOUS conversion (default: enabled).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["ORSTAR_GUROBI_THREADS"] = str(args.gurobi_threads)
    response_files, input_root = discover_response_files(args.input_path.expanduser().resolve())
    output_root = args.output_path.expanduser().resolve() if args.output_path else input_root / "judged"
    print(f"found {len(response_files)} response files")
    file_results = [
        judge_file(path, input_root, output_root, args)
        for path in response_files
    ]
    write_summary(output_root, file_results, args)


if __name__ == "__main__":
    main()
