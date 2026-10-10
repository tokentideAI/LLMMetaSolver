#!/usr/bin/env python3
"""Stage 1: generate ORStar responses on a GPU machine without executing code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ORSTAR_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = ORSTAR_DIR / "test_data"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "inference_outputs"
DEFAULT_MIPLIB_DATA_NAME = (
    "MIPLIB-NL-Fixed/MIPLIB_train_v2_compact.jsonl"
    if (DEFAULT_DATA_PATH / "MIPLIB-NL-Fixed" / "MIPLIB_train_v2_compact.jsonl").is_file()
    else "MIPLIB.jsonl"
)
DEFAULT_DATA_NAMES = [
    "MAMO_ComplexLP_fixed.jsonl",
    "IndustryOR_fixedV2.json",
    "OptMATH_Bench_166.jsonl",
    "NL4OPT.jsonl",
    "OptiBench.jsonl",
    "MAMO_EasyLP_fixed.jsonl",
    "LogIOR.jsonl",
    DEFAULT_MIPLIB_DATA_NAME,
]

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from prompt import sdrl_prompt, sir_prompt


PROMPT_TEMPLATES = {
    "sdrl_prompt": sdrl_prompt,
    "sir_prompt": sir_prompt,
}


def parse_data_names(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    if isinstance(parsed, str):
        return [name.strip() for name in parsed.split(",") if name.strip()]
    return list(parsed)


def parse_model_list(value: str | None) -> list[str]:
    if not value:
        return []
    candidate = Path(value).expanduser()
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if isinstance(parsed, dict):
            parsed = parsed.get("models", [])
        if isinstance(parsed, str):
            return [parsed]
        return list(parsed)
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    if isinstance(parsed, str):
        return [item.strip() for item in parsed.split(",") if item.strip()]
    return list(parsed)


def parse_model_spec(value: str) -> tuple[str | None, str]:
    """Parse either a plain model path or ``label<TAB>model_path``."""
    value = str(value).strip()
    if "\t" in value:
        label, model_path = value.split("\t", 1)
        label = safe_output_name(label.strip())
        model_path = model_path.strip()
        if not model_path:
            raise ValueError(f"Model path is empty in model spec: {value!r}")
        return label, model_path
    return None, value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def safe_output_name(value: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return safe_name or "model"


def infer_run_name(model_name: str) -> str:
    path = Path(model_name)
    for part in path.parts:
        if part.startswith("models--"):
            return safe_output_name(part.removeprefix("models--").replace("--", "__"))
    if "/" in model_name and not model_name.startswith(("/", ".", "~")):
        return safe_output_name(model_name.replace("/", "__"))
    return safe_output_name(path.name)


def configure_model_source(model_source: str) -> None:
    if model_source == "modelscope":
        os.environ["VLLM_USE_MODELSCOPE"] = "True"
    elif model_source == "huggingface":
        os.environ["VLLM_USE_MODELSCOPE"] = "False"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_data(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        # Several ORStar benchmark files use JSONL content despite a .json suffix.
        return load_jsonl(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise TypeError(f"Unsupported data format in {path}: {type(data).__name__}")


def resolve_tensor_parallel_size(value: str) -> int:
    if str(value).lower() != "auto":
        return int(value)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        return max(1, len([item for item in visible_devices.split(",") if item.strip()]))
    import torch

    return max(1, torch.cuda.device_count())


def infer_fsdp_world_size(checkpoint_path: str) -> int:
    checkpoint_dir = Path(checkpoint_path)
    world_sizes = set()
    for path in checkpoint_dir.glob("model_world_size_*_rank_*.pt"):
        match = re.search(r"model_world_size_(\d+)_rank_\d+\.pt$", path.name)
        if match:
            world_sizes.add(int(match.group(1)))
    if not world_sizes:
        raise FileNotFoundError(f"No FSDP shard found under {checkpoint_dir}")
    if len(world_sizes) != 1:
        raise ValueError(f"Multiple FSDP world sizes under {checkpoint_dir}: {sorted(world_sizes)}")
    return world_sizes.pop()


def convert_fsdp_checkpoint(checkpoint_path: str, model_path: str, world_size: int | None) -> str:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output_path = Path(checkpoint_path) / "pretrained"
    if output_path.is_dir():
        return str(output_path)

    state_dict: dict[str, list[Any]] = defaultdict(list)
    shard_count = world_size or infer_fsdp_world_size(checkpoint_path)
    for rank in range(shard_count):
        shard_path = Path(checkpoint_path) / f"model_world_size_{shard_count}_rank_{rank}.pt"
        print(f"loading FSDP shard: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu")
        for key, value in shard.items():
            state_dict[key].append(value.to_local())
    merged = {key: torch.cat(values, dim=0) for key, values in state_dict.items()}
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(merged)
    model.save_pretrained(output_path, max_shard_size="10GB")
    AutoTokenizer.from_pretrained(model_path).save_pretrained(output_path)
    return str(output_path)


def prepare_tokenizer_path(model_path: str, output_path: Path) -> str:
    model_dir = Path(model_path).expanduser()
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        return model_path
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model_path
    except Exception as load_error:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise load_error
        extra_tokens = config.get("extra_special_tokens")
        if not isinstance(extra_tokens, list):
            raise load_error
        compat_dir = output_path / ".tokenizer_compat"
        compat_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ):
            source = model_dir / filename
            if source.is_file():
                shutil.copy2(source, compat_dir / filename)
        config.pop("extra_special_tokens", None)
        config.setdefault("additional_special_tokens", extra_tokens)
        (compat_dir / "tokenizer_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"using tokenizer compatibility copy: {compat_dir}")
        return str(compat_dir)


def patch_vllm_transformers_tokenizer_compat() -> None:
    """Bridge vLLM 0.11's legacy tokenizer property on Transformers 5.x.

    vLLM 0.11 accesses ``all_special_tokens_extended`` while Transformers 5.x
    removed that property from ``PreTrainedTokenizerBase``.  The vLLM cache
    only needs the token strings here, so the current ``all_special_tokens``
    list is a safe fallback for Qwen2/Qwen3 tokenizers.
    """
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
            lambda tokenizer: tokenizer.all_special_tokens
        )
        print("using vLLM/Transformers tokenizer compatibility shim")


def get_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    common = {
        "n": args.num_samples,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "stop": ["<|im_end|>", "</s>", "[/INST]"],
        "repetition_penalty": 1.05,
    }
    if args.decoding_type == "top_p":
        temperature = 0.5 if args.temperature is None else args.temperature
        return SamplingParams(temperature=temperature, top_p=0.95, **common)
    if args.decoding_type == "greedy":
        return SamplingParams(temperature=0.0, top_p=1.0, top_k=1, **common)
    if args.decoding_type == "min_p":
        temperature = 0.8 if args.temperature is None else args.temperature
        return SamplingParams(temperature=temperature, min_p=0.05, **common)
    if args.decoding_type == "beamsearch":
        return SamplingParams(temperature=0.0, top_p=1.0, use_beam_search=True, best_of=5, **common)
    raise ValueError(f"Unsupported decoding type: {args.decoding_type}")


def render_prompt(item: dict[str, Any], tokenizer: Any, args: argparse.Namespace) -> str:
    if "en_question" not in item:
        raise KeyError("dataset row has no en_question")
    template = PROMPT_TEMPLATES[args.prompt_name]
    messages = []
    if template.get("system"):
        messages.append({"role": "system", "content": template["system"].strip()})
    messages.append({"role": "user", "content": template["user"].format(Question=item["en_question"]).strip()})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def generate_batches(model: Any, prompts: list[str], sampling_params: Any, batch_size: int) -> list[list[str]]:
    responses = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
        batch = prompts[start : start + batch_size]
        outputs = model.generate(batch, sampling_params)
        responses.extend([[candidate.text for candidate in output.outputs] for output in outputs])
    return responses


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def valid_existing_output(
    path: Path,
    expected_items: int,
    num_samples: int,
    seed: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        rows = load_jsonl(path)
    except Exception:
        return False
    if len(rows) != expected_items * num_samples:
        return False
    if not all(isinstance(row.get("response"), str) for row in rows):
        return False
    for row in rows:
        metadata = row.get("_eval3") or {}
        if isinstance(metadata, dict) and "seed" in metadata:
            if metadata.get("seed") != seed:
                return False
        elif seed is not None:
            # Do not silently reuse a pre-seed response for an explicitly seeded run.
            return False
    if num_samples == 1:
        return True
    samples_by_item: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        metadata = row.get("_eval3") or {}
        item_index = metadata.get("item_index")
        sample_index = metadata.get("sample_index")
        if not isinstance(item_index, int) or not isinstance(sample_index, int):
            return False
        if metadata.get("num_samples") != num_samples:
            return False
        samples_by_item[item_index].add(sample_index)
    expected_sample_indexes = set(range(num_samples))
    return (
        set(samples_by_item) == set(range(expected_items))
        and all(sample_indexes == expected_sample_indexes for sample_indexes in samples_by_item.values())
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_model_inference_only(
    args: argparse.Namespace,
    model_name: str,
    run_name: str,
    output_path: Path,
    data_names: list[str],
    tensor_parallel_size: int,
) -> None:
    from transformers import AutoTokenizer

    patch_vllm_transformers_tokenizer_compat()
    from vllm import LLM

    model_load_path = model_name
    if args.checkpoint:
        model_load_path = convert_fsdp_checkpoint(args.checkpoint, model_name, args.fsdp_world_size)
    tokenizer_source = args.tokenizer_name or model_load_path
    tokenizer_path = prepare_tokenizer_path(tokenizer_source, output_path)
    llm_kwargs = {
        "model": model_load_path,
        "tokenizer": tokenizer_path,
        "tensor_parallel_size": tensor_parallel_size,
        "seed": args.seed,
        "dtype": args.vllm_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    model = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    sampling_params = get_sampling_params(args)
    manifest_files = []

    for dataset_name in data_names:
        dataset_path = Path(args.data_path) / dataset_name
        rows = load_data(dataset_path)
        if not rows:
            print(f"skip empty dataset: {dataset_name}")
            continue
        prefix = Path(dataset_name).stem
        sampling_suffix = f"_pass{args.num_samples}" if args.num_samples > 1 else ""
        response_path = output_path / f"{run_name}_{prefix}_{args.decoding_type}{sampling_suffix}_response.jsonl"
        if not args.overwrite and valid_existing_output(
            response_path,
            len(rows),
            args.num_samples,
            args.seed,
        ):
            print(f"skip existing complete response file: {response_path}")
        else:
            prompts = [render_prompt(row, tokenizer, args) for row in rows]
            print(
                f"generating {len(prompts)} prompts x {args.num_samples} samples "
                f"for {dataset_name}"
            )
            if args.verbose:
                print(prompts[0][:1000])
            responses_by_item = generate_batches(model, prompts, sampling_params, args.batch_size)
            if len(responses_by_item) != len(rows):
                raise RuntimeError(
                    f"response group count {len(responses_by_item)} != input count {len(rows)}"
                )
            output_rows = []
            for item_index, (row, responses) in enumerate(zip(rows, responses_by_item)):
                if len(responses) != args.num_samples:
                    raise RuntimeError(
                        f"item {item_index} produced {len(responses)} samples; "
                        f"expected {args.num_samples}"
                    )
                for sample_index, response in enumerate(responses):
                    output_row = deepcopy(row)
                    output_row["response"] = response
                    output_row["_eval3"] = {
                        "stage": "inference",
                        "dataset": dataset_name,
                        "item_index": item_index,
                        "sample_index": sample_index,
                        "num_samples": args.num_samples,
                        "model_name": model_name,
                        "run_name": run_name,
                        "decoding_type": args.decoding_type,
                        "seed": args.seed,
                        "temperature": args.temperature,
                        "prompt_name": args.prompt_name,
                        "enable_thinking": args.enable_thinking,
                    }
                    output_rows.append(output_row)
            write_jsonl_atomic(response_path, output_rows)
            print(f"saved responses: {response_path}")
        manifest_files.append(
            {
                "dataset": dataset_name,
                "items": len(rows),
                "rows": len(rows) * args.num_samples,
                "num_samples": args.num_samples,
                "response_file": response_path.name,
                "response_file_absolute": str(response_path),
                "sha256": file_sha256(response_path),
            }
        )

    manifest = {
        "protocol": "orstar-eval3-response-v1",
        "stage": "inference",
        "model_name": model_name,
        "run_name": run_name,
        "model_load_path": model_load_path,
        "tokenizer_path": tokenizer_path,
        "tensor_parallel_size": tensor_parallel_size,
        "decoding_type": args.decoding_type,
        "seed": args.seed,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "prompt_name": args.prompt_name,
        "enable_thinking": args.enable_thinking,
        "max_tokens": args.max_tokens,
        "files": manifest_files,
    }
    manifest_path = output_path / "inference_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"saved inference manifest: {manifest_path}")
    del model
    del tokenizer
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=os.getenv("ORSTAR_DATA_PATH", str(DEFAULT_DATA_PATH)))
    parser.add_argument("--data_name", default=DEFAULT_DATA_NAMES)
    parser.add_argument("--output_path", default=os.getenv("ORSTAR_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))
    parser.add_argument("--model_name", default=os.getenv("ORSTAR_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--model_list", default=os.getenv("ORSTAR_MODEL_LIST"))
    parser.add_argument("--tokenizer_name", default=os.getenv("ORSTAR_TOKENIZER_NAME"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fsdp_world_size", type=int, default=None)
    parser.add_argument(
        "--prompt_name",
        default="sdrl_prompt",
        choices=sorted(PROMPT_TEMPLATES),
    )
    parser.add_argument("--model_source", default=os.getenv("ORSTAR_MODEL_SOURCE", "auto"), choices=["auto", "modelscope", "huggingface"])
    parser.add_argument("--vllm_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--tensor_parallel_size", default=os.getenv("ORSTAR_TENSOR_PARALLEL_SIZE", "auto"))
    parser.add_argument("--gpu_memory_utilization", type=float, default=float(os.getenv("ORSTAR_GPU_MEMORY_UTILIZATION", "0.95")))
    parser.add_argument("--max_model_len", type=int, default=int(os.getenv("ORSTAR_MAX_MODEL_LEN")) if os.getenv("ORSTAR_MAX_MODEL_LEN") else None)
    parser.add_argument("--decoding_type", default="top_p", choices=["greedy", "top_p", "beamsearch", "min_p"])
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("ORSTAR_SEED")) if os.getenv("ORSTAR_SEED") else 1,
        help="Random seed for vLLM model initialization and sampling (default: 1).",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--enable_thinking", type=parse_bool, default=False)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1")
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative")
    configure_model_source(args.model_source)
    data_names = parse_data_names(args.data_name)
    model_names = parse_model_list(args.model_list) or [args.model_name]
    tensor_parallel_size = resolve_tensor_parallel_size(args.tensor_parallel_size)
    for index, model_spec in enumerate(model_names, start=1):
        model_label, model_name = parse_model_spec(model_spec)
        inferred_name = model_label or infer_run_name(model_name)
        if args.run_name and model_label:
            run_name = f"{safe_output_name(args.run_name)}_{inferred_name}"
        elif args.run_name:
            run_name = safe_output_name(args.run_name)
        else:
            run_name = inferred_name
        output_path = Path(args.output_path) / run_name
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(model_names)}] inference-only: {model_name}")
        evaluate_model_inference_only(
            args,
            model_name,
            run_name,
            output_path,
            data_names,
            tensor_parallel_size,
        )


if __name__ == "__main__":
    main()
