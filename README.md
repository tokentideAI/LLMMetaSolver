
<div align="center">
  <h2>LLMs as Adaptive Meta-Solvers: Strategy-Diverse RL for Industrial-Scale Optimization</h2>
  <a href="https://arxiv.org/abs/2609.34427"><img src="https://img.shields.io/badge/Paper-arXiv%202609.34427-red" alt="Paper"></a>
  <a href="https://huggingface.co/tokentideAI/SDRL-Qwen3-32B"><img src="https://img.shields.io/badge/Model-%F0%9F%A4%97%20HuggingFace-yellow" alt="Model"></a>
  <a href="https://github.com/tokentideAI/LLMMetaSolver"><img src="https://img.shields.io/badge/GitHub-LLMMetaSolver-black?logo=github" alt="GitHub"></a>
</div>

## Overview

We introduce **Strategy-Diverse Reinforcement Learning (SDRL)**, a framework that train open-source LLMs into adaptive optimization meta-solvers for real-world, industrial-scale tasks.
Instead of committing to a single solver-integrated paradigm, the model adaptively routes each problem to the most appropriate computational strategy:

1. **Exact Solver (Gurobi)** for compact LP/MILP formulations with tractable scale
2. **Exact Combinatorial Algorithm** (such as DP / greedy / graph / backtracking) for problems with exploitable structure
3. **Heuristic Search** (such as ALNS, GRASP, local search, simulated annealing) for large-scale NP-hard instances

SDRL leverages the empirical complementarity of these three strategy families through a **correctness-gated hierarchical diversity reward** that promotes exploration both across strategies and within each strategy, preventing premature strategy collapse. A **mixed-format training scheme** jointly supports self-contained textual problems and file-grounded industrial instances whose data is distributed across external files.

## 📢 News & Updates

- **[2026.11.12]** 🔬 Dataset & Evaluation Update: We release the training dataset and extend our evaluation to Automatic Algorithm Design (AAD) and FrontierOR benchmarks, providing new results on LLM-driven algorithm discovery and feasibility verification across challenging optimization problems.

- **[2026.10.08]** 🚀 **Model & Code Release:** Model checkpoint and benchmark inference code are now available on [GitHub](https://github.com/tokentideAI/LLMMetaSolver) and Hugging Face ([SDRL-Qwen3-4B](https://huggingface.co/tokentideAI/SDRL-Qwen3-4B) & [SDRL-Qwen3-32B](https://huggingface.co/tokentideAI/SDRL-Qwen3-32B)).
- **[2026.09.28]** 📄 **Paper Release:** Our paper [LLMs as Adaptive Meta-Solvers: Strategy-Diverse RL for Industrial-Scale Optimization](https://arxiv.org/abs/2609.34427) is now published on arXiv.


## Training data

The cleaned training subset contains **20,320 instances**, including self-contained
and file-grounded optimization problems. Download [data/train.parquet](https://github.com/tokentideAI/LLMMetaSolver/raw/refs/heads/main/data/train.parquet)
(22.2 MiB), or clone this repository. No Git LFS or shard restoration is required.
See [data/README.md](data/README.md) for the schema, embedded CSV assets, loading
example, and checksum.

## Evaluation
### Evaluation on the calssical LLMs for optimization modeling benchmark
We evaluate on seven NL-to-Opt benchmarks: NL4Opt,
MAMO-EasyLP and MAMO-ComplexLP, IndustryOR,
OptMATH-Bench, OptiBench, and MIPLIB-NL (Li et al., 2026).
The first six use self-contained textual inputs, while MIPLIB-NL introduces file-grounded,
industrial-scale optimization tasks that demand the ability to dynamically read and process external data files at runtime.
Following the rigorous evaluation protocol used by  [SIRL](https://huggingface.co/chenyitian-shanshu/SIRL), a solution is considered valid if the relative error is less than 1e-6.
The performance metrics are as follows:

| Method | Self-contained | | | | | | File-grounded | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | NL4Opt | MAMO-E | MAMO-C | Ind-OR | OptMath | OptiBench | MIPLIB-NL | |
| **Base Models** | | | | | | | | |
| Qwen3-4B-Instruct-2507 | 72.7 | 68.9 | 42.4 | 40.0 | 17.5 | 54.1 | 7.7 | 43.3 |
| Qwen3-32B | 85.3 | 88.8 | 66.0 | 39.0 | 17.5 | 60.0 | 10.5 | 52.4 |
| **Fine-tuned Models** | | | | | | | | |
| ORLM-Llama3-8B* | 85.7 | 82.3 | 37.4 | 24.0 | 2.6 | 51.1 | 0.6 | 40.5 |
| OptMATH-Qwen2.5-7B* | 94.7 | 86.5 | 51.2 | 20.0 | 24.4 | 57.9 | 0.0 | 47.8 |
| OptMATH-Qwen2.5-32B* | 95.9 | 89.9 | 54.1 | 31.0 | 34.7 | 66.1 | 5.5 | 53.9 |
| SIRL-Qwen2.5-7B* | 96.3 | 91.7 | 51.7 | 33.0 | 30.5 | 58.0 | 0.5 | 51.7 |
| SIRL-Qwen2.5-32B* | **98.0** | 94.6 | 61.1 | 42.0 | 45.8 | 67.4 | 1.4 | 58.6 |
| StepORLM-Qwen3-8B | 89.8 | 89.9 | 52.2 | 40.0 | 14.5 | 56.5 | 0.9 | 49.1 |
| SDRL-Qwen3-4B | <u>96.3</u> | 92.4 | 79.3 | 55.0 | 41.0 | 67.1 | 23.6 | 64.6 |
| SDRL-Qwen3-32B | **96.3** | **96.0** | **81.8** | **56.0** | **59.0** | **69.1** | **30.0** | **69.7** |
| **Frontier Models** | | | | | | | | |
| DeepSeek-V4-Pro | 95.5 | 90.0 | 87.2 | 63.0 | 49.4 | 66.9 | 28.2 | 68.6 |
| Qwen3.5-397B-A17B | 94.7 | 91.4 | 81.3 | 64.0 | 42.2 | 65.5 | 32.3 | 67.3 |
| GPT-5.5 | 95.5 | 93.8 | 92.1 | 67.0 | 40.4 | 69.6 | 29.1 | 69.6 |
| Claude-Opus-4.8 | 92.2 | 94.2 | 93.1 | 64.0 | 60.2 | 64.6 | 39.1 | 72.5 |
| Gemini-3.1-Pro-Preview | 90.2 | 91.6 | 93.0 | 67.0 | 51.8 | 66.1 | 40.9 | 71.5 |

*Note: \* denotes results from original or reproduced papers.*

### Evaluation on the LLM-based AAD tasks
We further investigate whether SDRL's strategy-diverse training paradigm transfers beyond natural-language optimization modeling to Automatic Algorithm Design (AAD) tasks.
In particular, we present a direct  comparison with Evolution of Heuristics (EoH) task on the first fit setting, evaluating the ability of LLMs to generate effective optimization algorithms and discover high-quality solutions.

**Table 1. Online bin packing: excess bins over the lower bound (%). Lower is better.**

| Method | 1k_C100 | 5k_C100 | 10k_C100 | 1k_C500 | 5k_C500 | 10k_C500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First Fit | 5.32 | 4.40 | 4.44 | 4.97 | 4.27 | 4.28 |
| Best Fit | 4.87 | 4.08 | 4.09 | 4.50 | 3.91 | 3.95 |
| FunSearch | 3.78 | 0.80 | 0.33 | 6.75 | 1.47 | 0.74 |
| EoH | 2.24 | 0.80 | 0.61 | 2.13 | 0.78 | 0.61 |
| SDRL (Ours) | 0.709 | 0.432 | 0.313 | 0.000 | 0.000 | 0.000 |

**Table 2. Traveling salesman problem: distance gap to the best-known solution (%). Lower is better.**

| Method | rd100 | pr124 | bier127 | kroA150 | u159 | kroB200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NI | 19.91 | 15.50 | 23.21 | 18.17 | 23.59 | 24.10 |
| FI | 9.38 | 4.43 | 8.04 | 8.54 | 11.15 | 7.54 |
| OR-Tools | 0.01 | 0.55 | 0.66 | 0.02 | 1.75 | 2.57 |
| AM | 3.41 | 3.68 | 5.91 | 3.78 | 7.55 | 7.11 |
| POMO | 0.01 | 0.60 | 13.72 | 0.70 | 0.95 | 1.58 |
| LEHD | 0.01 | 1.11 | 4.76 | 1.40 | 1.13 | 0.64 |
| EoH | 0.01 | 0.00 | 0.42 | 0.00 | 0.00 | 0.20 |
| SDRL (Ours) | 0.000 | 2.370 | 1.084 | 0.980 | 0.000 | 3.737 |

**Table 3. Flow shop scheduling: relative makespan gap (%). Lower is better.**

| Method | n20m10 | n20m20 | n50m10 | n50m20 | n100m10 | n100m20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GUPTA | 23.42 | 21.79 | 20.11 | 22.78 | 15.03 | 21.00 |
| CDS | 12.87 | 10.35 | 12.72 | 15.03 | 9.36 | 13.55 |
| NEH | 4.05 | 3.06 | 3.47 | 5.48 | 2.07 | 3.58 |
| NEHFF | 4.15 | 2.72 | 3.62 | 5.10 | 1.88 | 3.73 |
| PFSPNet | 14.78 | 14.69 | 11.95 | 16.95 | 8.21 | 16.47 |
| PFSPNet-NEH | 4.04 | 2.96 | 3.48 | 5.05 | 1.72 | 3.56 |
| EoH | 0.30 | 0.10 | 0.19 | 0.60 | 0.14 | 0.41 |
| SDRL (Ours) | 2.276 | 1.698 | 7.041 | 4.361 | 1.317 | 4.057 |

### Evaluation on the FrontierOR benchmark
We further evaluate SDRL on FrontierOR, a benchmark featuring computationally challenging optimization problems with diverse structures and practical requirements.
Unlike conventional objective-based evaluations, FrontierOR explicitly incorporates independent feasibility verification alongside objective-based performance assessment.
All experiments follow the default evaluation setting, with a 3,600-second time limit per instance.

| Model | Full Execution | Full Feasibility | Full Solution quality | Full QTE | Hard Execution | Hard Feasibility | Hard Solution quality | Hard QTE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude Opus 4.6 | 0.93 | 0.62 | 0.48 | 0.31 | 0.94 | 0.60 | 0.44 | 0.32 |
| GPT-5.3-Codex | 0.98 | 0.60 | 0.48 | 0.26 | 0.98 | 0.49 | 0.30 | 0.18 |
| Gemini 3.1 Pro | 0.93 | 0.61 | 0.52 | 0.25 | 1.00 | 0.64 | 0.44 | 0.22 |
| DeepSeek-R1 | 0.74 | 0.42 | 0.31 | 0.17 | 0.82 | 0.37 | 0.20 | 0.11 |
| Grok-4.20-beta | 0.74 | 0.28 | 0.22 | 0.13 | 0.76 | 0.20 | 0.14 | 0.06 |
| Qwen3-Coder-Plus | 0.60 | 0.26 | 0.20 | 0.09 | 0.52 | 0.21 | 0.12 | 0.07 |
| LLaMA-4-Maverick | 0.47 | 0.18 | 0.13 | 0.06 | 0.52 | 0.13 | 0.07 | 0.02 |
| SIR-RL-Qwen3-32B | 0.86 | 0.22 | 0.15 | 0.09 | 0.82 | 0.15 | 0.12 | 0.10 |
| SDRL-Qwen3-32B | 0.90 | 0.33 | 0.26 | 0.18 | 0.88 | 0.29 | 0.22 | 0.22 |


## EoH evaluation

The additional [EoH task set](test_data/eoh_eval.jsonl) contains 18 instances
(online bin packing, TSP, and flow shop scheduling) with 30 input CSVs in
`test_data/eoh_instances/`. Its separate evaluator reports **relative gap (%)**,
with a default 3600-second execution limit and support for returned incumbents
without requiring proof of optimality.

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_NAME="$PWD/models/eval-model" \
  OUTPUT_PATH=outputs/eoh TENSOR_PARALLEL_SIZE=1 \
  SEED=1 NUM_SAMPLES=1 TEMPERATURE=0.5 bash eoh_eval/inference.sh

INPUT_PATH=outputs/eoh bash eoh_eval/judge.sh
```

See [eoh_eval/README.md](eoh_eval/README.md) for setup, gap aggregation, timeout
handling, and limitations. Use this separate judge for EoH; it does not use
correctness or Pass@K scoring.

## Inference

We recommend using the strategy-routing meta-solver prompt below (from `eval/prompt.py` in our evaluation suite). Replace `{question}` with the natural-language OR problem.

### System prompt

```text
You are an expert Operations Research (OR) Engineer and Algorithm Designer.
Your task is to analyze optimization problems, dynamically route them to the single most appropriate solution paradigm based on their mathematical characteristics and scale, and implement the solution in Python.

### Step 1: Problem Analysis
Briefly analyze the problem based on the following dimensions:
- Problem Type: E.g., Routing, Scheduling, Assignment, Inventory, Network Flow.
- Mathematical Structure: Continuous vs. Discrete/Combinatorial? Linear vs. Non-linear? Does it have optimal substructure?
- Scale Estimation: Approximate the number of variables and constraints. And determine if it's Small/Medium (exact solvers manageable) vs. Large-scale (prone to combinatorial explosion).

### Step 2: Strategy Routing (Choose ONE Paradigm)
Based on the analysis, strictly select ONE of the following strategies:
1. Exact Solver (Gurobi):
   - CHOOSE IF: The problem is a clearly defined Linear Programming (LP), Mixed-Integer Linear Programming (MILP), or Convex Quadratic problem, AND the scale is manageable (small to medium).
   - IMPLEMENTATION: Write rigorous mathematical formulation using `gurobipy`. Start code with: `import gurobipy as gp\nfrom gurobipy import GRB`
2. Algorithmic Design (DP/Greedy/Graph):
   - CHOOSE IF: The problem exhibits optimal substructure, overlapping subproblems, or standard graph properties (e.g., Knapsack, Shortest Path, Interval Scheduling).
   - IMPLEMENTATION: Use Dynamic Programming, Greedy, Divide-and-Conquer, or Backtracking.
3. Metaheuristic / Heuristic:
   - CHOOSE IF: The problem is large-scale, highly non-linear, or a complex Combinatorial Optimization problem (NP-hard, e.g., large VRP/TSP) where exact solvers would time out.
   - IMPLEMENTATION: Use a high-performance heuristic (e.g., ALNS, GRASP, Iterated Local Search, Simulated Annealing).

### Step 3: Output Format
You must structure your response exactly as follows:
1. First, output your step-by-step reasoning (Analysis & Routing Decision).
2. Finally, output the complete, executable Python code in a single code block starting with ```python
3. The absolute final line of your Python script must be exactly: `print("Result:", objective_value)` (replace objective_value with your final computed variable)."""
```

### Quick start

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "tokentideAI/SDRL-Qwen3-4B"  # replace with the actual repo id

SYSTEM_PROMPT = """..."""  # paste the full system prompt above

question = (
    "An industrial tire company delivers large tires for equipment to remote engineering sites "
    "either by cargo planes or ultrawide trucks. Each cargo plane can transport 10 tires per trip "
    "and costs $1000. Each ultrawide truck can transport 6 tires per trip and costs $700. The company "
    "needs to transport at least 200 tires and has available $22000. Because most remote sites don't "
    "have proper airports, the number of plane trips cannot exceed the number of ultrawide truck trips. "
    "How many trips of each should be done to minimize the total number of trips?"
)

model = LLM(
    MODEL_ID,
    tensor_parallel_size=1,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sampling_params = SamplingParams(
    n=1,
    temperature=0.5,
    top_p=0.95,
    max_tokens=16384,
    repetition_penalty=1.02,
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT.strip()},
    {"role": "user", "content": f"Solve the following algorithm design and optimization problem:\n{question}\n\nReason step-by-step to derive the modeling process and strategy selection, then generate the corresponding Python script.\n/no_think"},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
response = model.generate(prompt, sampling_params)
print(response[0].outputs[0].text)
```

The generated script's final output line should be `print("Result:", objective_value)`, where `objective_value` is the actual computed objective value. Extract the code block and execute it with `python3`.
A Gurobi license is required for the MILPLib-NL benchmark; the free edition is sufficient for the other benchmarks.



## Citation

If you find the proposed SDRL framework useful or relevant to your research, please consider citing our paper:

```bibtex
@article{zhang2026llm,
  title={LLMs as Adaptive Meta-Solvers: Strategy-Diverse RL for Industrial-Scale Optimization},
  author={Zhang, Shihao and Liu, Weiting and Shao, Siyu and Chen, Yitian and Feng, Jianfeng and Ge, Dongdong and Ye, Yinyu},
  journal={arXiv preprint arXiv:2609.34427},
  year={2026}
}
```
