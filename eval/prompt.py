"""Prompt templates for LLMMetaSolver evaluation."""
sir_prompt = {
    "system": """You are a helpful assistant with expertise in mathematical modeling and the Gurobi solver. When the user provides an Operations Research (OR) problem, you must:
1. Carefully analyze the problem and clearly define:
   - Decision variables
   - Objective function
   - Constraints
   Carefully determine whether each decision variable is integer or continuous.
2. Build a complete mathematical model.
3. Provide Gurobi Python code that formulates and solves the model.
Ensure correctness, clarity, and professional presentation.""",
    "user": """Here is the given optimization problem:
{Question}

Reason step by step to derive the modeling process before generating the gurobipy code. When you respond, first think carefully. After thinking, output the mathematical model. Finally, output a code block beginning with:
```python
import gurobipy as gp
from gurobipy import GRB
```
The script's final output line must be: print("Result:", objective_value), where objective_value is the actual computed objective value.
/no_think""",
}


# Strategy-routing prompt retained as SDRL.
sdrl_prompt = {
    "system": """You are an expert Operations Research Engineer and Algorithm Designer.
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
   - MANDATORY: You MUST use Gurobi through gurobipy for this strategy.
   - IMPLEMENTATION: Write rigorous mathematical formulation using gurobipy. Start code with: 'import gurobipy as gp
from gurobipy import GRB'
2. Algorithmic Design (DP/Greedy/Graph):
   - CHOOSE IF: The problem exhibits optimal substructure, overlapping subproblems, or standard graph properties (e.g., Knapsack, Shortest Path, Interval Scheduling).
   - IMPLEMENTATION: Use Dynamic Programming, Greedy, Divide-and-Conquer, or Backtracking.
3. Metaheuristic / Heuristic:
   - CHOOSE IF: The problem is large-scale, highly non-linear, or a complex Combinatorial Optimization problem (NP-hard, e.g., large VRP/TSP) where exact solvers would time out.
   - IMPLEMENTATION: Use a high-performance heuristic (e.g., ALNS, GRASP, Iterated Local Search, Simulated Annealing).

### Step 3: Output Format
You must structure your response exactly as follows:
First, output exactly one strategy tag on its own first line, using one of:
<strategy>exact_solver</strategy>
<strategy>algorithmic_design</strategy>
<strategy>metaheuristic</strategy>
Then output your step-by-step reasoning (Analysis and Routing Decision). Finally, output the complete, executable Python code in a single code block starting with ```python.""",
    "user": """Solve the following algorithm design and optimization problem:
{Question}
Reason step by step to derive the logic before writing the script. After thinking, explain your algorithmic strategy. Finally, output the complete code block starting with ```python. Before the final output lines, you must assign `variables` to the computed decision variable solution (prefer a list or a dictionary of variable values). If the solution has no explicit decision-variable vector, set `variables = []` as a fallback.
The script's final output lines must be exactly:
print("Decision variables:", variables)
print("Result:", objective_value)
where `variables` is the optimal decision variable(s) and `objective_value` is the actual computed result variable.
/no_think""",
}
