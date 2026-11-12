# Training data

[Download train.parquet](https://github.com/tokentideAI/LLMMetaSolver/raw/refs/heads/main/data/train.parquet)

This release provides the cleaned training subset: **20,320 instances** in a
single Parquet file (23,280,252 bytes, approximately 22.2 MiB). The file is stored
directly in Git; no Git LFS or shard restoration is needed.

The subset was prepared by removing identified exact and near-duplicate overlaps
with the evaluation data during the training-data audit. This describes the
released subset and does not imply that previously reported models were retrained
on this cleaned version.

## Contents

| Field | Description |
| --- | --- |
| `index`, `source`, `data_source` | Record identifiers and source labels. |
| `en_question` | Optimization problem statement. |
| `en_answer` | Numeric reference objective. |
| `prompt` | Chat messages with `role` and `content`. |
| `ability` | Task/ability label. |
| `reward_model.ground_truth` | Numeric reward reference. |
| `extra_info` | Instance metadata and embedded file assets when present. |

The `data_source` counts are 15,826 `optmath`, 3,449 `optmath_file`, 1,013
`canonical`, and 32 `sft_temp`. The 3,449 file-grounded records include CSV assets
in `extra_info.asset_files`, with relative paths, contents, columns and row counts.
Materialize these assets at their specified relative paths before executing
file-grounded solutions. The released schema contains objective references;
it does not include reference solution code.

## Load the data

```bash
python -m pip install pandas pyarrow
```

From the repository root:

```python
import pandas as pd

data = pd.read_parquet("data/train.parquet")
print(len(data))  # 20320
print(data.columns.tolist())
```

SHA-256:

```text
ff1eea7d2aede88e35991b5fc73843d9eece5a87e23b81b5bca269a7f2b78794
```
