# BFCL Evaluation for JaisPlus

This guide explains how to run the Berkeley Function Calling Leaderboard (BFCL) evaluation for JaisPlus models.

## Prerequisites

1. Install the BFCL package:
   ```bash
   cd berkeley-function-call-leaderboard
   pip install -e .
   ```

2. Ensure you have access to the model checkpoint.

## Running Tests

### Using the Bash Script (Recommended)

```bash
./run_bfcl_tests.sh <MODEL_PATH> <FOLDER_NAME> <TEST_CATEGORIES>
```

**Arguments:**
- `MODEL_PATH`: Path to the model checkpoint directory
- `FOLDER_NAME`: Unique name for the results folder (e.g., `experiment_v1`, `baseline_run`)
- `TEST_CATEGORIES`: Comma-separated test categories (e.g., `simple_python,parallel`)

**Note:** The model name is hardcoded as `JaisPlus`. Only the results folder name is configurable.

**Example:**
```bash
./run_bfcl_tests.sh \
  /lustre/scratch/users/ahmed.frikha/ckpts/LC_8B_SFT_baseline_251203_TC_3X/hf/checkpoint_10734_to_hf \
  baseline_v1 \
  "simple_python"
```

Results will be saved to: `/lustre/scratch/users/abhishek.maiti/BFCL_results/<FOLDER_NAME>/`

### Manual Execution

**Step 1: Generate Responses**
```bash
bfcl generate \
  --model JaisPlus \
  --test-category simple_python \
  --backend vllm \
  --num-gpus 1 \
  --local-model-path /path/to/model
```

**Step 2: Evaluate Responses**
```bash
bfcl evaluate \
  --model JaisPlus \
  --test-category simple_python
```

## Available Test Categories

**Individual categories:**
- `simple_python`, `simple_java`, `simple_javascript`
- `parallel`, `multiple`, `parallel_multiple`
- `irrelevance`
- `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`
- `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`

**Groups:**
- `all` - All test categories
- `single_turn` - All single-turn tests
- `multi_turn` - All multi-turn tests
- `live` - All live (user-contributed) tests
- `non_live` - All non-live tests

List all available categories:
```bash
bfcl test-categories
```

## Output Structure

Results are saved in the following structure:
```
/lustre/scratch/users/abhishek.maiti/BFCL_results/<FOLDER_NAME>/
├── result/     # Generated model responses
└── score/      # Evaluation scores
```
