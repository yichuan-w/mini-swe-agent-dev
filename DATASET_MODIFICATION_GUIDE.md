# How to Download and Modify SWE-Bench Dataset

This guide shows you how to download a single instance from SWE-Bench, modify its problem statement, and run tests.

## Quick Start

### Step 1: Download a Single Instance

Download instance index 0 from the verified test split:

```bash
python download_and_modify_dataset.py download-single verified test 0 ./datasets
```

This will:
- Download only the instance at index 0 (not the entire dataset)
- Save it to `./datasets/verified_test_<instance_id>.json`
- Display the instance ID and file path

### Step 2: Modify the Problem Statement

Edit the JSON file directly:

```bash
# Open the file in your editor
vim ./datasets/verified_test_<instance_id>.json
# or
code ./datasets/verified_test_<instance_id>.json
```

Find the `"problem_statement"` field and modify it. Make sure to:
- Keep valid JSON format (quotes, brackets, etc.)
- Use `\n` for newlines if needed
- Escape any quotes in your text as `\"`

### Step 3: Run the Test

Run the instance with your modified problem statement:

```bash
python run_local_instance.py ./datasets/verified_test_<instance_id>.json \
    -i <instance_id> \
    -m gemini/gemini-3-pro-preview \
    -o ./output_dir
```

## Complete Example

```bash
# 1. Download instance index 0
python download_and_modify_dataset.py download-single verified test 0 ./datasets

# Output will show something like:
# Downloaded instance 0: astropy__astropy-12907
# Saved to ./datasets/verified_test_astropy__astropy-12907.json

# 2. Edit the JSON file
vim ./datasets/verified_test_astropy__astropy-12907.json
# Modify the "problem_statement" field

# 3. Run the test
python run_local_instance.py ./datasets/verified_test_astropy__astropy-12907.json \
    -i astropy__astropy-12907 \
    -m gemini/gemini-3-pro-preview \
    -o ./output_dir
```

## Parameters

### download_and_modify_dataset.py

- `subset`: Dataset subset (`verified`, `lite`, `full`, etc.)
- `split`: Dataset split (`test`, `dev`)
- `index`: Instance index (0-based)
- `output_dir`: Output directory (default: `./datasets`)

### run_local_instance.py

- `json_file`: Path to local JSON dataset file (required)
- `-i, --instance`: Instance ID (required)
- `-m, --model`: Model name (e.g., `gemini/gemini-3-pro-preview`)
- `-o, --output`: Output trajectory file path
- `-c, --config`: Config file path (optional)
- `--exit-immediately`: Exit immediately without confirmation

## Notes

- **Direct JSON Editing**: You can directly edit the JSON file - no need for separate modify commands
- **Backup**: Consider backing up the original file before editing:
  ```bash
  cp ./datasets/verified_test_<instance_id>.json ./datasets/verified_test_<instance_id>.json.backup
  ```
- **JSON Format**: Ensure the JSON remains valid after editing (matching brackets, proper quotes, etc.)
