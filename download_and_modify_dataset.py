#!/usr/bin/env python3
"""Download a single instance from SWE-Bench dataset."""

import json
import sys
from pathlib import Path

from datasets import load_dataset

# Dataset mapping (same as in swebench.py)
DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def download_single_instance(subset: str, split: str, instance_index: int, output_dir: Path):
    """Download a single instance from SWE-Bench dataset by index.
    
    Args:
        subset: Dataset subset name (e.g., 'verified', 'lite')
        split: Dataset split (e.g., 'test', 'dev')
        instance_index: Instance index (0-based)
        output_dir: Output directory
    """
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset from {dataset_path}, split {split}...")
    
    dataset = load_dataset(dataset_path, split=split)
    instances = list(dataset)
    
    if instance_index < 0 or instance_index >= len(instances):
        print(f"Error: Instance index {instance_index} out of range (0-{len(instances)-1})")
        sys.exit(1)
    
    instance = instances[instance_index]
    instance_id = instance["instance_id"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{subset}_{split}_{instance_id}.json"
    
    # Save as a list with single instance (for compatibility with run_local_instance.py)
    with open(output_file, "w") as f:
        json.dump([instance], f, indent=2)
    
    print(f"Downloaded instance {instance_index}: {instance_id}")
    print(f"Saved to {output_file}")
    print(f"\nYou can now:")
    print(f"  1. Edit the 'problem_statement' field directly in {output_file}")
    print(f"  2. Run: python run_local_instance.py {output_file} -i {instance_id} -m <model> -o <output_dir>")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python download_and_modify_dataset.py download-single <subset> <split> <index> [output_dir]")
        print("")
        print("Example:")
        print("  python download_and_modify_dataset.py download-single verified test 0 ./datasets")
        sys.exit(1)
    
    subset = sys.argv[2]
    split = sys.argv[3]
    instance_index = int(sys.argv[4])
    output_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("./datasets")
    
    download_single_instance(subset, split, instance_index, output_dir)
