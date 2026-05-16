"""
Batch scoring: Evaluate multiple video edits from a CSV file.

Expected CSV format:
    original_video,edited_video,instruction
    path/to/orig1.mp4,path/to/edit1.mp4,"make it snowy"
    path/to/orig2.mp4,path/to/edit2.mp4,"add a red hat"

Usage:
    python examples/batch_scoring.py \
        --csv edits.csv \
        --output results.csv
"""

import argparse
import csv

import torch
from vefx_reward import VEFXReward


def main():
    parser = argparse.ArgumentParser(description="Batch score video edits")
    parser.add_argument("--csv", required=True, help="Input CSV with columns: original_video, edited_video, instruction")
    parser.add_argument("--output", default="results.csv", help="Output CSV path")
    parser.add_argument("--model", default="xiangbog/VEFX-Reward-4B",
                        help='Model path, HF ID, or alias ("4B" / "32B")')
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = VEFXReward(args.model, device=args.device)

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} samples from {args.csv}")

    results = []
    for i, row in enumerate(rows):
        scores = model.score(row["original_video"], row["edited_video"], row["instruction"])
        results.append({**row, **scores})
        print(f"[{i+1}/{len(rows)}] IF={scores['IF']:.2f}  RQ={scores['RQ']:.2f}  EE={scores['EE']:.2f}  Overall={scores['Overall']:.2f}")

    fieldnames = list(rows[0].keys()) + ["IF", "RQ", "EE", "Overall"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
