"""
Quick start: Score a single video edit with VEFX-Reward.

Usage:
    python examples/quick_start.py \
        --original examples/sample_videos/object_removal_original.mp4 \
        --edited examples/sample_videos/object_removal_edited.mp4 \
        --instruction "Remove the woman with the grey backpack walking on the right side of the frame."

    # Or score all included samples:
    python examples/quick_start.py --run_samples
"""

import argparse
import json
import os

from vefx_reward import VEFXReward


def main():
    parser = argparse.ArgumentParser(description="Score a video edit with VEFX-Reward")
    parser.add_argument("--original", help="Path to original video")
    parser.add_argument("--edited", help="Path to edited video")
    parser.add_argument("--instruction", help="Editing instruction")
    parser.add_argument("--model", default="xiangbog/VEFX-Reward-4B", help="Model path or HF ID")
    parser.add_argument("--device", default="cuda", help="Device (cuda / cpu)")
    parser.add_argument("--run_samples", action="store_true", help="Score all included sample video pairs")
    args = parser.parse_args()

    model = VEFXReward(args.model, device=args.device)

    if args.run_samples:
        samples_dir = os.path.join(os.path.dirname(__file__), "sample_videos")
        with open(os.path.join(samples_dir, "prompts.json")) as f:
            samples = json.load(f)

        for sample in samples:
            scores = model.score(
                os.path.join(samples_dir, sample["original"]),
                os.path.join(samples_dir, sample["edited"]),
                sample["instruction"],
            )
            print(f"\n[{sample['category']}]")
            print(f"  Instruction: {sample['instruction'][:80]}...")
            print(f"  IF={scores['IF']:.2f}  RQ={scores['RQ']:.2f}  EE={scores['EE']:.2f}  Overall={scores['Overall']:.2f}")
    else:
        if not all([args.original, args.edited, args.instruction]):
            parser.error("--original, --edited, and --instruction are required (or use --run_samples)")
        scores = model.score(args.original, args.edited, args.instruction)

        print("\n" + "=" * 50)
        print("VEFX-Reward Scores")
        print("=" * 50)
        print(f"  Instructional Following (IF): {scores['IF']:.2f}")
        print(f"  Render Quality          (RQ): {scores['RQ']:.2f}")
        print(f"  Edit Exclusivity        (EE): {scores['EE']:.2f}")
        print(f"  Overall                     : {scores['Overall']:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
