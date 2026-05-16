"""
Multi-GPU parallel scoring using subprocess workers.

Splits a CSV of video edits across multiple GPUs for faster inference.

Usage:
    python examples/multi_gpu_scoring.py \
        --csv edits.csv \
        --output results.csv \
        --num_gpus 4
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile


def worker_main(args):
    """Single-GPU worker: load model, score shard, write results."""
    import torch
    from vefx_reward import VEFXReward

    with open(args.shard_file) as f:
        shard = json.load(f)

    model = VEFXReward(args.model, device="cuda:0")

    results = []
    for i, item in enumerate(shard):
        try:
            scores = model.score(item["original_video"], item["edited_video"], item["instruction"])
            results.append({**item, **scores})
            print(f"[GPU {args.gpu_id}] [{i+1}/{len(shard)}] "
                  f"IF={scores['IF']:.2f} RQ={scores['RQ']:.2f} EE={scores['EE']:.2f}", flush=True)
        except Exception as e:
            print(f"[GPU {args.gpu_id}] [{i+1}/{len(shard)}] ERROR: {e}", flush=True)
            results.append({**item, "IF": None, "RQ": None, "EE": None, "Overall": None, "error": str(e)})

    with open(args.output_file, "w") as f:
        json.dump(results, f)
    print(f"[GPU {args.gpu_id}] Done — {len(results)} results", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU video edit scoring")
    parser.add_argument("--csv", required=True, help="Input CSV")
    parser.add_argument("--output", default="results.csv", help="Output CSV")
    parser.add_argument("--model", default="xiangbog/VEFX-Reward-4B",
                        help='Model path, HF ID, or alias ("4B" / "32B")')
    parser.add_argument("--num_gpus", type=int, default=4)
    # Internal worker args
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu_id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard_file", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--output_file", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        worker_main(args)
        return

    # --- Launcher mode ---
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} samples, distributing across {args.num_gpus} GPUs")

    items = [dict(row) for row in rows]
    shards = [[] for _ in range(args.num_gpus)]
    for i, item in enumerate(items):
        shards[i % args.num_gpus].append(item)

    tmpdir = tempfile.mkdtemp(prefix="vefx_multi_")
    script = os.path.abspath(__file__)
    procs = []
    for gid in range(args.num_gpus):
        if not shards[gid]:
            continue
        sf = os.path.join(tmpdir, f"shard_{gid}.json")
        of = os.path.join(tmpdir, f"result_{gid}.json")
        with open(sf, "w") as f:
            json.dump(shards[gid], f)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["TOKENIZERS_PARALLELISM"] = "false"
        p = subprocess.Popen(
            [sys.executable, script,
             "--_worker", "--gpu_id", str(gid),
             "--shard_file", sf, "--output_file", of,
             "--model", args.model],
            env=env, stdout=sys.stdout, stderr=sys.stderr,
        )
        procs.append((p, of))

    for p, _ in procs:
        p.wait()

    # Merge results
    all_results = []
    for _, of in procs:
        if os.path.exists(of):
            with open(of) as f:
                all_results.extend(json.load(f))

    fieldnames = list(rows[0].keys()) + ["IF", "RQ", "EE", "Overall"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nAll done — {len(all_results)} results saved to {args.output}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
