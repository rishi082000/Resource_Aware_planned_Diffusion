"""Collect the per-run metrics JSONs written by run_arc_experiments.sh into summary.csv and print a speedup table.

Usage: python scripts/summarize_results.py results/arc_<run id>
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

COLUMNS = [
    "mode",
    "prompt_id",
    "repeat",
    "gpus",
    "scheduler",
    "latency_seconds",
    "speedup_vs_baseline",
    "gpu_seconds",
    "tokens_generated",
    "peak_memory_gib_rank0",
    "stages",
    "chunk_lengths",
    "diffusion_seconds",
    "mean_idle_fraction",
    "load_imbalance",
    "verify_mismatches",
    "file",
]


def summarize_run(path: Path):
    # File names: baseline_p01_r1, dist_p01_g2_lpt_r1, verify_p01_g2_lpt
    parts = path.stem.split("_")
    mode, prompt_id = parts[0], parts[1]
    repeat = int(parts[-1][1:]) if parts[-1].startswith("r") and parts[-1][1:].isdigit() else None

    metrics = json.loads(path.read_text())
    distributed = metrics.get("distributed")
    stage_records = distributed["stages"] if distributed else []

    idle_fractions = [
        worker["idle_seconds"] / record["stage_seconds"]
        for record in stage_records
        for worker in record["workers"]
        if record["stage_seconds"] > 0
    ]
    # Slowest rank over the average rank, per stage; 1.0 means perfectly balanced
    imbalances = [
        max(computes) / statistics.mean(computes)
        for computes in ([worker["compute_seconds"] for worker in record["workers"]] for record in stage_records)
        if statistics.mean(computes) > 0
    ]

    gpus = distributed["world_size"] if distributed else 1

    return {
        "mode": mode,
        "prompt_id": prompt_id,
        "repeat": repeat,
        "gpus": gpus,
        "scheduler": distributed["scheduler"] if distributed else "combined",
        "latency_seconds": metrics["latency_seconds"],
        "speedup_vs_baseline": None,
        "gpu_seconds": metrics["latency_seconds"] * gpus,
        "tokens_generated": metrics["tokens_generated"],
        "peak_memory_gib_rank0": metrics["peak_memory_allocated_gib"],
        "stages": len(metrics["stages"]),
        "chunk_lengths": " | ".join(
            ",".join(str(block["predicted_content_length"]) for block in stage["blocks"])
            for stage in metrics["stages"]
        ),
        "diffusion_seconds": sum(record["stage_seconds"] for record in stage_records) if stage_records else None,
        "mean_idle_fraction": statistics.mean(idle_fractions) if idle_fractions else None,
        "load_imbalance": statistics.mean(imbalances) if imbalances else None,
        "verify_mismatches": (
            sum(record["verify_mismatch_count"] for record in stage_records)
            if distributed and distributed.get("verify")
            else None
        ),
        "file": path.name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    rows = [
        summarize_run(path)
        for path in sorted(args.results_dir.glob("*.json"))
        if path.stem.split("_")[0] in ("baseline", "dist", "verify")
    ]

    baseline_latency = defaultdict(list)
    for row in rows:
        if row["mode"] == "baseline":
            baseline_latency[row["prompt_id"]].append(row["latency_seconds"])

    # Verify runs also re-run each stage on one GPU, so their latency is not a timing result
    for row in rows:
        if row["mode"] == "dist" and baseline_latency[row["prompt_id"]]:
            row["speedup_vs_baseline"] = statistics.mean(baseline_latency[row["prompt_id"]]) / row["latency_seconds"]

    summary_path = args.results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} runs to {summary_path}\n")

    groups = defaultdict(list)
    for row in rows:
        if row["mode"] in ("baseline", "dist"):
            groups[(row["mode"], row["gpus"], row["scheduler"])].append(row)

    print(f"{'config':<28}{'runs':>5}{'mean latency (s)':>18}{'speedup (geo-mean)':>20}{'mean idle':>11}")
    for (mode, gpus, scheduler), group in sorted(groups.items(), key=lambda item: (item[0][0] != "baseline", item[0][1], item[0][2])):
        label = "baseline, 1 GPU" if mode == "baseline" else f"{gpus} GPU, {scheduler}"
        speedups = [row["speedup_vs_baseline"] for row in group if row["speedup_vs_baseline"]]
        idles = [row["mean_idle_fraction"] for row in group if row["mean_idle_fraction"] is not None]
        print(
            f"{label:<28}{len(group):>5}"
            f"{statistics.mean(row['latency_seconds'] for row in group):>18.2f}"
            f"{(f'{statistics.geometric_mean(speedups):.2f}x' if speedups else '-'):>20}"
            f"{(f'{statistics.mean(idles):.0%}' if idles else '-'):>11}"
        )

    verify_rows = [row for row in rows if row["mode"] == "verify"]
    if verify_rows:
        mismatched = [row for row in verify_rows if row["verify_mismatches"]]
        print(f"\nCorrectness: {len(verify_rows) - len(mismatched)}/{len(verify_rows)} verified runs match the single-GPU output")
        for row in mismatched:
            print(f"  {row['file']}: {row['verify_mismatches']} tokens differ")


if __name__ == "__main__":
    main()
