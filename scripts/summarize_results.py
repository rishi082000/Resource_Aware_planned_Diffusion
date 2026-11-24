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
    "workload_class",
    "repeat",
    "gpus",
    "scheduler",
    "prefix_cache",
    "latency_seconds",
    "speedup_vs_baseline",
    "speedup_vs_1gpu_same_settings",
    "gpu_seconds",
    "tokens_generated",
    "peak_memory_gib_rank0",
    "stages",
    "total_chunks",
    "mean_chunk_length",
    "max_chunk_length",
    "chunk_length_cv",
    "chunk_lengths",
    "diffusion_seconds",
    "mean_idle_fraction",
    "load_imbalance",
    "verify_mismatches",
    "verify_tokens_compared",
    "verify_mismatch_rate",
    "verify_exact_match",
    "file",
]


def summarize_run(path: Path, workload_classes):
    # File names: baseline_p01_r1, dist_p01_g2_lpt_r1, verify_p01_g2_lpt
    parts = path.stem.split("_")
    mode, prompt_id = parts[0], parts[1]
    repeat = int(parts[-1][1:]) if parts[-1].startswith("r") and parts[-1][1:].isdigit() else None

    metrics = json.loads(path.read_text())
    distributed = metrics.get("distributed")
    stage_records = distributed["stages"] if distributed else []
    chunk_lengths = [
        block["predicted_content_length"]
        for stage in metrics["stages"]
        for block in stage["blocks"]
    ]

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
    verification_enabled = bool(distributed and distributed.get("verify"))
    verify_mismatches = (
        sum(record["verify_mismatch_count"] for record in stage_records)
        if verification_enabled
        else None
    )
    verify_tokens_compared = (
        sum(
            record.get(
                "verify_tokens_compared",
                sum(
                    length
                    for worker in record["workers"]
                    for length in worker["chunk_lengths"]
                ),
            )
            for record in stage_records
        )
        if verification_enabled
        else None
    )
    verify_mismatch_rate = (
        verify_mismatches / verify_tokens_compared
        if verification_enabled and verify_tokens_compared
        else 0.0 if verification_enabled
        else None
    )

    return {
        "mode": mode,
        "prompt_id": prompt_id,
        "workload_class": workload_classes.get(prompt_id, "unknown"),
        "repeat": repeat,
        "gpus": gpus,
        "scheduler": distributed["scheduler"] if distributed else "combined",
        "prefix_cache": bool(distributed.get("prefix_cache")) if distributed else None,
        "latency_seconds": metrics["latency_seconds"],
        "speedup_vs_baseline": None,
        "speedup_vs_1gpu_same_settings": None,
        "gpu_seconds": metrics["latency_seconds"] * gpus,
        "tokens_generated": metrics["tokens_generated"],
        "peak_memory_gib_rank0": metrics["peak_memory_allocated_gib"],
        "stages": len(metrics["stages"]),
        "total_chunks": len(chunk_lengths),
        "mean_chunk_length": statistics.mean(chunk_lengths) if chunk_lengths else None,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else None,
        "chunk_length_cv": (
            statistics.pstdev(chunk_lengths) / statistics.mean(chunk_lengths)
            if chunk_lengths and statistics.mean(chunk_lengths) > 0
            else None
        ),
        "chunk_lengths": " | ".join(
            ",".join(str(block["predicted_content_length"]) for block in stage["blocks"])
            for stage in metrics["stages"]
        ),
        "diffusion_seconds": sum(record["stage_seconds"] for record in stage_records) if stage_records else None,
        "mean_idle_fraction": statistics.mean(idle_fractions) if idle_fractions else None,
        "load_imbalance": statistics.mean(imbalances) if imbalances else None,
        "verify_mismatches": verify_mismatches,
        "verify_tokens_compared": verify_tokens_compared,
        "verify_mismatch_rate": verify_mismatch_rate,
        "verify_exact_match": verify_mismatches == 0 if verification_enabled else None,
        "file": path.name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()

    run_info_path = args.results_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    workload_classes = run_info.get("prompt_workload_classes", {})

    rows = [
        summarize_run(path, workload_classes)
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

    # The single-GPU baseline never caches the prefix, so it flatters --prefix_cache runs. This speedup instead
    # compares each run with the 1-GPU --distributed run that used the same scheduler and cache setting.
    one_gpu_latency = defaultdict(list)
    for row in rows:
        if row["mode"] == "dist" and row["gpus"] == 1:
            one_gpu_latency[(row["prompt_id"], row["scheduler"], row["prefix_cache"])].append(row["latency_seconds"])

    for row in rows:
        same_settings = one_gpu_latency.get((row["prompt_id"], row["scheduler"], row["prefix_cache"]))
        if row["mode"] == "dist" and same_settings:
            row["speedup_vs_1gpu_same_settings"] = statistics.mean(same_settings) / row["latency_seconds"]

    summary_path = args.results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} runs to {summary_path}\n")

    def geo_mean(values):
        values = [value for value in values if value]
        return f"{statistics.geometric_mean(values):.2f}x" if values else "-"

    groups = defaultdict(list)
    for row in rows:
        if row["mode"] in ("baseline", "dist"):
            groups[(row["mode"], row["gpus"], row["scheduler"], row["prefix_cache"])].append(row)

    print(
        f"{'config':<36}{'runs':>5}{'mean latency (s)':>18}{'speedup (geo-mean)':>20}"
        f"{'vs 1-GPU same settings':>24}{'mean idle':>11}"
    )
    group_order = lambda item: (item[0][0] != "baseline", item[0][1], item[0][2], bool(item[0][3]))
    for (mode, gpus, scheduler, prefix_cache), group in sorted(groups.items(), key=group_order):
        label = (
            "baseline, 1 GPU"
            if mode == "baseline"
            else f"{gpus} GPU, {scheduler}" + (", prefix cache" if prefix_cache else "")
        )
        idles = [row["mean_idle_fraction"] for row in group if row["mean_idle_fraction"] is not None]
        print(
            f"{label:<36}{len(group):>5}"
            f"{statistics.mean(row['latency_seconds'] for row in group):>18.2f}"
            f"{geo_mean(row['speedup_vs_baseline'] for row in group):>20}"
            f"{geo_mean(row['speedup_vs_1gpu_same_settings'] for row in group):>24}"
            f"{(f'{statistics.mean(idles):.0%}' if idles else '-'):>11}"
        )

    class_groups = defaultdict(list)
    for row in rows:
        if row["mode"] == "dist":
            class_groups[(row["workload_class"], row["gpus"], row["scheduler"], row["prefix_cache"])].append(row)

    if class_groups:
        print(f"\n{'workload class':<24}{'GPUs':>6}{'scheduler':>18}{'runs':>7}{'speedup (geo-mean)':>20}{'vs 1-GPU same':>16}")
        for (workload_class, gpus, scheduler, prefix_cache), group in sorted(class_groups.items()):
            scheduler_label = scheduler + (" +cache" if prefix_cache else "")
            print(
                f"{workload_class:<24}{gpus:>6}{scheduler_label:>18}{len(group):>7}"
                f"{geo_mean(row['speedup_vs_baseline'] for row in group):>20}"
                f"{geo_mean(row['speedup_vs_1gpu_same_settings'] for row in group):>16}"
            )

    verify_rows = [row for row in rows if row["mode"] == "verify"]
    if verify_rows:
        mismatched = [row for row in verify_rows if row["verify_mismatches"]]
        print(f"\nCorrectness: {len(verify_rows) - len(mismatched)}/{len(verify_rows)} verified runs match the single-GPU output")
        for row in mismatched:
            print(
                f"  {row['file']}: {row['verify_mismatches']}/"
                f"{row['verify_tokens_compared']} tokens differ "
                f"({row['verify_mismatch_rate']:.2%})"
            )

        print(f"\n{'verification class':<24}{'exact runs':>12}{'tokens different':>20}")
        verify_groups = defaultdict(list)
        for row in verify_rows:
            verify_groups[row["workload_class"]].append(row)

        for workload_class, group in sorted(verify_groups.items()):
            exact = sum(row["verify_exact_match"] for row in group)
            mismatches = sum(row["verify_mismatches"] for row in group)
            compared = sum(row["verify_tokens_compared"] for row in group)
            rate = mismatches / compared if compared else 0.0
            print(
                f"{workload_class:<24}{f'{exact}/{len(group)}':>12}"
                f"{f'{mismatches}/{compared} ({rate:.2%})':>20}"
            )


if __name__ == "__main__":
    main()
