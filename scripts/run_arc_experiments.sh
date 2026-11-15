#!/usr/bin/env bash
# Runs the single-GPU baseline and the multi-GPU chunk-distribution experiments on one ARC GPU node
# and writes one metrics JSON per run, plus summary.csv, under results/arc_<run id>/.
#
# ARC batch submission from the repository root:
#   CONDA_ENV=planned-diffusion USE_FP16=1 REPEATS=1 sbatch scripts/run_arc_experiments.sh
# For the final repeated campaign, set REPEATS=3.
#
# Every setting below can be overridden from the environment, e.g.
#   GPU_COUNTS="1 2" SCHEDULERS=lpt REPEATS=3 bash scripts/run_arc_experiments.sh
#
#SBATCH --job-name=pd-multigpu
#SBATCH --partition=a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$script_dir/../eval/pd_generate.py" ]]; then
    REPO_ROOT="$(cd "$script_dir/.." && pwd)"
else
    # sbatch runs a spooled copy of this script, so fall back to the directory it was submitted from
    REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
fi
cd "$REPO_ROOT" || exit 1
if [[ ! -f eval/pd_generate.py ]]; then
    echo "error: run or submit this script from the repository root" >&2
    exit 1
fi

export MODEL_PATH="${MODEL_PATH:-dmisrael/planned-diffusion-dream7b-sft-16ep}"
export PROMPTS_FILE="${PROMPTS_FILE:-scripts/arc_prompts.txt}"
export GPU_COUNTS="${GPU_COUNTS:-1 2 4}"            # GPU counts for the multi-GPU runs; larger than the node has are skipped
export SCHEDULERS="${SCHEDULERS:-lpt round_robin}"
export VERIFY_PROMPT_IDS="${VERIFY_PROMPT_IDS:-p01 p04 p08 p12 p15}"  # one representative from each workload class
export REPEATS="${REPEATS:-1}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
export USE_FP16="${USE_FP16:-0}"                    # 0 = bfloat16, 1 = float16
export RUN_TIMEOUT="${RUN_TIMEOUT:-20m}"            # per run; a hung run is killed instead of blocking the rest of the job
CONDA_ENV="${CONDA_ENV:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}}"
OUT_DIR="${OUT_DIR:-results/arc_$RUN_ID}"
LOG_DIR="$OUT_DIR/logs"

if [[ -n "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || exit 1
fi

mkdir -p "$LOG_DIR"

if ! python -c "import torch, transformers" 2>/dev/null; then
    echo "error: torch/transformers not importable; activate the project environment first" >&2
    exit 1
fi

AVAILABLE_GPUS="${AVAILABLE_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if (( AVAILABLE_GPUS < 1 )); then
    echo "error: no GPUs visible; run this inside a GPU allocation (srun/sbatch with --gres=gpu:N)" >&2
    exit 1
fi

if ! python -c "import semantic_parallel.scheduler._scheduler_native" 2>/dev/null; then
    echo "Building the C++ scheduler extension..."
    python -c "import pybind11" 2>/dev/null || python -m pip install "pybind11>=3.0" || exit 1
    if ! python setup.py build_ext --inplace > "$LOG_DIR/build_ext.log" 2>&1; then
        echo "error: building the extension failed; see $LOG_DIR/build_ext.log" >&2
        exit 1
    fi
fi

PROMPT_IDS=()
WORKLOAD_CLASSES=()
PROMPTS=()
declare -A SEEN_PROMPT_IDS=()
declare -A PROMPT_INDEX_BY_ID=()

while IFS=$'\t' read -r prompt_id workload_class prompt; do
    [[ -z "$prompt_id" || "$prompt_id" =~ ^[[:space:]]*# ]] && continue

    if [[ ! "$prompt_id" =~ ^p[0-9]+$ || -z "$workload_class" || -z "$prompt" ]]; then
        echo "error: malformed prompt row in $PROMPTS_FILE: expected prompt_id<TAB>workload_class<TAB>prompt" >&2
        exit 1
    fi

    if [[ -n "${SEEN_PROMPT_IDS[$prompt_id]+present}" ]]; then
        echo "error: duplicate prompt id in $PROMPTS_FILE: $prompt_id" >&2
        exit 1
    fi

    SEEN_PROMPT_IDS[$prompt_id]=1
    PROMPT_INDEX_BY_ID[$prompt_id]="${#PROMPTS[@]}"
    PROMPT_IDS+=("$prompt_id")
    WORKLOAD_CLASSES+=("$workload_class")
    PROMPTS+=("$prompt")
done < "$PROMPTS_FILE"

if (( ${#PROMPTS[@]} == 0 )); then
    echo "error: no prompts found in $PROMPTS_FILE" >&2
    exit 1
fi

VERIFY_INDICES=()
for verify_prompt_id in $VERIFY_PROMPT_IDS; do
    if [[ -z "${PROMPT_INDEX_BY_ID[$verify_prompt_id]+present}" ]]; then
        echo "error: verification prompt id not found in $PROMPTS_FILE: $verify_prompt_id" >&2
        exit 1
    fi
    VERIFY_INDICES+=("${PROMPT_INDEX_BY_ID[$verify_prompt_id]}")
done

COMMON_ARGS=(--model_path "$MODEL_PATH" --temperature 0 --max_length "$MAX_LENGTH")
if [[ "$USE_FP16" == "1" ]]; then
    COMMON_ARGS+=(--use_fp16)
fi
SINGLE_GPU=(python -m eval.pd_generate "${COMMON_ARGS[@]}")
MULTI_GPU=(python -m torch.distributed.run --standalone)

MULTI_GPU_COUNTS=""
for gpus in $GPU_COUNTS; do
    if (( gpus > AVAILABLE_GPUS )); then
        echo "Skipping $gpus GPUs: only $AVAILABLE_GPUS visible"
    else
        MULTI_GPU_COUNTS="$MULTI_GPU_COUNTS $gpus"
    fi
done

python - "$OUT_DIR/run_info.json" "$PROMPTS_FILE" <<'EOF'
import json, os, platform, subprocess, sys
from pathlib import Path

import torch

def shell(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip()

prompt_rows = []
for raw_line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    prompt_id, workload_class, prompt = raw_line.split("\t", 2)
    prompt_rows.append((prompt_id, workload_class, prompt))

info = {
    "hostname": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "git_commit": shell("git rev-parse HEAD"),
    "git_uncommitted_changes": bool(shell("git status --porcelain --untracked-files=no")),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "settings": {
        key: os.environ.get(key)
        for key in ["MODEL_PATH", "GPU_COUNTS", "SCHEDULERS", "VERIFY_PROMPT_IDS", "REPEATS", "MAX_LENGTH", "USE_FP16", "RUN_TIMEOUT"]
    },
    "prompts": {prompt_id: prompt for prompt_id, _, prompt in prompt_rows},
    "prompt_workload_classes": {
        prompt_id: workload_class
        for prompt_id, workload_class, _ in prompt_rows
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
EOF

FAILURES=0

# run <metrics json> <command...>: runs one generation with a time limit, logs its output,
# and records failures without stopping the remaining runs
run() {
    local json="$1"
    shift
    local log="$LOG_DIR/$(basename "${json%.json}").log"
    local start=$SECONDS

    timeout --kill-after=60 "$RUN_TIMEOUT" "$@" --metrics_json "$json" > "$log" 2>&1
    local status=$?

    if (( status == 0 )) && [[ -f "$json" ]]; then
        echo "  ok    $(( SECONDS - start ))s  $json"
    elif [[ -f "$json" ]]; then
        # The metrics were written, so the measurement is complete; only process shutdown failed
        echo "  ok*   $(( SECONDS - start ))s  $json  (results saved, but exit status $status; see $log)"
        echo "$json exit=$status" >> "$OUT_DIR/unclean_exits.txt"
    else
        echo "  FAIL  $(( SECONDS - start ))s  $json  (exit status $status; see $log)"
        echo "$json exit=$status" >> "$OUT_DIR/failures.txt"
        FAILURES=$(( FAILURES + 1 ))
    fi
}

echo "Results: $OUT_DIR   (${#PROMPTS[@]} prompts, $AVAILABLE_GPUS GPUs visible, multi-GPU counts:${MULTI_GPU_COUNTS:- none})"

echo "== 1/3 correctness: multi-GPU output vs single-GPU output ($VERIFY_PROMPT_IDS)"
EXPECTED_VERIFY_RUNS=0
for gpus in $MULTI_GPU_COUNTS; do
    (( gpus > 1 )) || continue
    for i in "${VERIFY_INDICES[@]}"; do
        EXPECTED_VERIFY_RUNS=$(( EXPECTED_VERIFY_RUNS + 1 ))
        prompt_id="${PROMPT_IDS[$i]}"
        run "$OUT_DIR/verify_${prompt_id}_g${gpus}_lpt.json" \
            "${MULTI_GPU[@]}" --nproc_per_node="$gpus" -m eval.pd_generate "${COMMON_ARGS[@]}" \
            --distributed --scheduler lpt --verify_distributed --prompt "${PROMPTS[$i]}"
    done
done

if ! python - "$OUT_DIR" "$EXPECTED_VERIFY_RUNS" <<'EOF'
import glob, json, sys

expected = int(sys.argv[2])
paths = sorted(glob.glob(f"{sys.argv[1]}/verify_*.json"))
problems = []
for path in paths:
    stages = json.load(open(path))["distributed"]["stages"]
    mismatches = sum(stage["verify_mismatch_count"] for stage in stages)
    if mismatches:
        problems.append(f"  {path}: {mismatches} tokens differ")

if len(paths) != expected:
    print(f"ERROR: expected {expected} verification results, but found {len(paths)}")
    sys.exit(1)
elif problems:
    print("ERROR: multi-GPU output differs from single-GPU output; timing comparisons are not valid")
    print("\n".join(problems))
    sys.exit(1)
elif expected == 0:
    print("  no multi-GPU correctness checks requested")
else:
    print("  all checked stages match the single-GPU output")
EOF
then
    echo "Stopping before timing experiments because correctness verification failed"
    exit 1
fi

for (( repeat = 1; repeat <= REPEATS; repeat++ )); do
    echo "== 2/3 single-GPU baseline (repeat $repeat/$REPEATS)"
    for i in "${!PROMPTS[@]}"; do
        prompt_id="${PROMPT_IDS[$i]}"
        run "$OUT_DIR/baseline_${prompt_id}_r${repeat}.json" "${SINGLE_GPU[@]}" --prompt "${PROMPTS[$i]}"
    done

    echo "== 3/3 multi-GPU runs (repeat $repeat/$REPEATS)"
    for gpus in $MULTI_GPU_COUNTS; do
        for scheduler in $SCHEDULERS; do
            for i in "${!PROMPTS[@]}"; do
                prompt_id="${PROMPT_IDS[$i]}"
                run "$OUT_DIR/dist_${prompt_id}_g${gpus}_${scheduler}_r${repeat}.json" \
                    "${MULTI_GPU[@]}" --nproc_per_node="$gpus" -m eval.pd_generate "${COMMON_ARGS[@]}" \
                    --distributed --scheduler "$scheduler" --prompt "${PROMPTS[$i]}"
            done
        done
    done
done

python scripts/summarize_results.py "$OUT_DIR"

if [[ -f "$OUT_DIR/unclean_exits.txt" ]]; then
    echo "Some runs saved their results but did not exit cleanly; see $OUT_DIR/unclean_exits.txt"
fi

if (( FAILURES > 0 )); then
    echo "$FAILURES run(s) failed; they are listed in $OUT_DIR/failures.txt"
    exit 1
fi
