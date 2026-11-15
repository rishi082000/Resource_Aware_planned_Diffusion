"""Multi-GPU stage execution over torch.distributed, one rank per GPU (launched with torchrun)."""
from __future__ import annotations

import dataclasses
import traceback
from time import perf_counter

import torch
import torch.distributed as dist

from semantic_parallel.scheduler import schedule_lpt, schedule_round_robin
from semantic_parallel.stage import PlannedStage

from .runtime import TaskExecution, reconstruct_outputs, validate_static_assignments

SCHEDULER_POLICIES = {
    "lpt": schedule_lpt,
    "round_robin": schedule_round_robin,
}

_STAGE = "stage"
_STOP = "stop"


def _identity_tokens(step, tokens, logits):
    return tokens


def _identity_logits(step, tokens, logits):
    return logits


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _stage_to(stage: PlannedStage, device):
    return dataclasses.replace(
        stage,
        prefix=stage.prefix.to(device),
        full_sequence=stage.full_sequence.to(device),
        attention_mask=stage.attention_mask.to(device),
    )


def _run_assigned_chunks(model, stage, chunk_ids, generation_config, threshold, device):
    # Errors are returned instead of raised so every rank still reaches the gather and nobody hangs
    try:
        _synchronize(device)
        start = perf_counter()

        with torch.no_grad():
            outputs = model._execute_chunks(
                stage,
                list(chunk_ids),
                generation_config,
                _identity_tokens,
                _identity_logits,
                threshold=threshold,
            )

        _synchronize(device)

        return {
            "outputs": {chunk_id: tokens.cpu() for chunk_id, tokens in outputs.items()},
            "compute_seconds": perf_counter() - start,
            "error": None,
        }
    except Exception:
        return {
            "outputs": {},
            "compute_seconds": 0.0,
            "error": traceback.format_exc(),
        }


class DistributedStageExecutor:
    """Rank 0 side. planned_diffusion_generate calls it once per stage instead of the combined _diff_sample."""

    def __init__(self, model, device, policy: str = "lpt", verify: bool = False):
        if policy not in SCHEDULER_POLICIES:
            raise ValueError(f"unknown scheduler policy: {policy}")

        if dist.get_rank() != 0:
            raise RuntimeError("DistributedStageExecutor must run on rank 0; other ranks call serve_stages")

        self.model = model
        self.device = device
        self.policy = policy
        self.verify = verify
        self.world_size = dist.get_world_size()
        self.stage_records = []

    def __call__(self, stage: PlannedStage, generation_config, threshold=None):
        stage_start = perf_counter()

        assignments = SCHEDULER_POLICIES[self.policy](stage.tasks, self.world_size)
        tasks_by_id = validate_static_assignments(stage.tasks, assignments)
        schedule_seconds = perf_counter() - stage_start

        request = {
            "command": _STAGE,
            "stage": _stage_to(stage, "cpu"),
            "chunk_ids": [assignment.chunk_ids for assignment in assignments],
            "generation_config": generation_config,
            "threshold": threshold,
        }
        dist.broadcast_object_list([request], src=0)
        _synchronize(self.device)
        broadcast_seconds = perf_counter() - stage_start - schedule_seconds

        own_result = _run_assigned_chunks(
            self.model,
            stage,
            assignments[0].chunk_ids,
            generation_config,
            threshold,
            self.device,
        )

        gathered = [None] * self.world_size
        dist.gather_object(own_result, gathered, dst=0)
        stage_seconds = perf_counter() - stage_start

        errors = [
            f"rank {rank}:\n{result['error']}"
            for rank, result in enumerate(gathered)
            if result["error"] is not None
        ]
        if errors:
            raise RuntimeError("chunk execution failed on " + "\n".join(errors))

        task_executions = [
            TaskExecution(
                chunk_id=chunk_id,
                worker_id=rank,
                output=tokens,
                # A rank runs all its chunks in one diffusion pass, so they share its compute time
                elapsed_seconds=result["compute_seconds"],
            )
            for rank, result in enumerate(gathered)
            for chunk_id, tokens in result["outputs"].items()
        ]
        ordered_outputs = reconstruct_outputs(task_executions, list(tasks_by_id))

        sequence = stage.full_sequence.clone()
        for chunk_id, tokens in zip(sorted(tasks_by_id), ordered_outputs):
            task = tasks_by_id[chunk_id]
            sequence[:, task.start:task.end_exclusive] = tokens.to(sequence.device)

        record = {
            "stage_id": stage.stage_id,
            "policy": self.policy,
            "world_size": self.world_size,
            "diffusion_steps": stage.diffusion_steps,
            "prefix_length": stage.prefix.shape[1],
            "schedule_seconds": schedule_seconds,
            "broadcast_seconds": broadcast_seconds,
            "stage_seconds": stage_seconds,
            "workers": [
                {
                    "rank": rank,
                    "chunk_ids": list(assignment.chunk_ids),
                    "chunk_lengths": [tasks_by_id[chunk_id].content_length for chunk_id in assignment.chunk_ids],
                    "predicted_load": assignment.estimated_load,
                    "compute_seconds": result["compute_seconds"],
                    "idle_seconds": max(0.0, stage_seconds - result["compute_seconds"]),
                }
                for rank, (assignment, result) in enumerate(zip(assignments, gathered))
            ],
        }

        if self.verify:
            record["verify_mismatch_count"] = self._count_mismatches_against_combined(
                stage, sequence, generation_config, threshold
            )
            record["verify_tokens_compared"] = sum(
                task.content_length for task in stage.tasks
            )

        self.stage_records.append(record)
        return sequence

    def _count_mismatches_against_combined(self, stage, sequence, generation_config, threshold):
        # Re-runs the single-GPU combined stage on rank 0; only meaningful at temperature 0
        with torch.no_grad():
            combined = self.model._diff_sample(
                stage.full_sequence.clone(),
                attention_mask=stage.attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=_identity_tokens,
                generation_logits_hook_func=_identity_logits,
                format_inputs=False,
                threshold=threshold,
                block_info=[
                    (task.start, task.end_exclusive, task.content_length)
                    for task in stage.tasks
                ],
            )

        combined_sequence = combined.sequences if hasattr(combined, "sequences") else combined
        return (combined_sequence != sequence).sum().item()

    def shutdown(self):
        dist.broadcast_object_list([{"command": _STOP}], src=0)


def serve_stages(model, device):
    """Ranks 1..N-1: run the chunks rank 0 assigns, stage after stage, until rank 0 sends stop."""
    rank = dist.get_rank()

    while True:
        message = [None]
        dist.broadcast_object_list(message, src=0)
        request = message[0]

        if request["command"] == _STOP:
            return

        result = _run_assigned_chunks(
            model,
            _stage_to(request["stage"], device),
            request["chunk_ids"][rank],
            request["generation_config"],
            request["threshold"],
            device,
        )
        dist.gather_object(result, None, dst=0)
