"""Multi-GPU stage execution over torch.distributed, one rank per GPU (launched with torchrun)."""
from __future__ import annotations

import dataclasses
import json
import time
import traceback
import uuid
from time import perf_counter

import torch
import torch.distributed as dist

from semantic_parallel.scheduler import dynamic_claim_order, schedule_lpt, schedule_round_robin
from semantic_parallel.stage import PlannedStage

from .runtime import TaskExecution, reconstruct_outputs, validate_static_assignments

SCHEDULER_POLICIES = {
    "lpt": schedule_lpt,
    "round_robin": schedule_round_robin,
}

# Workers claim free chunks from a shared queue instead of receiving a fixed assignment
DYNAMIC_POLICY = "dynamic"
DYNAMIC_ALGS = ("pd_entropy", "pd_maskgit_plus", "pd_topk_margin")

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


def _run_assigned_chunks(model, stage, chunk_ids, generation_config, threshold, device, use_prefix_cache=False):
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
                use_prefix_cache=use_prefix_cache,
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


def _default_store():
    # torchrun's rendezvous key-value store: shared by all ranks, with atomic add, set and blocking get
    return dist.distributed_c10d._get_default_store()


def _run_claimed_chunks(model, stage, order, key_prefix, max_active_chunks, generation_config, use_prefix_cache=False):
    """Every rank, rank 0 included: claim free chunks from the shared queue and publish each result under its tag."""
    store = _default_store()
    rank = dist.get_rank()
    claimed_at = {}
    published = set()

    def claim_next_chunk():
        # store.add is atomic across ranks, so every queue position goes to exactly one rank
        position = store.add(f"{key_prefix}/next", 1) - 1
        if position >= len(order):
            return None

        chunk_id = order[position]
        claimed_at[chunk_id] = time.time()
        return chunk_id

    def publish(chunk_id, tokens, error=None):
        store.set(f"{key_prefix}/chunk{chunk_id}", json.dumps({
            "stage_id": stage.stage_id,
            "chunk_id": chunk_id,
            "rank": rank,
            "tokens": tokens,
            "claimed_at": claimed_at.get(chunk_id),
            "finished_at": time.time(),
            "error": error,
        }))
        published.add(chunk_id)

    try:
        with torch.no_grad():
            model._diff_sample_dynamic(
                stage,
                claim_next_chunk,
                lambda chunk_id, tokens: publish(chunk_id, tokens[0].tolist()),
                generation_config,
                max_active_chunks,
                use_prefix_cache=use_prefix_cache,
            )
    except Exception:
        # Report the failure under every tag this rank claimed but did not finish, so rank 0 never waits forever
        error = traceback.format_exc()
        for chunk_id in claimed_at:
            if chunk_id not in published:
                publish(chunk_id, None, error)


def _collect_tagged_results(stage, key_prefix):
    """Rank 0: wait for every chunk's tagged result, check the tag, and fail loudly on worker errors."""
    store = _default_store()
    results = {}

    for task in stage.tasks:
        # Blocks until the rank that claimed this chunk publishes it
        result = json.loads(store.get(f"{key_prefix}/chunk{task.chunk_id}"))

        if result["error"] is not None:
            raise RuntimeError(f"chunk {task.chunk_id} failed on rank {result['rank']}:\n{result['error']}")

        if (result["stage_id"], result["chunk_id"]) != (stage.stage_id, task.chunk_id):
            raise ValueError(
                f"tag mismatch: expected stage {stage.stage_id} chunk {task.chunk_id}, "
                f"got stage {result['stage_id']} chunk {result['chunk_id']}"
            )

        if len(result["tokens"]) != task.end_exclusive - task.start:
            raise ValueError(
                f"chunk {task.chunk_id} returned {len(result['tokens'])} tokens, "
                f"expected {task.end_exclusive - task.start}"
            )

        results[task.chunk_id] = result

    return results


class DistributedStageExecutor:
    """Rank 0 side. planned_diffusion_generate calls it once per stage instead of the combined _diff_sample."""

    def __init__(
        self,
        model,
        device,
        policy: str = "lpt",
        verify: bool = False,
        max_active_chunks: int = 2,
        prefix_cache: bool = False,
    ):
        if policy not in SCHEDULER_POLICIES and policy != DYNAMIC_POLICY:
            raise ValueError(f"unknown scheduler policy: {policy}")

        if max_active_chunks < 1:
            raise ValueError("max_active_chunks must be at least 1")

        if dist.get_rank() != 0:
            raise RuntimeError("DistributedStageExecutor must run on rank 0; other ranks call serve_stages")

        self.model = model
        self.device = device
        self.policy = policy
        self.verify = verify
        self.world_size = dist.get_world_size()
        self.max_active_chunks = max_active_chunks
        # Each rank runs the stage's prefix once and reuses its keys and values on every diffusion step
        self.prefix_cache = prefix_cache
        # Keeps this run's queue and result keys apart from any other run sharing the store
        self.run_id = uuid.uuid4().hex[:12]
        self.stage_records = []

    def __call__(self, stage: PlannedStage, generation_config, threshold=None):
        if self.policy == DYNAMIC_POLICY:
            return self._run_dynamic_stage(stage, generation_config, threshold)

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
            "prefix_cache": self.prefix_cache,
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
            use_prefix_cache=self.prefix_cache,
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
            "prefix_cache": self.prefix_cache,
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

        return self._finish_stage(record, stage, sequence, generation_config, threshold)

    def _run_dynamic_stage(self, stage: PlannedStage, generation_config, threshold):
        # Checked here, before any rank starts claiming, so an unsupported algorithm cannot leave rank 0 waiting
        if generation_config.alg not in DYNAMIC_ALGS:
            raise ValueError(f"the dynamic scheduler does not support alg={generation_config.alg}")

        stage_start = perf_counter()
        wall_start = time.time()

        order = dynamic_claim_order(stage.tasks)
        key_prefix = f"pd/{self.run_id}/stage{stage.stage_id}"
        schedule_seconds = perf_counter() - stage_start

        request = {
            "command": _STAGE,
            "mode": DYNAMIC_POLICY,
            "stage": _stage_to(stage, "cpu"),
            "order": order,
            "key_prefix": key_prefix,
            "max_active_chunks": self.max_active_chunks,
            "generation_config": generation_config,
            "prefix_cache": self.prefix_cache,
        }
        dist.broadcast_object_list([request], src=0)
        _synchronize(self.device)
        broadcast_seconds = perf_counter() - stage_start - schedule_seconds

        _run_claimed_chunks(
            self.model,
            stage,
            order,
            key_prefix,
            self.max_active_chunks,
            generation_config,
            use_prefix_cache=self.prefix_cache,
        )

        results = _collect_tagged_results(stage, key_prefix)
        stage_seconds = perf_counter() - stage_start

        tasks_by_id = {task.chunk_id: task for task in stage.tasks}
        sequence = stage.full_sequence.clone()
        for chunk_id, result in results.items():
            task = tasks_by_id[chunk_id]
            sequence[:, task.start:task.end_exclusive] = torch.tensor(
                result["tokens"], dtype=sequence.dtype, device=sequence.device
            )

        chunk_ids_by_rank = {rank: [] for rank in range(self.world_size)}
        for chunk_id in sorted(results, key=lambda chunk_id: results[chunk_id]["claimed_at"]):
            chunk_ids_by_rank[results[chunk_id]["rank"]].append(chunk_id)

        workers = []
        for rank, chunk_ids in chunk_ids_by_rank.items():
            # A rank is busy from its first claim until its last chunk finishes
            busy_seconds = (
                max(results[chunk_id]["finished_at"] for chunk_id in chunk_ids)
                - min(results[chunk_id]["claimed_at"] for chunk_id in chunk_ids)
                if chunk_ids
                else 0.0
            )
            workers.append({
                "rank": rank,
                "chunk_ids": chunk_ids,
                "chunk_lengths": [tasks_by_id[chunk_id].content_length for chunk_id in chunk_ids],
                "compute_seconds": busy_seconds,
                "idle_seconds": max(0.0, stage_seconds - busy_seconds),
            })

        record = {
            "stage_id": stage.stage_id,
            "policy": self.policy,
            "world_size": self.world_size,
            "diffusion_steps": stage.diffusion_steps,
            "prefix_length": stage.prefix.shape[1],
            "prefix_cache": self.prefix_cache,
            "schedule_seconds": schedule_seconds,
            "broadcast_seconds": broadcast_seconds,
            "stage_seconds": stage_seconds,
            "max_active_chunks": self.max_active_chunks,
            "claim_order": order,
            "workers": workers,
            # Per-chunk timeline relative to the stage start, e.g. for a Gantt chart
            "chunks": [
                {
                    "chunk_id": chunk_id,
                    "rank": results[chunk_id]["rank"],
                    "content_length": tasks_by_id[chunk_id].content_length,
                    "claimed_seconds": results[chunk_id]["claimed_at"] - wall_start,
                    "finished_seconds": results[chunk_id]["finished_at"] - wall_start,
                }
                for chunk_id in order
            ],
        }

        return self._finish_stage(record, stage, sequence, generation_config, threshold)

    def _finish_stage(self, record, stage, sequence, generation_config, threshold):
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
    """Ranks 1..N-1: run the chunks rank 0 assigns (or claim them, under the dynamic policy) until rank 0 sends stop."""
    rank = dist.get_rank()

    while True:
        message = [None]
        dist.broadcast_object_list(message, src=0)
        request = message[0]

        if request["command"] == _STOP:
            return

        if request.get("mode") == DYNAMIC_POLICY:
            _run_claimed_chunks(
                model,
                _stage_to(request["stage"], device),
                request["order"],
                request["key_prefix"],
                request["max_active_chunks"],
                request["generation_config"],
                use_prefix_cache=request.get("prefix_cache", False),
            )
            continue

        result = _run_assigned_chunks(
            model,
            _stage_to(request["stage"], device),
            request["chunk_ids"][rank],
            request["generation_config"],
            request["threshold"],
            device,
            use_prefix_cache=request.get("prefix_cache", False),
        )
        dist.gather_object(result, None, dst=0)
