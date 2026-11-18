from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_parallel.stage import ChunkTask

@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    chunk_ids: tuple[int, ...]
    estimated_load: float

def _native_module():
    try:
        from . import _scheduler_native
    except ImportError as error:
        raise RuntimeError(
            "The native scheduler extension has not been built."
        ) from error

    return _scheduler_native

def _native_tasks(native_module, tasks: Sequence[ChunkTask]):
    return [
        native_module.TaskSpec(
            chunk_id=task.chunk_id,
            estimated_cost=float(task.content_length)
        )
        for task in tasks
    ]

def _schedule(
    tasks: Sequence[ChunkTask],
    worker_count: int,
    policy_name: str,
):
    native_module = _native_module()

    native_policy = getattr(native_module, policy_name)
    native_assignments = native_policy(_native_tasks(native_module, tasks), worker_count)

    return [
        WorkerAssignment(
            worker_id=assignment.worker_id,
            chunk_ids=tuple(assignment.chunk_ids),
            estimated_load=assignment.estimated_load
        )
        for assignment in native_assignments
    ]

def schedule_round_robin(
        tasks: Sequence[ChunkTask],
        worker_count:int
):
    return _schedule(tasks, worker_count, "schedule_round_robin")

def schedule_lpt(
        tasks: Sequence[ChunkTask],
        worker_count:int
):
    return _schedule(tasks, worker_count, "schedule_lpt")

def dynamic_claim_order(
        tasks: Sequence[ChunkTask]
):
    native_module = _native_module()
    return list(native_module.dynamic_claim_order(_native_tasks(native_module, tasks)))
