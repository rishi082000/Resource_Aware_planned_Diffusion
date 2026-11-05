from __future__ import annotations

import math
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, asdict
from time import sleep
from pathlib import Path
from semantic_parallel.distributed import (StageExecution, run_local_static_stage)
from semantic_parallel.scheduler import WorkerAssignment
from semantic_parallel.stage import ChunkTask

SchedulerPolicy = Callable[
    [Sequence[ChunkTask], int],
    list[WorkerAssignment]
]

@dataclass(frozen=True)
class SimulationResult:
    policy_name: str
    worker_count: int
    seconds_per_cost: float
    sequential_seconds: float
    measured_speedup: float
    stage_execution: StageExecution[str]

def make_sleep_executor(seconds_per_cost: float):
    if(not math.isfinite(seconds_per_cost) or seconds_per_cost < 0):
        raise ValueError("seconds per cost must be finite and non negative")

    def execute_task(task: ChunkTask, worker_id: int):
        sleep(task.content_length * seconds_per_cost)
        return f"result-{task.chunk_id}-from-{worker_id}"


    return execute_task

def run_scheduler_simulation(tasks: Sequence[ChunkTask], worker_count: int, scheduler_policy: SchedulerPolicy, seconds_per_cost: float = 0.001):
    execute_task = make_sleep_executor(seconds_per_cost)
    assignments = scheduler_policy(tasks, worker_count)

    stage_execution = run_local_static_stage(
        tasks=tasks,
        assignments=assignments,
        execute_task=execute_task
    )

    sequential_seconds = sum(task.content_length * seconds_per_cost for task in tasks)

    measured_speedup = (
        sequential_seconds / stage_execution.elapsed_seconds
        if stage_execution.elapsed_seconds > 0
        else 0.0
    )

    return SimulationResult(
        policy_name=scheduler_policy.__name__,
        worker_count=worker_count,
        seconds_per_cost=seconds_per_cost,
        sequential_seconds=sequential_seconds,
        measured_speedup=measured_speedup,
        stage_execution=stage_execution
    )


def simulation_result_to_dict(result: SimulationResult):
    return asdict(result)

def write_simulation_json(result: SimulationResult, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            simulation_result_to_dict(result),
            output_file,
            indent=2,
            allow_nan=False
        )

    return output_path
