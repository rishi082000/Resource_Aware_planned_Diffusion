"""Tests for machine-readable scheduler simulation metrics."""

import json
import math
from dataclasses import replace

import pytest

from semantic_parallel.distributed import (
    StageExecution,
    TaskExecution,
    WorkerExecution,
)
from semantic_parallel.profiling import (
    SimulationResult,
    simulation_result_to_dict,
    write_simulation_json,
)


def make_simulation_result() -> SimulationResult:
    task_execution = TaskExecution(
        chunk_id=0,
        worker_id=0,
        output="result-0-from-0",
        elapsed_seconds=0.02,
    )
    worker_execution = WorkerExecution(
        worker_id=0,
        assigned_chunk_ids=(0,),
        predicted_load=20.0,
        task_executions=(task_execution,),
        compute_seconds=0.02,
        idle_seconds=0.001,
    )
    stage_execution = StageExecution(
        worker_executions=(worker_execution,),
        ordered_outputs=("result-0-from-0",),
        elapsed_seconds=0.021,
    )
    return SimulationResult(
        policy_name="schedule_round_robin",
        worker_count=1,
        seconds_per_cost=0.001,
        sequential_seconds=0.02,
        measured_speedup=0.95,
        stage_execution=stage_execution,
    )


def test_simulation_result_to_dict_preserves_nested_metrics():
    metrics = simulation_result_to_dict(make_simulation_result())

    assert metrics["policy_name"] == "schedule_round_robin"
    assert metrics["worker_count"] == 1
    assert metrics["stage_execution"]["ordered_outputs"] == (
        "result-0-from-0",
    )
    worker = metrics["stage_execution"]["worker_executions"][0]
    assert worker["assigned_chunk_ids"] == (0,)
    assert worker["predicted_load"] == 20.0
    assert worker["task_executions"][0]["chunk_id"] == 0


def test_write_simulation_json_creates_directories_and_valid_json(tmp_path):
    output_path = tmp_path / "nested" / "simulation.json"

    returned_path = write_simulation_json(
        make_simulation_result(),
        output_path,
    )

    assert returned_path == output_path
    with output_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    assert metrics["policy_name"] == "schedule_round_robin"
    assert metrics["stage_execution"]["ordered_outputs"] == [
        "result-0-from-0"
    ]
    assert metrics["stage_execution"]["worker_executions"][0][
        "assigned_chunk_ids"
    ] == [0]


@pytest.mark.parametrize("invalid_value", [math.inf, -math.inf, math.nan])
def test_write_simulation_json_rejects_nonfinite_metrics(
    tmp_path,
    invalid_value,
):
    invalid_result = replace(
        make_simulation_result(),
        measured_speedup=invalid_value,
    )

    with pytest.raises(ValueError):
        write_simulation_json(
            invalid_result,
            tmp_path / "simulation.json",
        )
