"""CPU checks for prefix caching: the prefix runs once per stage, and cached runs match uncached runs exactly."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("transformers")
pytest.importorskip(
    "semantic_parallel.scheduler._scheduler_native",
    reason="native scheduler extension has not been built",
)

from dream.control_tags import ASYNC_START
from dream.generation_utils import build_chunk_inputs_after_prefix, build_chunk_subset_inputs
from semantic_parallel.distributed import DistributedStageExecutor, serve_stages
from test_dynamic_scheduler import generate, make_stage, make_tiny_model, prepared_config


def identity_tokens(step, tokens, logits):
    return tokens


def identity_logits(step, tokens, logits):
    return logits


@pytest.fixture(scope="module")
def tiny_model():
    return make_tiny_model()


@pytest.fixture(scope="module")
def combined_stage_output(tiny_model):
    # The uncached single-GPU stage: every cached run must reproduce it token for token
    stage = make_stage()
    with torch.no_grad():
        return tiny_model._diff_sample(
            stage.full_sequence.clone(),
            attention_mask=stage.attention_mask,
            generation_config=prepared_config(tiny_model, stage.diffusion_steps),
            generation_tokens_hook_func=identity_tokens,
            generation_logits_hook_func=identity_logits,
            format_inputs=False,
            threshold=None,
            block_info=[(task.start, task.end_exclusive, task.content_length) for task in stage.tasks],
        ).sequences


def test_prefix_cache_holds_every_layer_for_the_prefix_only(tiny_model):
    stage = make_stage()
    prefix_length = stage.prefix.shape[1]

    with torch.no_grad():
        cache = tiny_model._build_prefix_cache(stage)

    assert len(cache) == tiny_model.config.num_hidden_layers
    for keys, values in cache:
        assert keys.shape[1] == prefix_length
        assert values.shape[1] == prefix_length


def test_cached_inputs_drop_the_prefix_but_keep_its_positions():
    stage = make_stage()
    prefix_length = stage.prefix.shape[1]
    tasks = [stage.tasks[0], stage.tasks[2]]

    full_ids, full_mask, full_positions, full_blocks = build_chunk_subset_inputs(stage, tasks)
    input_ids, attention_mask, position_ids, block_info = build_chunk_inputs_after_prefix(stage, tasks)

    assert torch.equal(input_ids, full_ids[:, prefix_length:])
    assert attention_mask.shape[-2:] == (input_ids.shape[1], full_ids.shape[1])
    assert torch.equal(position_ids, full_positions)
    assert block_info == [(start - prefix_length, end - prefix_length, length) for start, end, length in full_blocks]
    # Each block opens with its own <async> tag, so no chunk ever needs a logit from a prefix position
    for start, _, _ in block_info:
        assert input_ids[0, start].item() == ASYNC_START


@pytest.mark.parametrize("chunk_ids", [[0], [1], [2], [0, 2], [0, 1, 2]])
def test_static_chunks_with_prefix_cache_match_the_combined_stage(tiny_model, combined_stage_output, chunk_ids):
    stage = make_stage()

    with torch.no_grad():
        outputs = tiny_model._execute_chunks(
            stage,
            chunk_ids,
            prepared_config(tiny_model, stage.diffusion_steps),
            identity_tokens,
            identity_logits,
            use_prefix_cache=True,
        )

    assert sorted(outputs) == sorted(chunk_ids)
    for chunk_id in chunk_ids:
        task = stage.tasks[chunk_id]
        assert torch.equal(outputs[chunk_id], combined_stage_output[:, task.start:task.end_exclusive])


@pytest.mark.parametrize("claim_order", [[0, 1, 2], [2, 0, 1]])
@pytest.mark.parametrize("max_active_chunks", [1, 2, 3])
def test_dynamic_chunks_with_prefix_cache_match_the_combined_stage(
    tiny_model, combined_stage_output, claim_order, max_active_chunks
):
    stage = make_stage()
    queue = list(claim_order)
    finished = {}

    with torch.no_grad():
        tiny_model._diff_sample_dynamic(
            stage,
            lambda: queue.pop(0) if queue else None,
            lambda chunk_id, tokens: finished.setdefault(chunk_id, tokens),
            prepared_config(tiny_model, stage.diffusion_steps),
            max_active_chunks,
            use_prefix_cache=True,
        )

    assert sorted(finished) == [0, 1, 2]
    for task in stage.tasks:
        assert torch.equal(finished[task.chunk_id], combined_stage_output[:, task.start:task.end_exclusive])


@pytest.mark.parametrize("path", ["static", "dynamic"])
def test_prefix_runs_once_and_each_step_feeds_only_chunk_tokens(tiny_model, monkeypatch, path):
    stage = make_stage()
    prefix_length = stage.prefix.shape[1]
    chunk_tokens = sum(task.end_exclusive - task.start for task in stage.tasks)
    calls = []
    base_forward = tiny_model.model.forward

    def recording_forward(*args, **kwargs):
        calls.append((kwargs["input_ids"].shape[1], kwargs.get("past_key_values") is not None, bool(kwargs.get("use_cache"))))
        return base_forward(*args, **kwargs)

    monkeypatch.setattr(tiny_model.model, "forward", recording_forward)
    config = prepared_config(tiny_model, stage.diffusion_steps)

    with torch.no_grad():
        if path == "static":
            tiny_model._execute_chunks(stage, [0, 1, 2], config, identity_tokens, identity_logits, use_prefix_cache=True)
        else:
            queue = [0, 1, 2]
            tiny_model._diff_sample_dynamic(
                stage, lambda: queue.pop(0) if queue else None, lambda chunk_id, tokens: None, config, 3,
                use_prefix_cache=True,
            )

    prefix_passes = [call for call in calls if call[2]]
    steps = [call for call in calls if not call[2]]
    assert prefix_passes == [(prefix_length, False, True)]
    # The dynamic loop claims one chunk first, so chunks 1 and 2 join one step later and finish one step later
    assert len(steps) == stage.diffusion_steps + (1 if path == "dynamic" else 0)
    assert all(has_cache and length <= chunk_tokens for length, has_cache, _ in steps)


def run_rank(rank, world_size, init_file, result_file, policy):
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // world_size))
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=3),
    )

    try:
        model = make_tiny_model()

        if rank != 0:
            serve_stages(model, "cpu")
            return

        executor = DistributedStageExecutor(
            model, "cpu", policy=policy, verify=True, max_active_chunks=2, prefix_cache=True
        )
        try:
            output = generate(model, executor)
            torch.save({"sequences": output.sequences, "records": executor.stage_records}, result_file)
        finally:
            executor.shutdown()
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def single_process_sequences():
    # Uncached, single process: the reference the cached multi-process runs must reproduce
    return generate(make_tiny_model()).sequences


@pytest.mark.parametrize("policy", ["lpt", "dynamic"])
def test_distributed_generation_with_prefix_cache_matches_single_process(tmp_path, single_process_sequences, policy):
    result_file = tmp_path / "result.pt"
    mp.spawn(
        run_rank,
        args=(2, str(tmp_path / "init"), str(result_file), policy),
        nprocs=2,
        join=True,
    )
    result = torch.load(result_file)

    assert torch.equal(result["sequences"], single_process_sequences)
    for record in result["records"]:
        assert record["prefix_cache"] is True
        assert record["verify_mismatch_count"] == 0
