"""CPU checks for the dynamic (work-queue) scheduler: claim order, per-chunk step counters, tags, and multi-process runs."""

import json
import os
import time
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

from dream.configuration_dream import DreamConfig
from dream.control_tags import (
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    PROMISE_END,
    PROMISE_START,
    SYNC_TOKEN_ID,
)
from dream.generation_utils import DreamGenerationConfig, DreamGenerationMixin
from dream.modeling_dream import DreamModel
from dream.pd_utils import create_pd_inputs, update_attention_mask
from semantic_parallel.distributed import DistributedStageExecutor, serve_stages
from semantic_parallel.distributed.torch_runtime import _collect_tagged_results
from semantic_parallel.scheduler import dynamic_claim_order
from semantic_parallel.stage import ChunkTask, PlannedStage

# Dream digit token IDs 16, 17 and 18 represent 1, 2 and 3.
STAGE_ONE_PLAN = (PROMISE_START, 101, 17, PROMISE_END,
                  PROMISE_START, 102, 16, PROMISE_END,
                  PROMISE_START, 103, 18, PROMISE_END, SYNC_TOKEN_ID)
STAGE_TWO_PLAN = (PROMISE_START, 104, 16, PROMISE_END,
                  PROMISE_START, 105, 16, PROMISE_END, EOS_TOKEN_ID)


def make_tiny_model():
    torch.manual_seed(0)
    config = DreamConfig(
        vocab_size=151680,  # must cover Dream's special-token IDs
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        use_cache=False,
        sliding_window=None,
        use_sliding_window=False,
        max_window_layers=0,
    )
    config._attn_implementation = "sdpa"

    # float64: random weights give near-tied confidences that float32 rounding can reorder
    model = DreamModel(config).eval().to(torch.float64)
    model.generation_config = DreamGenerationConfig(
        mask_token_id=MASK_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        bos_token_id=EOS_TOKEN_ID,
    )
    return model


def prepared_config(model, steps):
    config = model._prepare_generation_config(
        model.generation_config,
        max_length=512,
        return_dict_in_generate=True,
        alg="pd_entropy",
        temperature=0.0,
        alg_temp=0.0,
        top_p=None,
    )
    config.steps = steps
    return config


def make_stage():
    prompt = torch.randint(1000, 5000, (1, 12), generator=torch.Generator().manual_seed(1))
    prefix = torch.cat([prompt, torch.tensor([STAGE_ONE_PLAN[:-1]])], dim=1)
    length = prefix.shape[1]
    full_sequence, attention_mask, _, block_info = create_pd_inputs(
        input_ids=prefix,
        prev_attention_mask=torch.tril(torch.ones(1, length, length)).bool(),
        device=prefix.device,
    )
    tasks = [
        ChunkTask(chunk_id=chunk_id, content_length=content_length, start=start, end_exclusive=end)
        for chunk_id, (start, end, content_length) in enumerate(block_info)
    ]
    return PlannedStage(
        stage_id=0,
        prefix=prefix,
        full_sequence=full_sequence,
        attention_mask=attention_mask,
        tasks=tasks,
        diffusion_steps=max(task.content_length for task in tasks),
        terminator_token_id=SYNC_TOKEN_ID,
    )


def use_scripted_planner():
    # Replace autoregressive planning with a fixed two-stage plan; everything after planning is real
    plans = iter([STAGE_ONE_PLAN, STAGE_TWO_PLAN])

    def _ar_sample(self, input_ids, attention_mask, max_length, *args, **kwargs):
        if attention_mask is None:
            length = input_ids.shape[1]
            attention_mask = torch.tril(torch.ones(1, length, length)).bool()

        for token in next(plans):
            input_ids = torch.cat([input_ids, torch.tensor([[token]])], dim=-1)
            attention_mask = update_attention_mask(attention_mask, torch.ones(1, 1, 1))

        return input_ids, attention_mask

    DreamGenerationMixin._ar_sample = _ar_sample


def generate(model, stage_executor=None):
    use_scripted_planner()
    prompt = torch.randint(1000, 5000, (1, 12), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        return model.planned_diffusion_generate(
            prompt,
            generation_config=model.generation_config,
            max_length=512,
            return_dict_in_generate=True,
            alg="pd_entropy",
            temperature=0.0,
            alg_temp=0.0,
            top_p=None,
            stage_executor=stage_executor,
        )


def test_claim_order_is_longest_first_with_ties_by_chunk_id():
    tasks = [ChunkTask(i, length, 0, 0) for i, length in enumerate([20, 120, 35, 80, 35])]

    assert dynamic_claim_order(tasks) == [1, 3, 2, 4, 0]


@pytest.fixture(scope="module")
def tiny_model():
    return make_tiny_model()


@pytest.fixture(scope="module")
def combined_stage_output(tiny_model):
    stage = make_stage()
    with torch.no_grad():
        return tiny_model._diff_sample(
            stage.full_sequence.clone(),
            attention_mask=stage.attention_mask,
            generation_config=prepared_config(tiny_model, stage.diffusion_steps),
            generation_tokens_hook_func=lambda step, tokens, logits: tokens,
            generation_logits_hook_func=lambda step, tokens, logits: logits,
            format_inputs=False,
            threshold=None,
            block_info=[(task.start, task.end_exclusive, task.content_length) for task in stage.tasks],
        ).sequences


@pytest.mark.parametrize("claim_order", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
@pytest.mark.parametrize("max_active_chunks", [1, 2, 3])
def test_chunks_joining_at_any_time_match_the_combined_stage(
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
        )

    assert sorted(finished) == [0, 1, 2]
    for task in stage.tasks:
        assert torch.equal(finished[task.chunk_id], combined_stage_output[:, task.start:task.end_exclusive])


def test_unsupported_algorithm_is_rejected(tiny_model):
    stage = make_stage()
    config = prepared_config(tiny_model, stage.diffusion_steps)
    config.alg = "pd_confidence_threshold"

    with pytest.raises(ValueError):
        tiny_model._diff_sample_dynamic(stage, lambda: None, lambda chunk_id, tokens: None, config, 2)


@pytest.fixture
def single_rank_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1)
    yield dist.distributed_c10d._get_default_store()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ({"stage_id": 5, "chunk_id": 0}, "tag mismatch"),
        ({"stage_id": 0, "chunk_id": 2}, "tag mismatch"),
        ({"tokens": [1, 2, 3]}, "returned 3 tokens"),
        ({"error": "Traceback: boom"}, "failed on rank 0"),
    ],
)
def test_rank_zero_rejects_bad_tagged_results(single_rank_group, payload, expected_error):
    stage = make_stage()
    for task in stage.tasks:
        result = {
            "stage_id": stage.stage_id,
            "chunk_id": task.chunk_id,
            "rank": 0,
            "tokens": [0] * (task.end_exclusive - task.start),
            "error": None,
        }
        if task.chunk_id == 0:
            result.update(payload)
        single_rank_group.set(f"test/chunk{task.chunk_id}", json.dumps(result))

    with pytest.raises((ValueError, RuntimeError), match=expected_error):
        _collect_tagged_results(stage, "test")


def run_rank(rank, world_size, init_file, result_file, max_active_chunks, slow_rank, failing_rank):
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

        if rank == slow_rank:
            forward = model.forward

            def slow_forward(*args, **kwargs):
                time.sleep(1.0)
                return forward(*args, **kwargs)

            model.forward = slow_forward

        if rank == failing_rank:
            diff_sample_dynamic = model._diff_sample_dynamic

            def fail_after_first_claim(stage, claim_next_chunk, on_chunk_done, *args):
                def claim_then_fail():
                    if claim_next_chunk() is not None:
                        raise ValueError("injected failure")
                    return None

                return diff_sample_dynamic(stage, claim_then_fail, on_chunk_done, *args)

            model._diff_sample_dynamic = fail_after_first_claim

        if rank != 0:
            serve_stages(model, "cpu")
            return

        executor = DistributedStageExecutor(
            model, "cpu", policy="dynamic", verify=True, max_active_chunks=max_active_chunks
        )
        try:
            output = generate(model, executor)
            torch.save({"sequences": output.sequences, "records": executor.stage_records}, result_file)
        except RuntimeError as error:
            torch.save({"error": str(error)}, result_file)
        finally:
            executor.shutdown()
    finally:
        dist.destroy_process_group()


def run_distributed(tmp_path, world_size, max_active_chunks, slow_rank=None, failing_rank=None):
    result_file = tmp_path / "result.pt"
    mp.spawn(
        run_rank,
        args=(world_size, str(tmp_path / "init"), str(result_file), max_active_chunks, slow_rank, failing_rank),
        nprocs=world_size,
        join=True,
    )
    return torch.load(result_file)


@pytest.fixture(scope="module")
def single_process_sequences():
    return generate(make_tiny_model()).sequences


@pytest.mark.parametrize(("world_size", "max_active_chunks"), [(2, 1), (2, 2), (3, 2)])
def test_dynamic_generation_matches_single_process(tmp_path, single_process_sequences, world_size, max_active_chunks):
    result = run_distributed(tmp_path, world_size, max_active_chunks)

    assert "error" not in result, result.get("error")
    assert torch.equal(result["sequences"], single_process_sequences)

    for record in result["records"]:
        assert record["verify_mismatch_count"] == 0
        claimed = sorted(chunk_id for worker in record["workers"] for chunk_id in worker["chunk_ids"])
        assert claimed == sorted(chunk["chunk_id"] for chunk in record["chunks"])
        assert len(claimed) == len(set(claimed))


def test_free_rank_claims_the_work_a_slow_rank_cannot_take(tmp_path, single_process_sequences):
    result = run_distributed(tmp_path, 2, max_active_chunks=1, slow_rank=1)

    assert torch.equal(result["sequences"], single_process_sequences)
    first_stage = result["records"][0]
    chunks_by_rank = {worker["rank"]: len(worker["chunk_ids"]) for worker in first_stage["workers"]}
    assert chunks_by_rank[0] > chunks_by_rank[1]


def test_worker_failure_is_reported_under_its_tag(tmp_path):
    result = run_distributed(tmp_path, 2, max_active_chunks=1, failing_rank=1)

    assert "failed on rank 1" in result["error"]
    assert "injected failure" in result["error"]
