"""CPU checks that diffusing a subset of chunks matches the combined stage."""

import pytest
import torch

pytest.importorskip("transformers")

from dream.configuration_dream import DreamConfig
from dream.control_tags import (
    EOS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    PROMISE_END,
    PROMISE_START,
)
from dream.generation_utils import DreamGenerationConfig, build_chunk_subset_inputs
from dream.modeling_dream import DreamModel
from dream.pd_utils import create_pd_inputs
from semantic_parallel.stage import ChunkTask, PlannedStage

# Dream digit token IDs 16, 17 and 18 represent 1, 2 and 3.
PLAN = (PROMISE_START, 101, 17, PROMISE_END,
        PROMISE_START, 102, 16, PROMISE_END,
        PROMISE_START, 103, 18, PROMISE_END)


def make_stage() -> PlannedStage:
    prompt = torch.randint(1000, 5000, (1, 12), generator=torch.Generator().manual_seed(1))
    prefix = torch.cat([prompt, torch.tensor([PLAN])], dim=1)
    prefix_length = prefix.shape[1]
    causal_mask = torch.tril(torch.ones(1, prefix_length, prefix_length)).bool()

    full_sequence, attention_mask, _, block_info = create_pd_inputs(
        input_ids=prefix,
        prev_attention_mask=causal_mask,
        device=prefix.device,
    )

    tasks = [
        ChunkTask(
            chunk_id=chunk_id,
            content_length=content_length,
            start=start,
            end_exclusive=end,
        )
        for chunk_id, (start, end, content_length) in enumerate(block_info)
    ]

    return PlannedStage(
        stage_id=0,
        prefix=prefix,
        full_sequence=full_sequence,
        attention_mask=attention_mask,
        tasks=tasks,
        diffusion_steps=max(task.content_length for task in tasks),
        terminator_token_id=EOS_TOKEN_ID,
    )


def identity_tokens(step, tokens, logits):
    return tokens


def identity_logits(step, tokens, logits):
    return logits


@pytest.fixture(scope="module")
def tiny_model():
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


@pytest.fixture(scope="module")
def combined_run(tiny_model):
    stage = make_stage()
    generation_config = tiny_model._prepare_generation_config(
        tiny_model.generation_config,
        max_length=512,
        return_dict_in_generate=True,
        alg="pd_entropy",
        temperature=0.0,
        alg_temp=0.0,
        top_p=None,
    )
    generation_config.steps = stage.diffusion_steps

    with torch.no_grad():
        combined = tiny_model._diff_sample(
            stage.full_sequence.clone(),
            attention_mask=stage.attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=identity_tokens,
            generation_logits_hook_func=identity_logits,
            format_inputs=False,
            threshold=None,
            block_info=[
                (task.start, task.end_exclusive, task.content_length)
                for task in stage.tasks
            ],
        ).sequences

    return stage, generation_config, combined


def test_subset_inputs_keep_original_positions_and_compact_offsets():
    stage = make_stage()
    prefix_length = stage.prefix.shape[1]
    first, _, third = stage.tasks
    first_length = first.end_exclusive - first.start
    third_length = third.end_exclusive - third.start

    input_ids, attention_mask, position_ids, block_info = build_chunk_subset_inputs(stage, [first, third])

    expected_indices = torch.cat([
        torch.arange(prefix_length),
        torch.arange(first.start, first.end_exclusive),
        torch.arange(third.start, third.end_exclusive),
    ])

    assert torch.equal(position_ids[0], expected_indices)
    assert torch.equal(input_ids[0], stage.full_sequence[0, expected_indices])
    assert torch.equal(attention_mask, stage.attention_mask[:, expected_indices][:, :, expected_indices])
    assert block_info == [
        (prefix_length, prefix_length + first_length, first.content_length),
        (prefix_length + first_length, prefix_length + first_length + third_length, third.content_length),
    ]


@pytest.mark.parametrize("chunk_ids", [[0], [1], [0, 2], [2, 0], [0, 1, 2]])
def test_execute_chunks_matches_combined_stage(tiny_model, combined_run, chunk_ids):
    stage, generation_config, combined = combined_run

    with torch.no_grad():
        outputs = tiny_model._execute_chunks(
            stage,
            chunk_ids,
            generation_config,
            identity_tokens,
            identity_logits,
        )

    assert sorted(outputs) == sorted(chunk_ids)
    for chunk_id in chunk_ids:
        task = stage.tasks[chunk_id]
        assert torch.equal(outputs[chunk_id], combined[:, task.start:task.end_exclusive])


def test_execute_chunks_with_no_chunks_returns_empty(tiny_model, combined_run):
    stage, generation_config, _ = combined_run

    assert tiny_model._execute_chunks(stage, [], generation_config, identity_tokens, identity_logits) == {}


@pytest.mark.parametrize("chunk_ids", [[7], [0, 0]])
def test_execute_chunks_rejects_unknown_or_duplicate_ids(tiny_model, combined_run, chunk_ids):
    stage, generation_config, _ = combined_run

    with pytest.raises(ValueError):
        tiny_model._execute_chunks(stage, chunk_ids, generation_config, identity_tokens, identity_logits)
