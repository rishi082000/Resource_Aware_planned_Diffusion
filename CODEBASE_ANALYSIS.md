# Planned Diffusion codebase analysis

Analysis target: upstream `planned-diffusion/planned-diffusion`, commit
`5c52637473c1a1f1fe78d87bcc0e407e73881fbc` (the current `origin/main` at the
time of inspection).

## Executive conclusion

The proposal's main premise is supported at the attention-graph level: within
one planning stage, every generated async block attends to the common prefix and
to its own block, but not to another concurrent block's interior. That makes the
blocks plausible device-level tasks.

It is not yet safe to run those blocks separately without a small inference
refactor and a correctness experiment. The implementation currently constructs
all blocks in one combined sequence, performs one model forward per diffusion
iteration, and derives rotary positions from physical tensor indices. In
particular, `DreamBaseModel.forward` overwrites a supplied `position_ids`
argument. A compact `prefix + selected block` sequence would therefore give a
later block different RoPE positions than it has in the combined baseline.

The correct next step is to extract a single-stage representation and a
per-worker stage executor while retaining the combined single-GPU executor as
the correctness oracle. Do not start multi-stage distributed execution until
the isolated-block oracle test passes.

## Relevant repository architecture

| Area | Files | Responsibility |
| --- | --- | --- |
| User entry points | `eval/pd_generate.py`, `eval/demo.py` | Tokenize one prompt, load Dream, call `planned_diffusion_generate`, decode output. |
| Dataset evaluation | `eval/alpaca_eval_diffusion.py`, `eval/alpaca_eval_ar.py` | One process/model replica per GPU; shard independent dataset examples across ranks; merge outputs. |
| Planned Diffusion orchestration | `dream/generation_utils.py` | Non-cached AR planning, stage construction, diffusion loop, and stage continuation. |
| Cached orchestration | `dream/generation_utils_cached_pd.py` | Same state machine with AR KV reuse and selective replacement of unfinished diffusion positions. |
| Plan parsing and PDSA tensors | `dream/pd_utils.py` | Parse promise lengths, create async blocks and sparse masks, produce `block_info`, update masks, and perform block-local unmasking. |
| Dream model and positions | `dream/modeling_dream.py` | Transformer, SDPA, RoPE position construction, ordinary cache append, and dual-cache in-place replacement. |
| Special-token IDs | `dream/control_tags.py` | IDs for EOS, mask, async start/end, promise start/end, sync, and padding. |
| Training-time mask reference | `train/attention_mask_v2.py` | Defines the intended PASTA/PDSA isolation and post-sync visibility rules. |
| Dense-attention ablation | `train/attention_mask_v3.py` and `disable_block_sparsity` paths | Ablates separation of consecutive blocks; it is not the distributed target. |

## Exact ownership map

- **Model loading:** `DreamPreTrainedModel.from_pretrained` in
  `dream/modeling_dream.py:624` loads weights and replaces the Hugging Face
  generation config with `DreamGenerationConfig`. The user-facing loaders are
  `eval/pd_generate.py:45-55`, `eval/demo.py:91-103`, and
  `eval/alpaca_eval_diffusion.py:46-62`.
- **Plan generation:** `DreamGenerationMixin._ar_sample` in
  `dream/generation_utils.py:330` and the cached version in
  `dream/generation_utils_cached_pd.py:335` autoregress until EOS or `<sync/>`.
  There is no separate semantic-plan object or parser class.
- **Plan parsing and chunk creation:** `create_pd_inputs` in
  `dream/pd_utils.py:6` scans from the most recent sync, locates
  `<promise>-<topic> ... </topic>`, extracts the first one- or two-digit number,
  multiplies it by 10 by default (or `length_scale`), and creates
  `<async> [MASK]... </async>` blocks.
- **`block_info`:** created in `dream/pd_utils.py:28-109` as tuples
  `(absolute_start, absolute_end_exclusive, predicted_content_length)`. Start
  and end include the async delimiters in the slice, while `block_size` counts
  only masked content tokens.
- **PDSA attention mask:** `create_pd_inputs` at
  `dream/pd_utils.py:82-123`; the new block rows can see the entire existing
  prefix, each block's square is bidirectional, and concurrent block interiors
  are mutually hidden. `update_attention_mask` at line 127 combines a new stage
  with the previous stages. `invert_and_expand_attention_mask` at line 149
  converts the Boolean graph to additive attention bias. The training reference
  is `compute_pasta_diffusion_metadata_v2` in
  `train/attention_mask_v2.py:15`.
- **Diffusion sampling:** `_diff_sample` in
  `dream/generation_utils.py:597` (non-cached) and
  `dream/generation_utils_cached_pd.py:609` (cached). Each iteration performs a
  Dream forward, shifts logits by one position, samples all still-masked
  positions, and commits tokens per block through `block_unmask` or
  `block_unmask_confidence_threshold` in `dream/pd_utils.py:223` and `:275`.
- **Cached generation:** `DreamGenerationMixinWithCache` in
  `dream/generation_utils_cached_pd.py`. `create_replace_mask` and
  `pad_key_values` in `dream/pd_utils.py:165` and `:193` select unfinished
  positions and extend the cache. `DreamSdpaAttention.forward` in
  `dream/modeling_dream.py:397` appends ordinary AR cache entries or replaces
  selected positions when `dual_cache=True`.
- **Logical/rotary positions:** no separate logical position metadata exists.
  `DreamBaseModel.forward` in `dream/modeling_dream.py:708` constructs a simple
  `arange` at lines 762-770 and overwrites any caller-provided `position_ids`.
  The non-cached PDSA path sets `tok_idx=None` for its 3-D mask, so it also uses
  physical sequence positions. The cached path uses `replace_position` to pick
  RoPE rows for query replacement, not a scheduler-facing logical-position API.
- **Synchronization boundaries:** `_ar_sample` stops on EOS or `SYNC_TOKEN_ID`.
  `planned_diffusion_generate` removes that terminal token, creates and denoises
  the stage, then appends sync again and repeats planning. Thus `<sync/>` is the
  stage boundary. There is no inter-rank synchronization in this path.
- **Distributed evaluation:** `eval/alpaca_eval_diffusion.py:29-42` initializes
  NCCL, line 70 calls `Dataset.shard(num_shards=world_size, index=rank)`, every
  rank independently calls `planned_diffusion_generate`, and line 192 only
  barriers before rank 0 merges per-rank JSON files. The AR evaluator similarly
  assigns indices with `range(rank, len(instructions), world_size)` and gathers
  results. It is prompt-level data parallelism, not chunk-level parallelism.

## One-prompt execution trace

1. An entry point applies the tokenizer's chat template and passes one token
   tensor to `DreamModel.planned_diffusion_generate`.
2. `DreamModel.__getattribute__` selects the cached or non-cached generation
   mixin according to `config.use_cache`.
3. `_ar_sample` autoregressively extends the current sequence until it emits
   `<sync/>` or EOS. These emitted promise tags and predicted numeric lengths
   are the plan; the repository does not materialize a separate plan object.
4. The terminator is temporarily removed. `create_pd_inputs` parses promises
   since the last sync, converts predicted numbers to block lengths, appends all
   async mask blocks, constructs the sparse stage mask, and returns `block_info`.
5. The stage uses `max(predicted block length) * steps_ratio` diffusion
   iterations for a PD algorithm.
6. `_diff_sample` runs the entire combined tensor through Dream once per
   iteration. `block_unmask` makes the same global diffusion schedule act
   independently inside each block.
7. When all masks have been replaced, the stage sequence is complete. On sync,
   the sync token is restored and planning continues with the now-visible
   completed blocks; on EOS, generation ends.
8. The entry point decodes the full sequence and removes plan/async/sync tags for
   presentation. Original block order is simply tensor order; there is no
   explicit gather/reorder operation in upstream code.

## Independence verdict

For the default sparse configuration, concurrent blocks do not exchange hidden
states through attention. They share only the prefix/plan, model weights,
diffusion iteration count, and random-number stream. Therefore they do not
require communication at every transformer layer. This supports the proposed
task-parallel experiment.

The verdict has four qualifications:

1. `--disable_block_sparsity` deliberately makes the appended block area dense;
   isolated execution is invalid in that mode.
2. A later block must preserve its positions from the combined layout. The
   current model API prevents this because it discards explicit
   `position_ids`.
3. The combined baseline performs prefix computation once per diffusion step;
   naive isolated workers duplicate the prefix for every chunk assigned to
   them. This can dominate both latency and GPU-seconds.
4. Sampling equivalence requires a defined per-chunk RNG policy. A single
   process samples all masked tokens in tensor order, whereas separate ranks
   have separate RNG states. Greedy token selection (`temperature=0`) removes
   multinomial token sampling but entropy/confidence ranking, tie behavior, and
   floating-point differences still need checking.

## Smallest insertion seam

Insert the new layer between `create_pd_inputs` and `_diff_sample`, currently
the call sequence at `dream/generation_utils.py:460-498` and
`dream/generation_utils_cached_pd.py:476-512`.

Refactor without changing default behavior:

1. Extract a `PlannedStage` Python value containing the common prefix, full
   stage layout, sparse mask, ordered `ChunkTask` records, global diffusion-step
   count, and stage terminator.
2. Keep an upstream-compatible `execute_combined_stage(stage)` wrapper around
   the current `_diff_sample`; this remains the baseline and correctness oracle.
3. Add `execute_isolated_chunk(stage, task)` that builds `prefix + one block`
   but supplies the block's original logical positions. First modify
   `DreamBaseModel.forward` to honor explicit `position_ids`; test this change
   against unchanged combined generation.
4. Add `execute_scheduled_stage(stage, assignments)` outside `dream/` where
   possible. Rank 0 broadcasts only stage/task metadata; every rank executes its
   tasks in stable chunk-ID order; rank 0 gathers token tensors and writes them
   into the full stage layout before the existing continuation logic resumes.
5. Initially disable the cached PDSA executor for multi-GPU mode. Establish
   correctness with non-cached isolated chunks, then design cache replication
   or prefix-prefill reuse explicitly.

This keeps upstream changes narrow: one explicit-position fix plus a small hook
at the stage boundary. Scheduling, distributed control, metrics, and tests can
live under a new `semantic_parallel/` package.

## Correctness and engineering risks

| Risk | Why it matters | Required gate |
| --- | --- | --- |
| Position IDs | Compacting block B changes RoPE indices; caller positions are currently overwritten. | Combined-vs-isolated logits test with explicit original positions before distributed work. |
| Prefix duplication | Each replica may recompute the full prompt/plan during every diffusion iteration. | Measure prefix length and duplicated FLOPs; later evaluate per-rank prefix cache reuse. |
| KV caches | Cached PDSA stores/replaces entries at full-sequence indices. Per-worker compact caches cannot be gathered naively. | Start non-cached; add dedicated cache-index tests before enabling `--use_cache`. |
| Block order | Upstream order is physical tensor order, while completion order varies by rank. | Gather by immutable `chunk_id`, validate exact slices, then reconstruct in original order. |
| Stage barriers | Subsequent planning can attend to completed blocks only after sync. | One collective per stage; no rank may continue planning independently. |
| Sampling/RNG | Rank-local RNG streams do not match one combined sampling call. | First compare deterministic/greedy logits and tokens; define seed as `(request, stage, chunk)` for repeatability. |
| Unequal lengths | Upstream uses the largest block to set a shared number of steps. | Preserve the stage-global step count for the first correctness comparison. |
| Empty/failed tasks | A failed rank can deadlock collectives or leave holes. | Validate assignments and use fail-fast distributed error propagation/timeouts. |
| Dense ablation | Dense blocks are not independent. | Reject multi-GPU chunk mode when block sparsity is disabled. |
| Memory economics | Every rank holds a 7B replica and prefix/cache state. | Report peak memory and GPU-seconds, not latency alone. |

## Baseline blockers found during inspection

The documented `eval/pd_generate.py` command is not currently a trustworthy
baseline without small repairs:

- line 43 reads `args.use_fp16`, but the parser defines no `--use_fp16`
  argument, causing `AttributeError` on both CPU and CUDA paths that reach this
  branch;
- unlike `eval/demo.py`, the script does not move the loaded model to `device`,
  while it moves inputs there, causing a device mismatch on CUDA unless loading
  behavior happens to place the model implicitly.

These issues do not invalidate the architecture, but Phase 0 should fix and
test the CLI before collecting any timing data. GPU timings also need explicit
`torch.cuda.synchronize()` around measured regions because CUDA execution is
asynchronous.

## Incremental implementation plan

### Checkpoint 0 - reproducible baseline

- Repair only the baseline CLI defects above.
- Add structured output containing raw plan/scaffold, parsed block lengths,
  diffusion steps, latency with CUDA synchronization, and peak allocated memory.
- Preserve the existing output by default.
- Test on a tiny/local model if available, then on the official checkpoint on
  one allocated GPU.

### Checkpoint 1 - stage and task extraction

- Introduce `PlannedStage` and `ChunkTask` dataclasses.
- Unit-test promise parsing, absolute/exclusive boundaries, sync-stage IDs, and
  length scaling with synthetic token tensors.
- Make combined generation consume the extracted stage without token changes.

### Checkpoint 2 - C++ scheduler and local thread runtime

- Use static scheduling for the main implementation: every task is assigned to
  a worker before stage execution begins. Round robin and LPT are the required
  checkpoint 2 policies.
- Implement standalone C++ round-robin and LPT policies plus estimated worker
  loads; expose them with pybind11 or a small PyTorch extension.
- Unit-test empty inputs, workers greater than tasks, ties, invalid costs, and
  the proposal's `120/80/35/20` example.
- Implement a reusable static-stage workflow for assignment validation, worker
  execution, stage completion, gathering, and reconstruction by `chunk_id`.
- Exercise that workflow locally with one CPU thread per worker and an injected
  sleep-based executor for 1/2/4 workers. Validate barriers, reconstruction,
  speedup, idle time, and machine-readable metrics.
- Keep the execution backend separate from the shared workflow. Checkpoint 4
  replaces local threads and ordinary return values with `torchrun` ranks and
  distributed collectives while retaining the task, assignment, validation,
  reconstruction, and metrics contracts.
- Defer dynamic worker-pull dispatch until after the static multi-GPU path and
  core experiments are complete. That optional extension can later compare
  FIFO/chunk-order and longest-estimated-cost-first queues with the static
  policies using the same runtime records and metrics.

### Checkpoint 3 - isolated single-GPU oracle

- Make Dream honor explicit position IDs.
- Execute each chunk independently on one GPU and reconstruct the stage.
- Compare combined and isolated per-iteration logits/tokens with sparse
  attention, the same global step count, and controlled RNG. This is the
  go/no-go gate for multi-GPU execution.Do

### Checkpoint 4 - one-stage multi-GPU path

- Use one full model replica per `torchrun` rank.
- Rank 0 plans, extracts tasks, invokes the C++ scheduler, and broadcasts
  metadata. Ranks execute assigned chunks, gather results, and reconstruct on
  rank 0.
- Support exactly one stage first; reject later sync stages clearly rather than
  silently producing incorrect output.

### Checkpoint 5 - multiple stages and optional cache work

- Add a barrier/gather/reconstruction cycle for every sync stage.
- Broadcast reconstructed state before the next stage.
- Only then evaluate prefix prefill/KV reuse. Keep non-cached mode as the
  reference implementation.

### Checkpoint 6 - experiments and adaptive scheduling

- Record 1/2/4-GPU latency, per-rank compute/wait time, scheduling and
  collective overhead, assigned predicted work, memory, utilization when
  available, and GPU-seconds in CSV/JSON.
- Sweep chunk count, lengths, imbalance, prefix length, and diffusion steps.
- Fit a simple measured cost model only after enough observations exist; compare
  round robin, LPT length-aware, measured-cost-aware, and the combined PDSA
  baseline.
- Here, adaptive scheduling means updating cost estimates or policy choices from
  measured executions across stages or requests. Dynamic worker-pull dispatch,
  if time permits, remains a separate final extension.

## Recommended first test matrix

1. Synthetic mask test proving block A and block B cannot attend to each other's
   interiors but both attend to the prefix.
2. Combined vs isolated forward-logit test with one block, then two unequal
   blocks, using explicit original positions.
3. Combined vs reconstructed token test at `temperature=0` for one stage.
4. Repeatability test using per-chunk seeds.
5. Mock scheduler tests for 1/2/4 workers.
6. Distributed smoke test with two tiny tasks and a deliberate rank failure to
   verify fail-fast behavior.

No final-system implementation was made in this checkpoint.



### Detailed Explanation: -- 

The first three incremental plans are Checkpoints 0, 1, and 2. They deliberately avoid multi-GPU model execution initially: first establish a reliable measurement, then represent work explicitly, then validate scheduling independently.
Use this small example throughout:
Prompt: “Name France’s capital and describe its location.”
Suppose the model generates this plan:
<promise>-<topic>capital 2</topic>
<promise>-<topic>location 5</topic>
With the default length scale of 10:
Chunk A: capital  → 2 × 10 = 20 tokens
Chunk B: location → 5 × 10 = 50 tokens
The stage becomes conceptually:
[prompt + plan | <async>20 masks</async> | <async>50 masks</async>]
Checkpoint 0 — Reproducible baseline
The objective is to make the existing single-GPU implementation trustworthy before changing its architecture.
Right now, pd_generate.py has two basic problems:
- It reads args.use_fp16, but the argument is not defined.
- It moves inputs to CUDA but does not explicitly move the model there.
We repair those issues and add reliable measurements.
For our example, the baseline output should record something like:
{
  "plan": [
    {"topic": "capital", "predicted_length": 20},
    {"topic": "location", "predicted_length": 50}
  ],
  "diffusion_steps": 50,
  "latency_seconds": 7.92,
  "peak_gpu_memory_gib": 15.4,
  "tokens_generated": 86
}
Why 50 diffusion steps?
max(20, 50) * steps_ratio
With steps_ratio=1.0, the larger block determines the stage-wide step count:
max(20, 50) × 1.0 = 50
Both blocks participate in those diffusion iterations, even though the first block is shorter and may finish earlier.
CUDA operations are asynchronous, so accurate timing requires synchronization:
torch.cuda.synchronize()
start = time.time()

output = model.planned_diffusion_generate(...)

torch.cuda.synchronize()
elapsed = time.time() - start
Without synchronization, Python may stop the timer while the GPU is still working.
Checkpoint 0 therefore answers:
What does the original single-GPU implementation produce, how long does it take, and how much memory does it use?

This becomes the baseline against which all later changes are compared.
Checkpoint 1 — Stage and task extraction
Currently, the plan exists mainly as tokens embedded in one large sequence. There is no explicit Python object saying, “This stage contains these two tasks.”
Checkpoint 1 introduces objects similar to:
@dataclass
class ChunkTask:
    chunk_id: int
    topic: str
    content_length: int
    start: int
    end: int

@dataclass
class PlannedStage:
    prefix: torch.Tensor
    full_sequence: torch.Tensor
    tasks: list[ChunkTask]
    diffusion_steps: int
    stage_id: int
Our example might be extracted as:
stage = PlannedStage(
    stage_id=0,
    prefix=prompt_and_plan_tokens,
    tasks=[
        ChunkTask(
            chunk_id=0,
            topic="capital",
            content_length=20,
            start=40,
            end=62,
        ),
        ChunkTask(
            chunk_id=1,
            topic="location",
            content_length=50,
            start=62,
            end=114,
        ),
    ],
    diffusion_steps=50,
)
The slice lengths are 22 and 52 because the boundaries also include the <async> and </async> tokens:
Chunk A: <async> + 20 content positions + </async>
Chunk B: <async> + 50 content positions + </async>
The end boundary is exclusive:
chunk_tokens = full_sequence[start:end]
Why make these objects?
Later, a scheduler needs an explicit list of work:
Task 0: 20 tokens
Task 1: 50 tokens
It should not have to understand model-specific promise tags or scan raw token tensors.
Checkpoint 1 also tests cases such as:
- extracting the correct numeric lengths;
- multiplying them by length_scale;
- preserving absolute boundaries;
- assigning stable chunk_id values;
- identifying which <sync/> stage each block belongs to.
Most importantly, generation must remain unchanged:
Before extraction: original tokens
After extraction:  exactly the same tokens
At this checkpoint, the program still sends the combined sequence to one GPU. We have only converted implicit information into structured information.
Checkpoint 1 answers:
Exactly what independent tasks exist, where do they belong in the sequence, and how can a scheduler refer to them safely?

Checkpoint 2 — C++ scheduler and local thread runtime
Now that we have explicit tasks, we can build the scheduler without involving the 7B model or GPUs.
For our example:
Task A cost: 20
Task B cost: 50
Workers:     2
The scheduler produces assignments:
Worker 0 → Task B, estimated load 50
Worker 1 → Task A, estimated load 20
A simple simulated executor stands in for chunk computation:
def execute_task(task, worker_id):
    time.sleep(task.content_length * 0.01)
    return f"result-{task.chunk_id}"
Therefore:
Task A sleeps for 0.20 seconds
Task B sleeps for 0.50 seconds
Sequential execution
One worker performs both:
A: 0.20 seconds
B: 0.50 seconds
Total ≈ 0.70 seconds
Parallel mock execution
Two workers execute concurrently:
Worker 0: B ─────────────────── 0.50 s
Worker 1: A ──────── 0.20 s, then idle
Total ≈ 0.50 seconds
The theoretical speedup is:
0.70 / 0.50 = 1.4×
After execution, results must be reconstructed by chunk_id, not completion time. Task A finishes before Task B, but output order must remain:
[prefix | result of Task A | result of Task B]
not:
[prefix | result of Task B | result of Task A]

The local runtime is an incremental implementation of the final stage workflow,
not disposable benchmark code. It validates this sequence:
tasks → assignments → worker execution → stage completion → gather → ordered reconstruction.
The executor is injected, so checkpoint 2 can use sleeping without embedding
simulation behavior in the runtime itself. Shared validation and reconstruction
remain useful in the multi-GPU path.

The worker backend changes in checkpoint 4. The local version uses threads that
share Python memory and return values normally. The multi-GPU version uses one
`torchrun` process per GPU, broadcasts assignments from rank 0, synchronizes
ranks with distributed collectives, and gathers results back to rank 0. That
later checkpoint must additionally validate serialization, tensor placement,
rank failure handling, CUDA synchronization, and isolated-chunk correctness.

Scheduling policies
Checkpoint 2 initially implements two policies:
- Round robin: distribute tasks successively across workers.
- LPT: sort tasks by largest predicted cost first, then assign each to the least-loaded worker.
For a more revealing example from the analysis:
Task costs: 120, 80, 35, 20
Workers: 2
A deterministic LPT assignment is:
Worker 0: 120 = 120
Worker 1: 80 + 35 + 20 = 135
Estimated completion time:
max(120, 135) = 135

Checkpoint 2 is complete when the native static policies, local stage runtime,
1/2/4-worker simulations, reconstruction, timing, idle metrics, and JSON output
pass. Dynamic dispatch is deferred until after the static multi-GPU path and
core experiments. If implemented, idle workers will pull unstarted tasks from a
FIFO or longest-estimated-cost-first queue; running tasks will not migrate, be
preempted, or be split.

The scheduler must also handle:
- zero tasks;
- more workers than tasks;
- equal task costs;
- invalid or negative costs;
- idle workers;
- worker failure;
- waiting at the stage barrier.
Checkpoint 2 answers:
Given task costs and available workers, can we distribute, synchronize, gather, and reconstruct work correctly—without GPU/model complexity hiding scheduler bugs?

How the three checkpoints connect
Checkpoint 0
Reliable single-GPU reference
        ↓
Checkpoint 1
Convert token-embedded plans into explicit stages and tasks
        ↓
Checkpoint 2
Validate static round-robin and LPT assignments using a local thread runtime
with an injected simulated executor
After these three checkpoints, we still have not executed separate chunks on separate GPUs. That begins only after Checkpoint 3 proves that isolated chunk execution produces the same result as combined single-GPU execution.
