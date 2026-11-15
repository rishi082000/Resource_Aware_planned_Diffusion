import torch
from transformers import AutoTokenizer
import argparse
from dream.modeling_dream import DreamModel
from dream.control_tags import *
import time
import re
import json
import os
from datetime import timedelta
from pathlib import Path
import torch.distributed as dist
from semantic_parallel.distributed import DistributedStageExecutor, serve_stages

def main():
    """
    Example
    -------
    python -m eval.pd_generate \\
        --model_path path/to/model \\
        --prompt "What is Aurora Borealis? Please be concise." \\
    
     Multi-GPU (one rank per GPU):
     torchrun --nproc_per_node=4 -m eval.pd_generate \\
         --prompt "What is Aurora Borealis? Please be concise." \\
         --distributed --scheduler lpt --temperature 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="dmisrael/planned-diffusion-dream7b-sft-16ep", help="Path to the base model.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate text from.")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (ignored if do_sample=False)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--steps_ratio", type=float, default=1.0, help="Number of steps to generate.")
    parser.add_argument("--confidence_threshold", type=float, default=None, help="Confidence threshold for confidence threshold algorithm.")
    parser.add_argument("--use_cache", action="store_true", help="Use cache for generation.")
    parser.add_argument("--use_fp16", action="store_true", help="Use float16 instead of bfloat16 on CUDA.")
    parser.add_argument("--length_scale", type=float, default=None, help="Scale factor for async block lengths (None to use default 10).")
    parser.add_argument("--disable_block_sparsity", action="store_true", help="Disable block sparsity; use dense attention within generated blocks.")
    # parser.add_argument("--quiet_steps", action="store_true", help="Do not print intermediate generation steps.")
    parser.add_argument("--run_isolated_oracle", action="store_true", help="Run the isolated oracle generation algorithm.")
    parser.add_argument("--metrics_json", type=str,default=None, help="Optional path for structured JSON metrics.")
    parser.add_argument("--distributed", action="store_true", help="Spread each stage's chunks across GPUs; launch with torchrun.")
    parser.add_argument("--scheduler", type=str, default="lpt", choices=["lpt", "round_robin"], help="Chunk placement policy for --distributed.")
    parser.add_argument("--verify_distributed", action="store_true", help="Also run each stage combined on rank 0 and count token mismatches (use --temperature 0; slows the run).")
    args = parser.parse_args()

    if args.distributed and args.use_cache:
        parser.error("--distributed supports the non-cached path only; drop --use_cache")

    if args.distributed and args.disable_block_sparsity:
        parser.error("--distributed needs block sparsity: dense blocks are not independent")
    
    # set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank = 0

    if args.distributed:
        dist.init_process_group(backend="nccl" if device == "cuda" else "gloo", timeout=timedelta(minutes=30))
        rank = dist.get_rank()

        if device == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"rank {rank} needs GPU {local_rank} but only {torch.cuda.device_count()} are visible; "
                    "set --nproc_per_node to at most the number of GPUs"
                )

            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"

    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16 if args.use_fp16 else torch.bfloat16
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=args.use_cache,
        disable_block_sparsity=args.disable_block_sparsity
    )

    model = model.to(device).eval()

    if rank != 0:
        # Worker ranks only run the chunks rank 0 sends them
        serve_stages(model, device)
        dist.destroy_process_group()
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True
    ).to(device)

    print("============ Input Prompt ============\n")
    print(inputs)

    
    # def generation_tokens_hook_func(step, x, logits):
    #     # print(f"############ Step {step} ############")
    #     # print(f"Generated tokens: {x}")
    #     # print(f"Generated text: {tokenizer.decode(x[0].tolist())}")
    #     # print(tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, "[M]"))
    #     return x

    stage_executor = (
        DistributedStageExecutor(model, device, policy=args.scheduler, verify=args.verify_distributed)
        if args.distributed
        else None
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        memory_before = torch.cuda.memory_allocated()
    else:
        memory_before = 0
    
    # start_time = time.time()
    start_time = time.perf_counter()
    with torch.no_grad():
        try:
            output = model.planned_diffusion_generate(
                inputs,
                max_length=args.max_length,
                steps_ratio=args.steps_ratio,
                return_dict_in_generate=True,
                alg="pd_confidence_threshold" if args.confidence_threshold is not None else "pd_entropy",
                threshold=args.confidence_threshold,
                top_p=args.top_p,
                temperature=args.temperature,
                alg_temp=0.,
                length_scale=args.length_scale,
                run_isolated_oracle=args.run_isolated_oracle,
                stage_executor=stage_executor
            )
        finally:
            # Release the worker ranks even if generation failed
            if stage_executor is not None:
                stage_executor.shutdown()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    if device.startswith("cuda"):
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        peak_memory = 0
    
    elapsed_seconds = end_time - start_time

    structured_stages = []

    for stage in output.stage_metadata:
        stage_record = dict(stage)
        stage_record["planning_text"] = tokenizer.decode(
            stage["planning_token_ids"],
            skip_special_tokens=False,
        )

        del stage_record["planning_token_ids"]
        structured_stages.append(stage_record)
    
    tokens_generated = output.sequences.shape[1] - inputs.shape[1]
    tokens_per_second = (
        tokens_generated / elapsed_seconds if elapsed_seconds > 0 else None
    )

    metrics = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "device": device,
        "dtype": str(dtype),
        "random_seed": args.random_seed,
        "max_length": args.max_length,
        "steps_ratio": args.steps_ratio,
        "length_scale": args.length_scale,
        "use_cache": args.use_cache,
        "disable_block_sparsity": args.disable_block_sparsity,
        "input_tokens": inputs.shape[1],
        "tokens_generated": tokens_generated,
        "latency_seconds": elapsed_seconds,
        "tokens_per_second": tokens_per_second,
        "memory_before_bytes": memory_before,
        "peak_memory_allocated_bytes": peak_memory,
        "peak_memory_allocated_gib": peak_memory / (1024 ** 3),
        "stages": structured_stages,
        "run_isolated_oracle": args.run_isolated_oracle,
    }

    if stage_executor is not None:
        metrics["distributed"] = {
            "world_size": stage_executor.world_size,
            "scheduler": args.scheduler,
            "verify": args.verify_distributed,
            "stages": stage_executor.stage_records,
        }
    output_ids = output.sequences
    
    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=False,
    )
    
    def clean_text(text):
        text = re.sub(r"(<async>)(\d+)", r"\1\n\2", text) 
        regex = r'(\[PAD\]|<\/?async>|<sync\/>|<promise>.*?<\/topic>)'
        text = re.sub(regex, ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        text = text.replace("<|im_start|>assistant\n ", "<|im_start|>assistant\n")
        return text
    
    clean_generated_text = clean_text(generated_text)
    

    print("============ Clean Generation ============\n")
    print(clean_generated_text)
    
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Tokens per second: {tokens_generated / (end_time - start_time)}")

    if device.startswith("cuda"):
        print(f"Peak memory allocated: {peak_memory / (1024 ** 3)} GiB")

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_path}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main() 