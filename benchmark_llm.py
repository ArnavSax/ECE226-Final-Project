import argparse
import csv
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU monitoring via NVML
try:
    import pynvml
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


@dataclass
class MonitorStats:
    cpu_samples: list
    ram_samples_mb: list
    gpu_util_samples: list
    gpu_mem_samples_mb: list


class ResourceMonitor:
    def __init__(self, interval: float = 0.2, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stats = MonitorStats([], [], [], [])

        self.nvml_handle = None
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self.nvml_handle = None

    def _sample_loop(self):
        proc = psutil.Process(os.getpid())
        while self.running:
            try:
                self.stats.cpu_samples.append(psutil.cpu_percent(interval=None))
                mem_info = proc.memory_info()
                self.stats.ram_samples_mb.append(mem_info.rss / (1024 ** 2))

                if self.nvml_handle is not None:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    self.stats.gpu_util_samples.append(float(util.gpu))
                    self.stats.gpu_mem_samples_mb.append(mem.used / (1024 ** 2))
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def summarize(self):
        def avg(x):
            return sum(x) / len(x) if x else 0.0

        def peak(x):
            return max(x) if x else 0.0

        return {
            "avg_cpu_percent": avg(self.stats.cpu_samples),
            "peak_proc_ram_mb": peak(self.stats.ram_samples_mb),
            "avg_gpu_util_percent": avg(self.stats.gpu_util_samples),
            "peak_gpu_mem_mb": peak(self.stats.gpu_mem_samples_mb),
        }


def read_prompt(prompt_file: Optional[str], prompt_text: Optional[str]) -> str:
    if prompt_text:
        return prompt_text.strip()
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise ValueError("Provide either --prompt-file or --prompt-text")


def count_generated_tokens(output_ids: torch.Tensor, input_ids: torch.Tensor) -> int:
    return int(output_ids.shape[1] - input_ids.shape[1])


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_model_gpu(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
    )
    return model


def load_model_cpu(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map={"": "cpu"},
    )
    return model


def load_model_offload(model_name: str, offload_folder: str, gpu_mem: str, cpu_mem: str):
    os.makedirs(offload_folder, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        offload_folder=offload_folder,
        max_memory={0: gpu_mem, "cpu": cpu_mem},
    )
    return model


def get_device_for_inputs(model) -> str:
    if hasattr(model, "hf_device_map"):
        values = set(model.hf_device_map.values())

        # Accelerate may represent GPU devices as integers like 0, 1, ...
        if any(isinstance(v, int) for v in values):
            return "cuda:0"

        # Or strings like "cuda:0"
        if any(isinstance(v, str) and "cuda" in v for v in values):
            return "cuda:0"

        return "cpu"

    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cpu"


def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    assistant_model=None,
    assistant_tokenizer=None,
):
    input_device = get_device_for_inputs(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    if input_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Universal assisted decoding path for possibly different tokenizers
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model
        generate_kwargs["tokenizer"] = tokenizer
        if assistant_tokenizer is not None:
            generate_kwargs["assistant_tokenizer"] = assistant_tokenizer

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)
    if input_device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.perf_counter()

    total_time_s = end - start
    generated_tokens = count_generated_tokens(output_ids, inputs["input_ids"])
    tok_per_s = generated_tokens / total_time_s if total_time_s > 0 else 0.0
    latency_per_token_s = total_time_s / generated_tokens if generated_tokens > 0 else 0.0

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    peak_cuda_alloc_mb = 0.0
    peak_cuda_reserved_mb = 0.0
    if input_device.startswith("cuda"):
        peak_cuda_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_cuda_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return {
        "total_time_s": total_time_s,
        "generated_tokens": generated_tokens,
        "tokens_per_s": tok_per_s,
        "latency_per_token_s": latency_per_token_s,
        "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
        "output_text": decoded,
    }


def append_csv(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_models_for_mode(args) -> Tuple:
    model_main = None
    model_assistant = None
    tokenizer = None
    assistant_tokenizer = None

    if args.mode == "gpu":
        tokenizer = load_tokenizer(args.model_main)
        ensure_pad_token(tokenizer)
        model_main = load_model_gpu(args.model_main)

    elif args.mode == "cpu":
        tokenizer = load_tokenizer(args.model_main)
        ensure_pad_token(tokenizer)
        model_main = load_model_cpu(args.model_main)

    elif args.mode == "offload":
        tokenizer = load_tokenizer(args.model_main)
        ensure_pad_token(tokenizer)
        model_main = load_model_offload(
            args.model_main,
            offload_folder=args.offload_folder,
            gpu_mem=args.gpu_mem,
            cpu_mem=args.cpu_mem,
        )

    elif args.mode == "specdec":
        if args.model_assistant is None:
            raise ValueError("Speculative decoding requires --model-assistant")

        tokenizer = load_tokenizer(args.model_main)
        ensure_pad_token(tokenizer)

        assistant_tokenizer = load_tokenizer(args.model_assistant)
        ensure_pad_token(assistant_tokenizer)

        model_main = load_model_offload(
            args.model_main,
            offload_folder=args.offload_folder,
            gpu_mem=args.gpu_mem,
            cpu_mem=args.cpu_mem,
        )
        model_assistant = AutoModelForCausalLM.from_pretrained(
            args.model_assistant,
            dtype=torch.float16,
            device_map="cuda",
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return tokenizer, model_main, model_assistant, assistant_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference modes")
    parser.add_argument("--mode", required=True, choices=["gpu", "cpu", "offload", "specdec"])
    parser.add_argument("--model-main", required=True)
    parser.add_argument("--model-assistant", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpu-mem", default="10GiB")
    parser.add_argument("--cpu-mem", default="28GiB")
    parser.add_argument("--offload-folder", default="offload_qwen7b")
    parser.add_argument("--csv-path", default="results/benchmark_results.csv")
    parser.add_argument("--save-output-dir", default="logs")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(args.save_output_dir, exist_ok=True)

    prompt = read_prompt(args.prompt_file, args.prompt_text)

    print(f"[INFO] Loading models for mode={args.mode}")
    tokenizer, model_main, model_assistant, assistant_tokenizer = build_models_for_mode(args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Warmup runs
    for i in range(args.warmup_runs):
        print(f"[INFO] Warmup run {i+1}/{args.warmup_runs}")
        _ = run_generation(
            model_main,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            assistant_model=model_assistant,
            assistant_tokenizer=assistant_tokenizer,
        )

    # Measured runs
    for trial in range(1, args.trials + 1):
        print(f"[INFO] Trial {trial}/{args.trials}")

        monitor = ResourceMonitor(interval=0.2, gpu_index=0)
        monitor.start()

        result = run_generation(
            model_main,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            assistant_model=model_assistant,
            assistant_tokenizer=assistant_tokenizer,
        )

        monitor.stop()
        monitor_summary = monitor.summarize()

        row = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "tag": args.tag,
            "mode": args.mode,
            "model_main": args.model_main,
            "model_assistant": args.model_assistant or "",
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "trial": trial,
            "total_time_s": round(result["total_time_s"], 6),
            "generated_tokens": result["generated_tokens"],
            "tokens_per_s": round(result["tokens_per_s"], 6),
            "latency_per_token_s": round(result["latency_per_token_s"], 6),
            "peak_cuda_alloc_mb": round(result["peak_cuda_alloc_mb"], 3),
            "peak_cuda_reserved_mb": round(result["peak_cuda_reserved_mb"], 3),
            "avg_cpu_percent": round(monitor_summary["avg_cpu_percent"], 3),
            "peak_proc_ram_mb": round(monitor_summary["peak_proc_ram_mb"], 3),
            "avg_gpu_util_percent": round(monitor_summary["avg_gpu_util_percent"], 3),
            "peak_gpu_mem_mb": round(monitor_summary["peak_gpu_mem_mb"], 3),
            "prompt_chars": len(prompt),
        }

        append_csv(args.csv_path, row)

        output_path = os.path.join(
            args.save_output_dir,
            f"{run_id}_{args.mode}_trial{trial}.txt",
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["output_text"])

        print("[RESULT]", row)

    print(f"[DONE] Results saved to {args.csv_path}")


if __name__ == "__main__":
    main()