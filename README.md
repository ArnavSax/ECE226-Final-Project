# LLM Inference Benchmarking on Limited VRAM Systems

This project benchmarks different strategies for running large language models on systems with limited GPU memory. The experiments compare performance across several inference configurations including GPU-only execution, CPU-only execution, GPU–CPU offloading, and speculative decoding.

The goal is to understand how memory constraints affect LLM inference throughput and latency, and to evaluate techniques that enable larger models to run on constrained hardware.

---

# Project Overview

Modern LLMs often exceed the available VRAM on consumer GPUs. This project evaluates several approaches for running these models despite memory limitations.

Inference modes tested:

1. GPU-only baseline  
   Model runs entirely in GPU VRAM

2. CPU-only baseline  
   Model runs entirely on CPU

3. GPU + CPU offloading  
   Model weights are dynamically moved between CPU RAM and GPU VRAM

4. Layer-wise Swapping
   Model layers are intentionally distributed between the CPU RAM and GPU VRAM.

5. Speculative decoding  
   A smaller draft model proposes tokens which are verified by a larger model

Each experiment records:

- inference throughput (tokens/sec)
- latency per generated token
- GPU utilization
- GPU memory usage
- CPU usage
- process RAM usage

---

# Repository Structure
```text
ECE226-Final-Project
│
├── benchmark_llm.py        # Main benchmarking script
├── prompts/
│   └── prompt.txt          # Input prompt used for experiments
│
├── results/                # CSV benchmark outputs
├── logs/                   # Generated text outputs
│
└── offload_qwen7b/         # Disk cache used during CPU offloading
```
---

# Requirements

The script requires Python with the following packages:

torch  
transformers  
accelerate  
psutil  
pynvml  

Install them with:

pip install torch transformers accelerate psutil pynvml

A CUDA-capable GPU is required for GPU and offloading modes.

---

# Benchmark Script

All experiments are run through the file:

benchmark_llm.py

The script measures inference performance and logs results to a CSV file.

Metrics recorded include:

- total inference time
- tokens generated
- tokens per second
- latency per token
- peak CUDA memory usage
- GPU utilization
- CPU utilization
- process RAM usage

---

# Running Experiments

All results are appended to the CSV file specified with:

--csv-path

---

# GPU Baseline

Runs the model entirely in GPU memory.

python benchmark_llm.py \
  --mode gpu \
  --model-main Qwen/Qwen2.5-3B-Instruct \
  --prompt-file prompts/prompt.txt \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --trials 3 \
  --csv-path results/platformB_results.csv \
  --save-output-dir logs \
  --tag platformB_gpu_3b

---

# CPU Baseline

Runs the model entirely on CPU.

python benchmark_llm.py \
  --mode cpu \
  --model-main Qwen/Qwen2.5-3B-Instruct \
  --prompt-file prompts/prompt.txt \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --trials 3 \
  --csv-path results/platformB_results.csv \
  --save-output-dir logs \
  --tag platformB_cpu_3b

---

# GPU + CPU Offloading

Allows a model larger than GPU VRAM to run by dynamically moving weights between CPU and GPU memory.

python benchmark_llm.py \
  --mode offload \
  --model-main Qwen/Qwen2.5-7B-Instruct \
  --prompt-file prompts/prompt.txt \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --trials 3 \
  --gpu-mem 10GiB \
  --cpu-mem 28GiB \
  --offload-folder offload_qwen7b \
  --csv-path results/platformB_results.csv \
  --save-output-dir logs \
  --tag platformB_offload_7b

Parameters:

--gpu-mem  
Maximum GPU memory allocation

--cpu-mem  
Maximum CPU memory allocation

--offload-folder  
Disk location used for offloaded weights

---

# Speculative Decoding

Uses a smaller draft model to propose tokens which are verified by a larger model.

This reduces the number of expensive forward passes required by the large model.

python benchmark_llm.py \
  --mode specdec \
  --model-main Qwen/Qwen2.5-7B-Instruct \
  --model-assistant Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-file prompts/prompt.txt \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --trials 3 \
  --gpu-mem 10GiB \
  --cpu-mem 28GiB \
  --offload-folder offload_qwen7b \
  --csv-path results/platformB_results.csv \
  --save-output-dir logs \
  --tag platformB_specdec

---

# Output Files

CSV Results

All experiment results are stored in /results

Each row contains:

- timestamp
- run id
- experiment tag
- inference mode
- tokens/sec
- latency per token
- GPU utilization
- CPU utilization
- memory statistics

Generated Text

Model outputs are saved for inspection:

logs/<run_id>_<mode>_trialX.txt

---

# Profiling

Experiments can be profiled using NVIDIA Nsight Systems:

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  -o results/nsys_offload_7b \
  python benchmark_llm.py ...

This produces GPU execution timelines useful for analyzing memory transfer and kernel execution patterns.

---

# Purpose

This project demonstrates how LLM inference performance changes under memory constraints and evaluates techniques that enable larger models to run on hardware with limited VRAM.

Key questions explored:

- How much slower is CPU offloading compared to GPU-resident inference?
- How much GPU utilization is lost due to weight transfers?
- Can speculative decoding recover some of this performance?

# Authors

Arnav Saxena, Michael Aquilina 
UC San Diego ECE
