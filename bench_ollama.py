"""
bench_ollama.py
Benchmark script to compare Ollama models (FP16 vs Q4_K_M) across 10 prompt categories.
Measures:
  - Generated tokens (eval_count)
  - Generation speed (tokens/sec)
  - Prompt-processing speed (tokens/sec)
  - Memory: model size & VRAM from /api/ps, peak system RAM & GPU VRAM via nvidia-smi / psutil.
Saves:
  - results.csv
  - outputs.json
"""

import json
import os
import subprocess
import threading
import time
import psutil
import requests
import pandas as pd

OLLAMA_API_URL = "http://localhost:11434"

PROMPTS = [
    {
        "id": 1,
        "category": "factual",
        "prompt": "What is the capital of Australia, when was it chosen as the capital, and what is its current estimated population?"
    },
    {
        "id": 2,
        "category": "arithmetic",
        "prompt": "Solve this arithmetic problem step by step: Calculate (145 * 18) - (936 / 12) + 84. What is the final exact number?"
    },
    {
        "id": 3,
        "category": "logic_puzzle",
        "prompt": "Three people (Alice, Bob, and Charlie) are sitting side by side on a bench facing forward. Alice is sitting directly to the left of Bob. Charlie is not sitting next to Alice. From left to right, what is the seating order of the three people? Explain step by step."
    },
    {
        "id": 4,
        "category": "code_generation",
        "prompt": "Write a Python function `is_palindrome(text: str) -> bool` that returns True if the string is a palindrome, ignoring non-alphanumeric characters, casing, and spaces. Include type hints, docstring, and 3 unit test assertions."
    },
    {
        "id": 5,
        "category": "code_debugging",
        "prompt": "Identify and explain the bug in this Python binary search implementation, then provide the corrected version:\n\ndef binary_search(arr, target):\n    low = 0\n    high = len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid - 1\n    return -1"
    },
    {
        "id": 6,
        "category": "summarisation",
        "prompt": "Summarise the following text in exactly two clear, factual sentences:\n'Transformer-based large language models have revolutionised artificial intelligence across multiple domains, from natural language generation and translation to automated reasoning and software engineering. However, serving modern generative models at scale introduces substantial computational bottlenecks, predominantly driven by high memory bandwidth consumption during the autoregressive decoding phase and quadratic memory scaling of key-value (KV) attention caches. Consequently, deploying LLMs efficiently on resource-constrained consumer GPUs necessitates targeted optimisation techniques, such as low-bit post-training quantization, grouped-query attention, and continuous batching systems like vLLM.'"
    },
    {
        "id": 7,
        "category": "translation_french_hindi",
        "prompt": "Translate the following English sentence accurately into both French and Hindi (in Devanagari script):\n'Careful optimization of artificial intelligence models allows researchers to run advanced language technology directly on consumer laptops without relying on cloud servers.'"
    },
    {
        "id": 8,
        "category": "json_extraction",
        "prompt": "Extract the structured details from the following biography into a single valid JSON object with keys 'full_name' (string), 'age' (integer), 'occupation' (string), 'affiliated_institution' (string), and 'research_topics' (array of strings). Do not include any conversational preamble:\n'Dr. Ananya Sharma, a 34-year-old principal research scientist at the Indian Institute of Science, leads projects on post-training quantization, sparse attention kernels, and edge model inference.'"
    },
    {
        "id": 9,
        "category": "constrained_writing",
        "prompt": "Write a 4-sentence short story about an engineer deploying an AI model. Strict constraints: exactly 4 sentences, and EVERY sentence must begin with the letter 'E'."
    },
    {
        "id": 10,
        "category": "hallucination_check",
        "prompt": "Who won the 2090 FIFA Men's World Cup, which stadium hosted the final match, and what was the final score?"
    }
]

def get_gpu_memory_used_mib():
    """Queries nvidia-smi for current VRAM usage in MiB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return None

def get_system_ram_used_mb():
    """Returns current system RAM usage in MB."""
    vm = psutil.virtual_memory()
    return vm.used / (1024 * 1024)

class MemoryMonitor:
    """Monitors peak RAM and VRAM during a benchmark run."""
    def __init__(self, interval_sec=0.1):
        self.interval = interval_sec
        self.running = False
        self.peak_system_ram_mb = 0.0
        self.peak_gpu_vram_mib = 0.0
        self.start_system_ram_mb = 0.0
        self.start_gpu_vram_mib = 0.0
        self.thread = None

    def _sample(self):
        while self.running:
            ram = get_system_ram_used_mb()
            if ram > self.peak_system_ram_mb:
                self.peak_system_ram_mb = ram

            vram = get_gpu_memory_used_mib()
            if vram is not None and vram > self.peak_gpu_vram_mib:
                self.peak_gpu_vram_mib = vram
            time.sleep(self.interval)

    def start(self):
        self.start_system_ram_mb = get_system_ram_used_mb()
        self.peak_system_ram_mb = self.start_system_ram_mb

        vram = get_gpu_memory_used_mib()
        self.start_gpu_vram_mib = vram if vram is not None else 0.0
        self.peak_gpu_vram_mib = self.start_gpu_vram_mib

        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return {
            "start_system_ram_mb": round(self.start_system_ram_mb, 2),
            "peak_system_ram_mb": round(self.peak_system_ram_mb, 2),
            "start_gpu_vram_mib": round(self.start_gpu_vram_mib, 2),
            "peak_gpu_vram_mib": round(self.peak_gpu_vram_mib, 2),
        }

def get_running_model_info(model_name):
    """Fetches model info from /api/ps."""
    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/ps", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for m in data.get("models", []):
                if model_name in m.get("name", "") or model_name in m.get("model", ""):
                    return m
            if data.get("models"):
                return data["models"][0]
    except Exception as e:
        print(f"Warning: /api/ps query failed: {e}")
    return None

def unload_model(model_name):
    """Unloads model by calling /api/generate with keep_alive=0."""
    try:
        requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=10
        )
        time.sleep(2)
    except Exception as e:
        print(f"Unload warning for {model_name}: {e}")

def run_model_benchmark(model_tag):
    print("=" * 70)
    print(f"STARTING BENCHMARK FOR MODEL: {model_tag}")
    print("=" * 70)

    # 1. Warm-up
    print("[1/3] Warming up model (loading into memory & initializing CUDA kernels)...")
    warmup_payload = {
        "model": model_tag,
        "prompt": "Hello! Please give a one sentence greeting.",
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 32
        }
    }
    warmup_res = requests.post(f"{OLLAMA_API_URL}/api/generate", json=warmup_payload, timeout=120)
    warmup_res.raise_for_status()
    print("Warm-up completed successfully.")

    # 2. Check /api/ps memory
    ps_info = get_running_model_info(model_tag)
    model_size_bytes = ps_info.get("size", 0) if ps_info else 0
    vram_bytes = ps_info.get("size_vram", 0) if ps_info else 0
    print(f"Ollama /api/ps: Size = {model_size_bytes / (1024**3):.3f} GB, VRAM = {vram_bytes / (1024**3):.3f} GB")

    # 3. Start memory monitor
    monitor = MemoryMonitor(interval_sec=0.1)
    monitor.start()

    prompt_results = []
    print(f"[2/3] Executing {len(PROMPTS)} benchmark prompts...")

    for item in PROMPTS:
        pid = item["id"]
        cat = item["category"]
        prompt_text = item["prompt"]
        print(f"  -> Prompt {pid:02d} [{cat}]...", end="", flush=True)

        payload = {
            "model": model_tag,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": 256
            }
        }

        t0 = time.perf_counter()
        resp = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload, timeout=180)
        wall_time = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()

        # Parse Ollama timings (in nanoseconds)
        eval_count = data.get("eval_count", 0)
        eval_dur_s = data.get("eval_duration", 0) / 1e9
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_dur_s = data.get("prompt_eval_duration", 0) / 1e9
        total_dur_s = data.get("total_duration", 0) / 1e9

        gen_tokens_sec = (eval_count / eval_dur_s) if eval_dur_s > 0 else 0.0
        prompt_tokens_sec = (prompt_eval_count / prompt_eval_dur_s) if prompt_eval_dur_s > 0 else 0.0

        print(f" done. Generated {eval_count} tokens @ {gen_tokens_sec:.2f} tok/s (prompt: {prompt_tokens_sec:.2f} tok/s)")

        prompt_results.append({
            "model": model_tag,
            "prompt_id": pid,
            "category": cat,
            "prompt": prompt_text,
            "response": data.get("response", "").strip(),
            "generated_tokens": eval_count,
            "generation_duration_sec": round(eval_dur_s, 4),
            "generation_tokens_per_sec": round(gen_tokens_sec, 2),
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_sec": round(prompt_eval_dur_s, 4),
            "prompt_processing_tokens_per_sec": round(prompt_tokens_sec, 2),
            "total_duration_sec": round(total_dur_s, 4),
            "wall_clock_sec": round(wall_time, 4)
        })

    mem_stats = monitor.stop()
    print(f"[3/3] Completed {model_tag}.")
    print(f"Peak VRAM: {mem_stats['peak_gpu_vram_mib']} MiB, Peak RAM: {mem_stats['peak_system_ram_mb']} MB")

    return {
        "model": model_tag,
        "ps_info": ps_info,
        "model_size_bytes": model_size_bytes,
        "vram_bytes": vram_bytes,
        "memory_stats": mem_stats,
        "prompt_results": prompt_results
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark FP16 vs Q4_K_M in Ollama")
    parser.add_argument("--fp16", default="qwen2.5:1.5b-instruct-fp16", help="FP16 model tag")
    parser.add_argument("--q4", default="qwen2.5:1.5b-instruct-q4_K_M", help="Q4 model tag")
    args = parser.parse_args()

    results_all = []
    runs = {}

    for tag in [args.fp16, args.q4]:
        run_data = run_model_benchmark(tag)
        runs[tag] = run_data
        for r in run_data["prompt_results"]:
            # Flatten memory info into rows
            row = dict(r)
            row["model_size_gb"] = round(run_data["model_size_bytes"] / (1024**3), 3)
            row["model_vram_gb"] = round(run_data["vram_bytes"] / (1024**3), 3)
            row["peak_system_ram_mb"] = run_data["memory_stats"]["peak_system_ram_mb"]
            row["peak_gpu_vram_mib"] = run_data["memory_stats"]["peak_gpu_vram_mib"]
            results_all.append(row)

        # Unload model to release memory before next setting
        unload_model(tag)
        time.sleep(3)

    # Save CSV
    df = pd.DataFrame(results_all)
    df.to_csv("results.csv", index=False)
    print("Saved results.csv successfully.")

    # Save JSON
    outputs_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "os": "Windows 11 Home Single Language",
                "cpu": "12th Gen Intel Core i5-12500H",
                "ram_gb": 16.0,
                "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)"
            },
            "parameters": {
                "temperature": 0,
                "seed": 42,
                "num_predict": 256
            }
        },
        "runs": runs
    }
    with open("outputs.json", "w", encoding="utf-8") as f:
        json.dump(outputs_payload, f, indent=2, ensure_ascii=False)
    print("Saved outputs.json successfully.")

if __name__ == "__main__":
    main()
