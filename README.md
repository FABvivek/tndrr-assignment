 

---

## Tasks Overview

### [Task 3: Running a model locally & optimisation](./task3/)
- **PDF Report (max 4 pages):** [task3/REPORT.pdf](./task3/REPORT.pdf)
- **Markdown Report:** [task3/REPORT.md](./task3/REPORT.md)
- **Benchmarked Models:** `qwen2.5:1.5b-instruct-fp16` vs. `qwen2.5:1.5b-instruct-q4_K_M`
- **Runner:** Ollama (REST API, v0.35.1) on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Artifacts & Data:**
  - [task3/results.csv](./task3/results.csv) (Raw per-prompt timing and memory metrics)
  - [task3/outputs.json](./task3/outputs.json) (Raw JSON inputs, outputs, and system metadata)
  - [task3/benchmark_comparison.png](./task3/benchmark_comparison.png) (Speed & memory comparative bar charts)
  - [task3/bench_ollama.py](./task3/bench_ollama.py) (Benchmark automation script)
