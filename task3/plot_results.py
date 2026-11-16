import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

models = ['FP16\n(qwen2.5:1.5b-fp16)', '4-bit Q4_K_M\n(qwen2.5:1.5b-q4_K_M)']
colors = ['#1f77b4', '#2ca02c']

# 1. Throughput Comparison (Tokens / sec)
gen_speeds = [23.64, 101.86]
prompt_speeds = [493.82, 950.35]

x = np.arange(len(models))
width = 0.35

rects1 = axes[0].bar(x - width/2, gen_speeds, width, label='Generation (tok/s)', color='#4C72B0')
rects2 = axes[0].bar(x + width/2, prompt_speeds, width, label='Prompt Eval (tok/s)', color='#55A868')

axes[0].set_ylabel('Throughput (Tokens / Second)', fontsize=11, fontweight='bold')
axes[0].set_title('Inference Speed Comparison', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=10)
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels
for rect in rects1:
    h = rect.get_height()
    axes[0].annotate(f'{h:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    axes[0].annotate(f'{h:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Memory Footprint Comparison (GB)
model_size_gb = [3.505, 1.086]
peak_vram_gb = [2493.0 / 1024, 1354.0 / 1024]  # convert MiB to GiB: 2.435 GB vs 1.322 GB

rects3 = axes[1].bar(x - width/2, model_size_gb, width, label='Model Weight Size (GB)', color='#DD8452')
rects4 = axes[1].bar(x + width/2, peak_vram_gb, width, label='Peak GPU VRAM Used (GiB)', color='#C44E52')

axes[1].set_ylabel('Memory (Gigabytes)', fontsize=11, fontweight='bold')
axes[1].set_title('Memory Footprint Comparison', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=10)
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].set_ylim(0, 4.2)

for rect in rects3:
    h = rect.get_height()
    axes[1].annotate(f'{h:.2f} GB',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in rects4:
    h = rect.get_height()
    axes[1].annotate(f'{h:.2f} GiB',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Ollama Local Benchmark: Qwen2.5-1.5B (FP16 vs 4-bit Q4_K_M) on NVIDIA RTX 3050 Laptop GPU',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("Successfully generated benchmark_comparison.png")
