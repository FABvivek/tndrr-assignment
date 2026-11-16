import json

with open('outputs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for p_id in [4, 5, 6]:
    fp = data['runs']['qwen2.5:1.5b-instruct-fp16']['prompt_results'][p_id-1]
    q4 = data['runs']['qwen2.5:1.5b-instruct-q4_K_M']['prompt_results'][p_id-1]
    print("=" * 60)
    print(f"PROMPT {p_id}: {fp['category']}")
    print("=" * 60)
    print("[FP16]:")
    print(fp['response'])
    print("\n[Q4_K_M]:")
    print(q4['response'])
