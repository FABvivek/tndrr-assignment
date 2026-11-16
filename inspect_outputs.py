import json

with open('outputs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

fp16_runs = data['runs']['qwen2.5:1.5b-instruct-fp16']['prompt_results']
q4_runs = data['runs']['qwen2.5:1.5b-instruct-q4_K_M']['prompt_results']

for fp, q4 in zip(fp16_runs, q4_runs):
    print("=" * 70)
    print(f"PROMPT {fp['prompt_id']}: {fp['category']}")
    print("=" * 70)
    print("PROMPT TEXT:")
    print(fp['prompt'])
    print("\n--- FP16 (Tokens: %d, Speed: %.2f tok/s) ---" % (fp['generated_tokens'], fp['generation_tokens_per_sec']))
    print(fp['response'])
    print("\n--- Q4_K_M (Tokens: %d, Speed: %.2f tok/s) ---" % (q4['generated_tokens'], q4['generation_tokens_per_sec']))
    print(q4['response'])
    print("\n")
