import anthropic
import pandas as pd
import torch
import transformers

from transformers import AutoModelForSequenceClassification, AutoTokenizer

client = anthropic.Client(api_key="your_api_key")

model = "gpt-4o"  # [gpt-4o, gpt-4o-mini, o1-preview, o1-mini]
method = "few_shot"  # [zero_shot, few_shot, few_shot_cot]
input_file = "climate_run_results"  # [logic_run_results, climate_run_results]
num_rows = 200

df = pd.read_csv(f"results/{input_file}.csv")
df=df[0:num_rows]
results=[]

for i,row in df.iterrows():
    if isinstance(row['articles'], float):
        results.append(0)
        continue
    with open(f"prompts/prompt_{method}.txt", encoding="ascii", errors="ignore") as f:
        prompt = f.read() + row['articles'] + " \nAnswer: "
    try:
        message = client.completions.create(
            model="claude-3.5-sonnet",
            max_tokens=1000,
            temperature=0.0,
            prompt=f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}",
        )
        result=message.get('completion', '').strip()
    except:
        result="500 ERROR RESULT"
    print(result)
    if "Valid" in result.strip():
        results.append(1)
        print(1)
    else:
        results.append(0)
        print(0)
    print(result)

df['result']=results
df.to_csv(f'results/{method}_{model}_{input_file}.csv')
