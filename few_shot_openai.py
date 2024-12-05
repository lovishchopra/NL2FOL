import openai
import pandas as pd
import transformers
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

openai.api_key = "your_openai_key_here"

model = "gpt-4o"  # [gpt-4o, gpt-4o-mini, o1-preview, o1-mini]
method = "few_shot_cot"  # [zero_shot, few_shot, few_shot_cot]
input_file = "climate_run_results"  # [logic_run_results, climate_run_results]
num_rows = 10

df = pd.read_csv(f"results/{input_file}.csv")
df=df[0:num_rows]
results=[]
explanations=[]

for i,row in df.iterrows():
    if isinstance(row['articles'],float):
        results.append(0)
        continue
    with open(f"prompts/prompt_{method}.txt", encoding="ascii", errors="ignore") as f:
        prompt = f.read() + row['articles']
    chat_completion = openai.ChatCompletion.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    result=chat_completion.choices[0].message.content
    print(result)
    if "Valid" in result.strip():
        results.append(1)
        print(1)
    else:
        results.append(0)
        print(0)
    print(result)

df['result']=results
df.to_csv(f"results/{method}_{model}_{input_file}.csv")
