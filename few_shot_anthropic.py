import anthropic
import pandas as pd
import torch
import transformers

from transformers import AutoModelForSequenceClassification, AutoTokenizer

client = anthropic.Anthropic(api_key="your_key_here")

model = "claude-3-5-sonnet-20241022"  # [claude-3-opus, claude-3.5-sonnet]
method = "few_shot"  # [zero_shot, few_shot, few_shot_cot]
input_file = "logic_run_results"  # [logic_run_results, climate_run_results]
num_rows = 200

df = pd.read_csv(f"results/{input_file}.csv")
df=df[0:num_rows]
results=[]
explanations=[]

for i,row in df.iterrows():
    if isinstance(row['articles'], float):
        results.append(0)
        explanations.append("No article content")
        continue
    with open(f"prompts/prompt_{method}.txt", encoding="ascii", errors="ignore") as f:
        prompt = f.read() + row['articles'] + " \nAnswer: "
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        result=message.content[0].text
        explanations.append(result)
    except Exception as e:
        print(e)
        result="500 ERROR RESULT"
    print(result)
    if result.strip().startswith("V"):
        results.append(1)
        print(1)
    else:
        results.append(0)
        print(0)
    print(result)

df['result']=results
df['explanation']=explanations
df.to_csv(f'results/{method}_{model}_{input_file}.csv')
