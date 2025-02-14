import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import torch
import anthropic
client = anthropic.Anthropic()
df=pd.read_csv('results/combined_results.csv')
df=df[0:400]
results=[]
explanations=[]
for i,row in df.iterrows():
    if isinstance(row['articles'],float):
        results.append(0)
        continue
    with open("prompts/prompt_few_shot.txt", encoding="ascii", errors="ignore") as f:
        prompt = f.read() + row['articles'] + " \nAnswer: "
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    result=message.content[0].text
    print(result)
    if result.strip().startswith("V"):
        results.append(1)
        print(1)
    else:
        results.append(0)
        print(0)
    print(result)
df['result']=results
df.to_csv('results/few_shot_claude_combined_results.csv')