import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import torch
from openai import OpenAI
client = OpenAI()
model="gpt-4"
df=pd.read_csv('results/combined_results.csv')
df=df[0:400]
results=[]
explanations=[]
for i,row in df.iterrows():
    if isinstance(row['articles'],float):
        results.append(0)
        continue
    with open("prompts/prompt_zero_shot.txt", encoding="ascii", errors="ignore") as f:
        prompt = f.read() + row['articles'] + " \nAnswer this question by implementing a solver function\n. def solver()\n#Let's write a Python program step by step, and then return the answer.\n"
    chat_completion = client.chat.completions.create(
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
df.to_csv('results/zero_shot_gpt4_combined_results.csv')