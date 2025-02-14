import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import torch
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    max_length=1024,
    device_map="auto",
)

def get_llm_result(tokenizer, pipeline, prompt):
    sequences = pipeline(prompt,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    return sequences[0]["generated_text"].removeprefix(prompt)

df=pd.read_csv('results/combined_results.csv')
results=[]
for i,row in df.iterrows():
    if isinstance(row['articles'],float):
        results.append(0)
        continue
    with open("prompts/prompt_few_shot.txt", encoding="ascii", errors="ignore") as f:
        prompt = f.read() + row['articles'] + " \nAnswer: "
    result=get_llm_result(tokenizer, pipeline,prompt)
    if result.strip().startswith('V'):
        results.append(1)
        print(1)
    else:
        results.append(0)
        print(0)
print(results)
df['result']=results
df.to_csv('results/few_shot_combined_results.csv')