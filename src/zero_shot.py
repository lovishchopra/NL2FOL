import pandas as pd
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
df=pd.read_csv('results/combined_results.csv')
candidate_labels = ['logical fallacy','logically valid sentence']
results=[]
for i,row in df.iterrows():
    if isinstance(row['articles'],float):
        results.append(0)
        continue
    res = classifier(row['articles'],candidate_labels)['labels'][0]
    print(res)
    if res=='logical fallacy':
        results.append(0)
    else:
        results.append(1)
print(results)
df['result']=results
df.to_csv('results/zero_shot_combined_results.csv')
    