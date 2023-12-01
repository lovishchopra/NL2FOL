import pandas as pd  
import numpy as np
import string

joining_phrases = ["Hence, ", "Therefore, ", "Thus, ", "This implies that ", "Consequently, ",
                   "As a consequence, ", "It follows that "]

def merge_with_random_phrase(row):
    random_phrase = np.random.choice(joining_phrases)
    sentence1 = row['sentence1']
    if not sentence1.endswith(tuple(string.punctuation)):
        sentence1 += "."
    sentence2 = row['sentence2'][0].lower() + row['sentence2'][1:] 
    if not sentence2.endswith(tuple(string.punctuation)):
        sentence2 += "."
    return sentence1 + " " + random_phrase + sentence2


data = pd.read_json(path_or_buf="data.jsonl", lines=True)
data = data[data['annotator_labels'].apply(lambda x: x == ['entailment'])][["sentence1", "sentence2"]]
data['sentence'] = data.apply(merge_with_random_phrase, axis=1)
data = data.reset_index(drop=True)
data.to_csv("valid_sentences.csv")