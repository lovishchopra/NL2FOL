import pandas as pd  
import numpy as np
import string
import sys

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

def process_data(data, col_name, output_file_path):
    data = data[data['annotator_labels'].apply(lambda x: x == [col_name])][["sentence1", "sentence2"]]
    data['sentence'] = data.apply(merge_with_random_phrase, axis=1)
    data = data.reset_index(drop=True)
    data.to_csv(output_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_json_nli_dataset.py jsonl_file_path")
        sys.exit(1)

    data = pd.read_json(path_or_buf=sys.argv[1], lines=True)
    print("Processing Entailments...")
    process_data(data, "entailment", "data/nli_entailments.csv")
    print("Processing Fallacies...")
    process_data(data, "contradiction", "data/nli_fallacies.csv")