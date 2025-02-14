import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_results(label, preds):
    acc = accuracy_score(label, preds)
    prec = precision_score(label, preds)
    rec = recall_score(label, preds)
    f1 = f1_score(label, preds)
    return acc, prec, rec, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model performance.')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.filename)
    label = df['label']
    preds=pd.Categorical(df['result'],categories=['LF','Valid']).codes
    preds = np.where(preds == -1, 1, preds)
    
    print(get_results(1- label, 1- preds))
