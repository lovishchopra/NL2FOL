import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
filename='results/final_run_results.csv'
def get_results(label,preds):
    acc=accuracy_score(label[label==1],preds[label==1])
    prec=precision_score(label,preds)
    rec=recall_score(label,preds)
    f1=f1_score(label,preds)
    return acc, prec, rec, f1

if __name__=='__main__':
    df=pd.read_csv(filename)
    label=1-df['label']
    preds=pd.Categorical(df['result'],categories=['Valid','LF']).codes
    preds = np.where(preds == -1, 1, preds)
    print(get_results(label,preds))
    print(get_results(1-label,1-preds))