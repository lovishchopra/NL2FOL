from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import pandas as pd

def get_nli_prob(tokenizer, model, premise, hypothesis):
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    logits = model(input_ids)[0]
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    true_prob = probs[:,1].item() * 100
    return true_prob

def get_dataset(length=100):
    df_fallacies=pd.read_csv('data/nli_fallacies_test.csv')
    df_fallacies['label']=[0]*len(df_fallacies)
    df_fallacies=df_fallacies[['sentence1','sentence2','label']]
    df_fallacies=df_fallacies.sample(length,random_state=683)
    df_valids=pd.read_csv('data/nli_entailments_test.csv')
    df_valids['label']=[1]*len(df_valids)
    df_valids=df_valids[['sentence1','sentence2','label']]
    df_valids=df_valids.sample(length,random_state=113)
    df = pd.concat([df_fallacies, df_valids])
    return df

if __name__ == '__main__':
    df=get_dataset()
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    results = []
    for i,row in df.iterrows():
        prob=get_nli_prob(tokenizer,model,row['sentence1'],row['sentence2'])
        if prob>0.5:
            results.append(1)
        else:
            results.append(0)
    df['result']=results
    df.to_csv('results/bart_nli_run.csv')