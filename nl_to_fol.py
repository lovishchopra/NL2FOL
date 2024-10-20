#!/share/software/user/open/python/3.9.0/bin/python3

from cvc import CVCGenerator
import ast
import json
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cvc import CVCGenerator
from helpers import *
from openai import OpenAI
import argparse

client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)

class NL2FOL:
    """
    Class to convert natural language to first-order logical expression
    """
    def __init__(self, sentence, model_type, pipeline, tokenizer, nli_model, nli_tokenizer, debug=False):
        self.sentence = sentence
        if not isinstance(self.sentence, str):
            self.sentence = ""
        self.model_type = model_type
        self.claim = ""
        self.implication = ""
        self.claim_ref_exp = ""
        self.implication_ref_exp = ""
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.equal_entities = []
        self.subset_entities = []
        self.entity_mappings = {}
        self.claim_properties = ""
        self.implication_properties = ""
        self.property_implications = []
        self.claim_lf = ""
        self.implication_lf = ""
        self.final_lf = ""
        self.final_lf2 = ""
        self.debug = debug

    def yield_data(data):
        for obj in data:
            yield obj

    def get_llm_result(self, prompt):
        if self.model_type=='llama':
            sequences = self.pipeline(prompt,
                do_sample=False,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            return sequences[0]["generated_text"].removeprefix(prompt)
        elif self.model_type=='gpt3.5':
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "prompt"}
            ]
            )
            return completion.choices[0].message.content

            

    def extract_claim_and_implication(self):
        with open("prompts/prompt_nl_to_ci.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read() + self.sentence
        result = self.get_llm_result(prompt)

        for line in result.split("\n"):
            if 'Claim' in line:
                self.claim = line[line.find(':') + 1:]
            if 'Implication' in line:
                self.implication = line[line.find(':') + 1:]
        if self.debug:
            print("Claim: ", self.claim)
            print("Implication: ", self.implication)

    def get_referring_expressions(self):
        with open("prompts/prompt_referring_expressions.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read()

        result_claim = self.get_llm_result(prompt + self.claim)
        for line in result_claim.split("\n"):
            if "Referring expressions" in line:
                self.claim_ref_exp = line[line.find(':') + 1:]
                break
        if self.debug:
            print("Referring Expressions, Claim:", self.claim_ref_exp)

        result_implication = self.get_llm_result(prompt + self.implication)
        for line in result_implication.split("\n"):
            if "Referring expressions" in line:
                self.implication_ref_exp = line[line.find(':') + 1:]
                break
        if self.debug:
            print("Referring Expressions, Implication:", self.implication_ref_exp)

    def get_properties(self):
        if(self.claim=="" or isinstance(self.claim_ref_exp,float) or isinstance(self.implication_ref_exp,float)):
            return
        with open("prompts/prompt_properties.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read()
        prompt_template="Input {} " \
        "Referring Expressions: {} " \
        "Properties: ".format(self.claim,label_values(self.claim_ref_exp,self.entity_mappings))
        prompt1=prompt+prompt_template
        self.claim_properties = first_non_empty_line(self.get_llm_result(prompt1))
        with open("prompts/prompt_properties2.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read()
        prompt_template="Input {}" \
        "Referring Expressions {}" \
        "Properties {}" \
        "Now extract the properties for the following input: " \
        "Input {} " \
        "Referring Expressions: {} " \
        "Properties: ".format(self.claim,label_values(self.claim_ref_exp,self.entity_mappings),self.claim_properties,self.implication,label_values(self.implication_ref_exp,self.entity_mappings))
        prompt1=prompt+prompt_template
        self.implication_properties = first_non_empty_line(self.get_llm_result(prompt1))
        self.claim_properties, self.implication_properties = fix_inconsistent_arities(split_string_except_in_brackets(self.claim_properties,','),split_string_except_in_brackets(self.implication_properties,','))
        if self.debug:
            print("Claim Properties: ", self.claim_properties)
            print("Implication Proeprties ", self.implication_properties)

    def get_properties_relations(self):
        if(self.claim=="" or isinstance(self.claim_ref_exp,float) or isinstance(self.implication_ref_exp,float)):
            return
        claim_properties=split_string_except_in_brackets(self.claim_properties, ',')
        implication_properties=split_string_except_in_brackets(self.implication_properties, ',')
        for c_p in claim_properties:
            for i_p in implication_properties:
                predicate1= c_p.split('(')[0]
                predicate2 = i_p.split('(')[0]
                if(predicate1.strip()==predicate2.strip()):
                    continue
                p=self.get_nli_prob(replace_variables(self.entity_mappings,c_p),replace_variables(self.entity_mappings,i_p))
                p2=self.get_nli_prob(replace_variables(self.entity_mappings,i_p),replace_variables(self.entity_mappings,c_p))
                if p>70:
                    self.property_implications.append((c_p,i_p))
                if p2>70:
                    self.property_implications.append((i_p,c_p))
        if self.debug:
            print("property implications: ",self.property_implications)
        
    def get_fol(self):
        with open("prompts/prompt_fol.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read()
        if(self.claim=="" or isinstance(self.claim_ref_exp,float) or isinstance(self.implication_ref_exp,float)):
            return
        prompt_template=" Input {} " \
        "Referring Expressions: {} " \
        "Properties: {} " \
        "Logical Form:".format(self.claim,label_values(self.claim_ref_exp,self.entity_mappings),self.claim_properties)
        prompt1=prompt+prompt_template
        self.claim_lf=first_non_empty_line(self.get_llm_result(prompt1))
        prompt_template=" Input {} " \
        "Referring Expressions: {} " \
        "Properties: {} " \
        "Logical Form: ".format(self.implication,label_values(self.implication_ref_exp,self.entity_mappings),self.implication_properties)

        prompt1=prompt+prompt_template
        self.implication_lf=first_non_empty_line(self.get_llm_result(prompt1))
        self.claim_lf=remove_text_after_last_parenthesis(self.claim_lf)
        self.implication_lf=remove_text_after_last_parenthesis(self.implication_lf)
        if self.debug:
            print("Claim Lf: ", self.claim_lf)
            print("Impliation Lf: ",self.implication_lf)

    def get_nli_prob(self, premise, hypothesis):
        input_ids = self.nli_tokenizer.encode(premise, hypothesis, return_tensors='pt')
        logits = self.nli_model(input_ids)[0]
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_prob = probs[:,1].item() * 100
        return true_prob

    def get_nli_prob_list(self, premise, hypothesis_list):
        result=[]
        for hyp in hypothesis_list:
            result.append(self.get_nli_prob(premise, hyp))
        return result
    
    def get_entity_relations(self):
        if(self.claim=="" or isinstance(self.claim_ref_exp,float) or isinstance(self.implication_ref_exp,float)):
            return
        claim_res=self.claim_ref_exp.split(",")
        implication_res=self.implication_ref_exp.split(",")
        for c_re in claim_res:
            for i_re in implication_res:
                if c_re.strip().lower() == i_re.strip().lower():
                    self.equal_entities.append((c_re,i_re))
                    continue
                premise=""
                hypothesis_list=["{} is a subset of {}".format(c_re,i_re),"{} is equal to {}".format(c_re,i_re),"{} is a subset of {}".format(i_re,c_re),'{} is not related to {}'.format(i_re,c_re)]
                probs=self.get_nli_prob_list(premise, hypothesis_list)
                result_idx=probs.index(max(probs))
                if(max(probs)<65):
                    result="Unrelated"
                else:
                    result=hypothesis_list[result_idx]
                    if result_idx==0:
                        self.subset_entities.append((c_re,i_re))
                    elif result_idx==1:
                        self.equal_entities.append((c_re,i_re))
                    elif result_idx==2:
                        self.subset_entities.append((i_re,c_re))
        if self.debug:
            print("Subset entities: ", self.subset_entities)
            print("Equal entities: ", self.equal_entities)

    def get_entity_mapping(self):
        current_char='a'
        for (s1,s2) in self.equal_entities:
            if s1 in self.entity_mappings:
                self.entity_mappings[s2]=self.entity_mappings[s1]
            elif s2 in self.entity_mappings:
                self.entity_mappings[s1]=self.entity_mappings[s2]
            else:
                self.entity_mappings[s1]=current_char
                self.entity_mappings[s2]=current_char
                current_char=chr(ord(current_char)+1)
        for s in self.claim_ref_exp.split(','):
            if s not in self.entity_mappings:
                self.entity_mappings[s]=current_char
                current_char=chr(ord(current_char)+1)
        for s in self.implication_ref_exp.split(','):
            if s not in self.entity_mappings:
                self.entity_mappings[s]=current_char
                current_char=chr(ord(current_char)+1)
        if self.debug:
            print("Mappings: ", self.entity_mappings)
        
    def get_final_lf(self):
        if isinstance(self.entity_mappings,float):
            return
        if isinstance(self.entity_mappings,str):
            map=ast.literal_eval(self.entity_mappings)
        else:
            map=self.entity_mappings
        claim_symbols=extract_propositional_symbols(self.claim_lf)
        implication_symbols=extract_propositional_symbols(self.implication_lf)
        claim_lf=self.claim_lf
        implication_lf=self.implication_lf
        if isinstance(self.subset_entities,str):
            subsets=ast.literal_eval(self.subset_entities)
        else:
            subsets=self.subset_entities
        for (subset,superset) in subsets:
            if map[subset] in claim_symbols and map[superset] in implication_symbols:
                if claim_lf.find("exists {} ".format(map[subset]))==-1 and claim_lf.find("forall {} ".format(map[subset]))==-1:
                    claim_lf="exists {} ({})".format(map[subset],claim_lf)
                if implication_lf.find("forall {}".format(map[superset]))==-1 and implication_lf.find("forall {} ".format(map[subset]))==-1:
                    implication_lf="forall {} ({})".format(map[superset],implication_lf)
            if map[subset] in implication_symbols and map[superset] in claim_symbols:
                if implication_lf.find("exists {} ".format(map[subset]))==-1 and implication_lf.find("forall {} ".format(map[subset]))==-1:
                    implication_lf="exists {} ({})".format(map[subset],implication_lf)
                if claim_lf.find("forall {} ".format(map[superset]))==-1 and claim_lf.find("exists {} ".format(map[subset]))==-1:
                    claim_lf="forall {} ({})".format(map[superset],claim_lf)
        if isinstance(self.property_implications,str):
            prop_imps=ast.literal_eval(self.property_implications)
        else:
            prop_imps=self.property_implications
        for (prop1,prop2) in prop_imps:
            lf="{} -> {}".format(prop1,prop2)
            lf_symbols=extract_propositional_symbols(lf)
            for symbol in lf_symbols:
                lf="forall {} ".format(symbol)+'('+lf+')'
            claim_lf=claim_lf+" & ("+lf+")"
        self.final_lf="({}) -> ({})".format(claim_lf,implication_lf)
        if self.debug:
            print("Final Lf= ",self.final_lf)
    
    def get_final_lf2(self):
        if isinstance(self.entity_mappings,float):
            return
        if isinstance(self.entity_mappings,str):
            map=ast.literal_eval(self.entity_mappings)
        else:
            map=self.entity_mappings
        claim_symbols=extract_propositional_symbols(self.claim_lf)
        implication_symbols=extract_propositional_symbols(self.implication_lf)
        claim_lf=self.claim_lf
        implication_lf=self.implication_lf
        if isinstance(self.subset_entities,str):
            subsets=ast.literal_eval(self.subset_entities)
        else:
            subsets=self.subset_entities
        for symbol in claim_symbols:
            if claim_lf.find("exists {} ".format(symbol))==-1:
                    claim_lf="exists {} ({})".format(symbol,claim_lf)
        for symbol in implication_symbols:
            if implication_lf.find("exists {}".format(symbol))==-1:
                    implication_lf="exists {} ({})".format(symbol,implication_lf)
        if isinstance(self.property_implications,str):
            prop_imps=ast.literal_eval(self.property_implications)
        else:
            prop_imps=self.property_implications
        current_char=chr(ord('a') + len(self.entity_mappings))
        for (prop1,prop2) in prop_imps:
            prop1,prop2,current_char=substitute_variables(prop1,prop2,current_char)
            lf="{} -> {}".format(prop1,prop2)
            lf_symbols=extract_propositional_symbols(lf)
            for symbol in lf_symbols:
                lf="forall {} ".format(symbol)+"("+lf+")"
            claim_lf=claim_lf+" & ("+lf+")"
        self.final_lf2="({}) -> ({})".format(claim_lf,implication_lf)
        if self.debug:
            print("Final Lf2= ",self.final_lf2)

    def convert_to_first_order_logic(self):
        try:
            self.extract_claim_and_implication()
            self.get_referring_expressions()
            self.get_entity_relations()
            self.get_entity_mapping()
            self.get_properties()
            self.get_properties_relations()
            self.get_fol()
            self.apply_heuristics()
            self.get_final_lf()
            self.get_final_lf2()
            return self.final_lf,self.final_lf2
        except Exception as e:
            print(f"Failed with error {e}")
            return "",""
    
    def apply_heuristics(self):
        self.claim_lf = self.claim_lf.replace('->','&')
        self.claim_lf = self.claim_lf.replace('&','and')
        self.claim_lf = self.claim_lf.replace('|','or')
        self.implication_lf = self.implication_lf.replace('->','&')
        self.implication_lf = self.implication_lf.replace('&','and')
        self.implication_lf = self.implication_lf.replace('|','or')
        if self.debug:
            print("Updated Claim Lf= ",self.claim_lf)
            print("Updated Implication Lf=",self.implication_lf)

def setup_dataset(fallacy_set='logic',length=100):
    print("called")
    if fallacy_set=='logic':
        df_fallacies=pd.read_csv('data/fallacies.csv')
        df_fallacies['label']=[0]*len(df_fallacies)
        df_fallacies=df_fallacies[['source_article','label','updated_label']]
        df_fallacies=df_fallacies.sample(length,random_state=683)
    elif fallacy_set=='logicclimate':
        df_fallacies=pd.read_csv('data/fallacies_climate.csv')
        df_fallacies['label']=[0]*len(df_fallacies)
        df_fallacies=df_fallacies[['source_article','logical_fallacies','label']]
        df_fallacies=df_fallacies.sample(length,random_state=683)
    elif fallacy_set=='nli':
        df_fallacies=pd.read_csv('data/nli_fallacies_test.csv')
        df_fallacies['label']=[0]*len(df_fallacies)
        df_fallacies=df_fallacies[['sentence','label']]
        df_fallacies=df_fallacies.sample(length,random_state=683)
    df_valids=pd.read_csv('data/nli_entailments_test.csv')
    df_valids['label']=[1]*len(df_valids)
    df_valids=df_valids[['sentence','label']]
    df_valids=df_valids.sample(length,random_state=113)
    df = pd.concat([df_fallacies, df_valids])
    df = df.reset_index(drop=True)
    df['articles'] = df['source_article'].combine_first(df['sentence'])
    df = df.drop(['source_article', 'sentence'], axis=1)
    return df

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available. GPUs are accessible.")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            compute_capability = torch.cuda.get_device_capability(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"Name: {gpu_name}")
            print(f"Compute Capability: {compute_capability}")
            print(f"Total Memory: {total_memory:.2f} GB")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 2):.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / (1024 ** 2):.2f} MB")
    parser = argparse.ArgumentParser(description="Run text generation and logic conversion pipeline")
    parser.add_argument('--model_name', type=str, required=True, help="Model name for text generation pipeline")
    parser.add_argument('--nli_model_name', type=str, required=True, help="Model name for NLI")
    parser.add_argument('--run_name', type=str, required=True, help="Run name for saving results")
    parser.add_argument('--length', type=int, required=True, help="Length for dataset setup")
    args = parser.parse_args()
    # Initialize models and tokenizers
    model = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model_name)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        max_length=1024,
        device_map="auto",
    )
    # Setup dataset
    df = setup_dataset(fallacy_set='logic', length=args.length)
    df.to_csv('dataset.csv', index=False)
    final_lfs=[]
    final_lfs2=[]
    count=0
    for i,row in df.iterrows():
        print(count)
        nl2fol=NL2FOL(row['articles'],'llama',pipeline,tokenizer,nli_model,nli_tokenizer,debug=True)
        nl2fol.convert_to_first_order_logic()
        final_lfs.append(nl2fol.final_lf)
        final_lfs2.append(nl2fol.final_lf2)
    #     results_dict={}
    #     results_dict['Claim']=nl2fol.claim
    #     results_dict['Implication']=nl2fol.implication
    #     results_dict['Referring expressions']=nl2fol.claim_ref_exp+" "+nl2fol.implication_ref_exp
    #     results_dict['Properties']=nl2fol.claim_properties+" "+nl2fol.implication_properties
    #     results_dict['Formula']=nl2fol.final_lf2
    #     json_object = json.dumps(results_dict, indent=4)
    #     with open("results/{}/{}.json".format(run_name,count), "w") as outfile:
    #         outfile.write(json_object)
    #     count=count+1
    df['Logical Form']=final_lfs
    df['Logical Form 2']=final_lfs2
    df.to_csv(f'results/{args.run_name}.csv',index=False)


    