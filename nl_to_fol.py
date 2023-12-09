from cvc import CVCGenerator
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
import torch
import ast
from helpers import label_values, first_non_empty_line, split_string_except_in_brackets, extract_propositional_symbols

class NL2FOL:
    """
    Class to convert natural language to first-order logical expression
    """
    def __init__(self, sentence, pipeline, tokenizer, nli_model, nli_tokenizer, debug=False):
        self.sentence = sentence
        self.claim = ""
        self.implication = ""
        self.claim_ref_exp = None
        self.implication_ref_exp = None
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.equal_entities = []
        self.subset_entities = []
        self.entity_mappings = {}
        self.claim_properties = []
        self.implication_properties = []
        self.property_implications = []
        self.claim_lf = ""
        self.implication_lf = ""
        self.final_lf = ""
        self.debug = debug

    def get_llm_result(self, prompt):
        sequences = self.pipeline(prompt,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return sequences[0]["generated_text"].removeprefix(prompt)

    def extract_claim_and_implication(self):
        with open("prompt_nl_to_ci.txt", encoding="ascii", errors="ignore") as f:
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
        with open("prompt_referring_expressions.txt", encoding="ascii", errors="ignore") as f:
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
        with open("prompt_properties.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read()
        prompt_template="Input {} " \
        "Referring Expressions: {} " \
        "Properties: ".format(self.claim,label_values(self.claim_ref_exp,self.entity_mappings))
        prompt1=prompt+prompt_template
        self.claim_properties = first_non_empty_line(self.get_llm_result(prompt1))
        prompt_template="Input {} " \
        "Referring Expressions: {} " \
        "Properties: ".format(self.implication,label_values(self.implication_ref_exp,self.entity_mappings))
        prompt1=prompt+prompt_template
        self.implication_properties = first_non_empty_line(self.get_llm_result(prompt1))
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
                p=self.get_nli_prob(c_p,i_p)
                p2=self.get_nli_prob(i_p,c_p)
                if p>70:
                    self.property_implications.append((c_p,i_p))
                if p2>70:
                    self.property_implications.append((i_p,c_p))
        if self.debug:
            print("property implications: ",self.property_implications)
        
    def get_fol(self):
        with open("prompt_fol.txt", encoding="ascii", errors="ignore") as f:
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
                premise=""
                hypothesis_list=["{} is a subset of {}".format(c_re,i_re),"{} is equal to {}".format(c_re,i_re),"{} is a subset of {}".format(i_re,c_re),'{} is not related to {}'.format(i_re,c_re)]
                # print(c_re,i_re)
                probs=self.get_nli_prob_list(premise, hypothesis_list)
                # print("Probs", probs)
                result_idx=probs.index(max(probs))
                if(max(probs)<65):
                    result="Unrelated"
                    # result_idx=3
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
                if claim_lf.find("exists {} ".format(map[subset]))==-1:
                    claim_lf="exists {} ".format(map[subset])+claim_lf
                if implication_lf.find("forall {} ".format(map[superset]))==-1:
                    implication_lf="forall {} ".format(map[superset])+implication_lf
            if map[subset] in implication_symbols and map[superset] in claim_symbols:
                if implication_lf.find("exists {} ".format(map[subset]))==-1:
                    implication_lf="exists {} ".format(map[subset])+implication_lf
                if claim_lf.find("forall {} ".format(map[superset]))==-1:
                    claim_lf="forall {} ".format(map[superset])+claim_lf
        self.final_lf="({}) -> ({})".format(claim_lf,implication_lf)
        if isinstance(self.property_implications,str):
            prop_imps=ast.literal_eval(self.property_implications)
        else:
            prop_imps=self.property_implications
        for (prop1,prop2) in prop_imps:
            lf="{} -> {}".format(prop1,prop2)
            lf_symbols=extract_propositional_symbols(lf)
            for symbol in lf_symbols:
                lf="forall {} ".format(symbol)+lf
            self.final_lf=self.final_lf+" & ("+lf+")"
        if self.debug:
            print("Final Lf= ",self.final_lf)

    def convert_to_first_order_logic(self):
        self.extract_claim_and_implication()
        self.get_referring_expressions()
        self.get_entity_relations()
        self.get_entity_mapping()
        self.get_properties()
        self.get_properties_relations()
        self.get_fol()
        self.get_final_lf()
        return self.final_lf

class NL2SMT:
    def __init__(self, sentence):
        self.sentence = sentence
    
    def save_smt(self, file_path):
        first_order_logic = NL2FOL(self.sentence).convert_to_first_order_logic()
        script = CVCGenerator(first_order_logic).generateCVCScript()
        with open(file_path, "w") as file:
            file.write(script)

def setup_dataset(length=100):
    df_fallacies=pd.read_csv('data/fallacies.csv')
    df_fallacies['label']=[0]*len(df_fallacies)
    df_fallacies=df_fallacies[['source_article','label','updated_label']]
    df_fallacies=df_fallacies.sample(length,random_state=683)
    df_valids=pd.read_csv('data/valid_sentences.csv')
    df_valids['label']=[1]*len(df_valids)
    df_valids=df_valids[['sentence','label']]
    df_valids=df_valids.sample(length,random_state=113)
    df = pd.concat([df_fallacies, df_valids])
    df['articles'] = df['source_article'].combine_first(df['sentence'])
    df = df.drop(['source_article', 'sentence'], axis=1)
    return df

if __name__ == '__main__':
    model = "meta-llama/Llama-2-7b-chat-hf"
    nli_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        max_length=1024,
        device_map="auto",
    )
    df=setup_dataset(length=10)
    final_lfs=[]
    for i,row in df.iterrows():
        print(row['articles'])
        nl2fol=NL2FOL(row['articles'],pipeline,tokenizer,nli_model,nli_tokenizer,debug=True)
        nl2fol.convert_to_first_order_logic()
        final_lfs.append(nl2fol.final_lf)
    df['Logical Form']=final_lfs
    df.to_csv('results/run1.csv',ignore_index=True)


    