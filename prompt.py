from transformers import AutoTokenizer
import transformers
import torch
import warnings
from cvc import CVCGenerator
warnings.filterwarnings("ignore")

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    max_length=1024,
    device_map="auto",
)

class NL2FOL:
    def __init__(self, sentence):
        self.sentence = sentence
        self.claim = None
        self.implication = None
        self.claim_ref_exp = None
        self.implication_ref_exp = None

    @staticmethod
    def get_llm_result(prompt):
        sequences = pipeline(prompt,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )

        return sequences[0]["generated_text"].removeprefix(prompt)

    def extract_claim_and_implication(self):
        with open("prompt_nl_to_ci.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read() + self.sentence
        
        result = NL2FOL.get_llm_result(prompt)

        for line in result.split("\n"):
            if 'Claim' in line:
                self.claim = line[line.find(':') + 1:]
            if 'Implication' in line:
                self.implication = line[line.find(':') + 1:]
        
        print("Claim: ", self.claim)
        print("Implication: ", self.implication)

    def get_referring_expressions(self):
        with open("prompt_referring_expressions.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read()

        result_claim = NL2FOL.get_llm_result(prompt + self.claim)
        for line in result_claim.split("\n"):
            if "Referring expressions" in line:
                self.claim_ref_exp = line[line.find(':') + 1:]
                break
        print("Referring Expressions, Claim:", self.claim_ref_exp)

        result_implication = NL2FOL.get_llm_result(prompt + self.implication)
        for line in result_implication.split("\n"):
            if "Referring expressions" in line:
                self.implication_ref_exp = line[line.find(':') + 1:]
                break
        print("Referring Expressions, Implication:", self.implication_ref_exp)

    def get_properties(self):
        with open("prompt_properties.txt", encoding="ascii", errors="ignore") as f:
            prompt = f.read()

        result_claim = NL2FOL.get_llm_result(prompt + self.claim)
        print(result_claim)

    def convert_to_first_order_logic(self):
        self.extract_claim_and_implication()
        self.get_referring_expressions()
        self.get_properties()

        return "exists x ( T(x) and L(x, cheese) ) -> forall y ( T(y) => L(y, cheese)) )" # Returning this just for now

class NL2SMT:
    def __init__(self, sentence):
        self.sentence = sentence
    
    def save_smt(self, file_path):
        first_order_logic = NL2FOL(self.sentence).convert_to_first_order_logic()
        script = CVCGenerator(first_order_logic).generateCVCScript()
        with open(file_path, "w") as file:
            file.write(script)

NL2SMT("A boy is jumping on skateboard in the middle of a red bridge. Thus, he does a skateboarding trick.").save_smt("skateboard.smt2")
