from transformers import AutoTokenizer
import transformers
import torch
import warnings
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
            prompt = f.read() + self.claim + "\n" + self.implication

        result = NL2FOL.get_llm_result(prompt)
        print(result)

    def convert_to_first_order_logic(self):
        self.extract_claim_and_implication()
        self.get_referring_expressions()

NL2FOL("A boy is jumping on skateboard in the middle of a red bridge. Thus, he does a skateboarding trick.").convert_to_first_order_logic()
# sequences = pipeline(prompt,
#     do_sample=False,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id
# )

# print(sequences)
