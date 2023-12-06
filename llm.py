"""
Utility file for running Llama 7B model and getting result for a prompt
"""

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

def get_llm_result(prompt):
    sequences = pipeline(prompt,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    return sequences[0]["generated_text"].removeprefix(prompt)