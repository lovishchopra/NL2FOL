from llm import get_llm_result
import json

class SMTResults:
    def __init__(self, output_file_path, sentence_data_path):
        with open(output_file_path, "r") as f:
            results = f.read()
        
        self.sat_or_unsat, self.counter_model = results.split("\n", 1)
        
        if "unsat" in self.sat_or_unsat:
            self.counter_example = None
        else:
            with open(sentence_data_path, "r") as f:
                data = json.load(f)
            with open("prompt_counter_example.txt", "r") as f:
                prompt = f.read().format(data["Claim"], data["Implication"], data["Referring expressions"], data["Properties"], data["Formula"], self.counter_model)

            print(get_llm_result(prompt))

SMTResults("cheese_out.txt", "other_data.json")
    
        