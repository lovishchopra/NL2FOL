"""
Python script to interpret results of SMT solver. It checks whether the output of the SMT file containing the negation of the formula is SAT or UNSAT. 
If it is SAT, then there exists a counter example. If it is UNSAT, then the actual sentence was valid.
For the counter example, we use an LLM to interpret the same.
"""
from llm import get_llm_result
import json
import sys

class SMTResults:
    """
    Class to find and interpret SMT results
    """
    def __init__(self, output_file_path, sentence_data_path):
        with open(output_file_path, "r") as f:
            results = f.read() + "\n"
        
        sat_or_unsat, counter_model = results.split("\n", 1)
        
        if "unsat" in sat_or_unsat:
            self.formula_type = "valid statement"
            self.counter_example = None
        else:
            self.formula_type = "logical fallacy"
            with open(sentence_data_path, "r") as f:
                data = json.load(f)
            with open("prompt_counter_example.txt", "r") as f:
                prompt = f.read().format(data["Claim"], data["Implication"], data["Referring expressions"], data["Properties"], data["Formula"], counter_model)

            self.counter_example = get_llm_result(prompt)
        
    def get_results(self, get_interpretation=True):
        """
        Print the SMT result and its interpretation
        """
        print("The given statement is a {0}".format(self.formula_type))
        if self.formula_type == "logical fallacy" and get_interpretation:
            print(self.counter_example)

if __name__ == "__main__":
    # Check if file paths are provided as a command-line argument
    if len(sys.argv) != 3:
        print("Usage: python interpret_smt_results.py <output_of_smt_file_path> <json to relevant sentence data with Claim, Implication, Referring expressions, Properties and Formula>")
        sys.exit(1)

    output_file_path = sys.argv[1]
    sentence_data_path = sys.argv[2]
    SMTResults(output_file_path, sentence_data_path).get_results()
    
        