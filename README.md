# NL2FOL: Translating Natural Language to First Order Logic for Logical Fallacy Detection

Necessary packages to run the code:
- transformers
- torch
- accelerate

Run instructions:
- For converting natural language to first order logic on the given dataset: python3 nl_to_fol.py
- For converting first order logic to SMT files and generate results use: python3 fol_to_cvc.py <file containing fol translations>
- For getting the final result metrics run: python3 get_metrics.py
- To interpret the SMT results use: python3 interpret_smt_result.py <output_of_smt_file_path> <json to relevant sentence data with Claim, Implication, Referring expressions, Properties and Formula>

