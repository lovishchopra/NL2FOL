# ExplainNLI: Translating Natural Language to First Order Logic for Logical Fallacy Detection

Instructions:
1. source setup_sherlock.sh
2. If python file uses LLM, run this on terminal and put your huggingface authorization token: huggingface-cli login
3. To run the LLM in this repository, you need access to meta-llama/Llama-2-7b-chat-hf. Get access here: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 
4. Run file using:
    python3 <filename>.py <args>

To install a package, set the paths above first. Then run:
- PYTHONUSERBASE=$GROUP_HOME/python pip3 install --user <package_name>

Necessary packages to run the code:
- transformers
- torch
- accelerate

Run instructions:
- For converting natural language to first order logic on the given dataset: python3 nl_to_fol.py
- For converting first order logic to SMT files and generate results use: python3 fol_to_cvc.py <file containing fol translations>
- For getting the final result metrics run: python3 get_metrics.py
- To interpret the SMT results use: python3 interpret_smt_result.py <output_of_smt_file_path> <json to relevant sentence data with Claim, Implication, Referring expressions, Properties and Formula>

