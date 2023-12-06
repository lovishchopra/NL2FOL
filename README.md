# CS257_Project

Instructions:
1. export PYTHONPATH=$GROUP_HOME/python/lib/python3.9/site-packages:$PYTHONPATH
2. export PATH=$GROUP_HOME/python/bin:$PATH
3. export TRANSFORMERS_CACHE=$GROUP_HOME/cache
4. export HF_HOME=$GROUP_HOME/cache:
5. ml python/3.9.0
6. If python file uses LLM, run this on terminal and put your huggingface authorization token: huggingface-cli login
7. To run the LLM in this repository, you need access to meta-llama/Llama-2-7b-chat-hf. Get access here: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 
8. Run file using:
    python3 <filename>.py <args>

To install a package, set the paths above first. Then run:
- PYTHONUSERBASE=$GROUP_HOME/python pip3 install --user <package_name>

Necessary packages to run the code:
- transformers
- torch
- accelerate

