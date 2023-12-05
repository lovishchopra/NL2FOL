# CS257_Project

Installation Instructions:
export PYTHONPATH=$GROUP_HOME/python/lib/python3.9/site-packages:$PYTHONPATH
export PATH=$GROUP_HOME/python/bin:$PATH
PYTHONUSERBASE=$GROUP_HOME/python pip3 install --user torch
export TRANSFORMERS_CACHE=$GROUP_HOME/cache
export HF_HOME=$GROUP_HOME/cache:

1. ml python/3.9.0
2. pip3 install transformers
3. pip3 install torch
4. huggingface-cli login: Login using your llama token
5. pip3 install accelerate

Run python files using:
python3 <filename>.py <relevant inputs>