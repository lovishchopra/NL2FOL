#!/bin/bash
#SBATCH --job-name=logic-llama-8b-1000
#SBATCH --output=logs/logic-llama-8b-1000.%j.out
#SBATCH --error=logs/logic-llama-8b-1000.%j.err
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH -C GPU_SKU:H100_SXM5


export PYTHONPATH=$GROUP_HOME/python/lib/python3.9/site-packages:$PYTHONPATH
export PATH=$GROUP_HOME/python/bin:$PATH
export TRANSFORMERS_CACHE=$GROUP_HOME/cache
export HF_HOME=$GROUP_HOME/cache:
export HF_TOKEN=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
ml python/3.9.0

python3 --version
python3 nl_to_fol.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --nli_model_name "facebook/bart-large-mnli" --run_name "run_1" --length 1