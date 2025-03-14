# NL2FOL: Translating Natural Language to First Order Logic for Logical Fallacy Detection 

## Dependencies
- `transformers`
- `torch`
- `accelerate`

## Run Instructions
For converting natural language to first order logic on the given dataset, run:
```
python3 src/nl_to_fol.py --model_name <your_model_name> --nli_model_name <your_nli_model_name>  --run_name <run_name> --dataset <logic or logicclimate> --length <number of datapoints to sample from dataset>
```
For converting first order logic to SMT files and generate results, run:
```
python3 fol_to_cvc.py <file containing fol translations>
```
For getting the final result metrics, run:
```
python3 get_metrics.py <path to results csv>
```

To interpret the SMT results, run:
```
python3 interpret_smt_result.py <output_of_smt_file_path> <json to relevant sentence data with Claim, Implication, Referring expressions, Properties and Formula>
```

## Citation
[Paper](https://arxiv.org/abs/2405.02318)
