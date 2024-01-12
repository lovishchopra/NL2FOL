import pandas as pd
import sys
from cvc import CVCGenerator
import subprocess

if __name__ == "__main__":
    # Check if file paths are provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python fol_to_cvc.py <run name>")
        sys.exit(1)
    run_name = sys.argv[1]
    csv_file = "results/{}.csv".format(run_name)
    data = pd.read_csv(csv_file)
    fol = data['Logical Form 2']
    results = []
    for i in range(len(fol)):
        print(i, fol[i])
        try:
            script = CVCGenerator(fol[i].replace("ForAll", "forall").replace("ThereExists", "exists").replace("&","and").replace("~","not ")).generateCVCScript()
            with open("results/{}_smt/{}.smt2".format(run_name,i), "w") as f:
                f.write(script)
            with open("results/{}_smt/{}_out.txt".format(run_name,i), "w") as f:
                # Run CVC5 and capture output
                proc = subprocess.run(["./cvc4", "results/{}_smt/{}.smt2".format(run_name,i)], capture_output=True, text=True, check=True)
                proc_result = proc.stdout
                f.write(proc_result)
                if len(proc_result) == 0:
                    results.append("")
                result, _ = proc_result.split("\n", 1)
                if "unknown" in result:
                    script = CVCGenerator(fol[i].replace("ForAll", "forall").replace("ThereExists", "exists").replace("&", "and").replace("~", "not ")).generateCVCScript(finite_model_finding=True)
                    with open("results/{}_smt/{}.smt2".format(run_name,i), "w") as f2:
                        f2.write(script)
                    # Run CVC5 and capture output
                    proc = subprocess.run(["./cvc4", "results/{}_smt/{}.smt2".format(run_name,i)], capture_output=True, text=True, check=True)
                    proc_result = proc.stdout
                    f.write(proc_result)
                    if len(proc_result) == 0:
                        results.append("")
                    result, _ = proc_result.split("\n", 1)
                        
                if "unsat" in result:
                    results.append("Valid")
                elif "unknown" in result or "sat" in result:
                    results.append("LF")
                else:
                    results.append("")

        except Exception as e:
            print("Cannot run this statement : {}".format(e))
            results.append("")
            pass
    data['result'] = results
    data.to_csv("results/{}_results.csv".format(run_name))

