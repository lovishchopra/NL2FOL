import pandas as pd
import sys
from cvc import CVCGenerator
import subprocess

if __name__ == "__main__":
    # Check if file paths are provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python fol_to_cvc.py <csv file with results>")
        sys.exit(1)

    csv_file = sys.argv[1]
    data = pd.read_csv(csv_file)['Logical Form']
    for i in range(len(data)):
        print(i, data[i])
        try:
            script = CVCGenerator(data[i].replace("ForAll", "forall").replace("ThereExists", "exists")).generateCVCScript()
            with open("results/run1_smt/{0}.smt2".format(i), "w") as f:
                f.write(script)
            with open("results/run1_smt/{0}_out.txt".format(i), "w") as f:
                # Run CVC5 and capture output
                result = subprocess.run(["cvc4", "results/run1_smt/{0}.smt2".format(i)], capture_output=True, text=True, check=True)
                f.write(result.stdout)
        except:
            pass
