import subprocess

def run_cvc5(smt2_file):
    try:
        # Run CVC5 and capture output
        result = subprocess.run(["cvc5", smt2_file], capture_output=True, text=True, check=True)

        # Print the output
        print("CVC5 Output:")
        print(result.stdout)
    
    except subprocess.CalledProcessError as e:
        print("Error running CVC5", e.stderr)

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python run_sat_solver.py <smt2_file>")
        sys.exit(1)

    smt2_file_path = sys.argv[1]
    run_cvc5(smt2_file_path)
