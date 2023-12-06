import subprocess
import sys

def run_cvc5(smt2_file_path, output_text_file_path):
    try:
        # Run CVC5 and capture output
        result = subprocess.run(["cvc4", smt2_file_path], capture_output=True, text=True, check=True)

        with open(output_text_file_path, "w") as f:
            f.write(result.stdout)

    except subprocess.CalledProcessError as e:
        print("Error running CVC4", e.stderr)

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 3:
        print("Usage: python run_sat_solver.py <smt2_file> <output_text_file>")
        sys.exit(1)

    smt2_file_path = sys.argv[1]
    output_text_file_path = sys.argv[2]
    run_cvc5(smt2_file_path, output_text_file_path)
