import jsonlines, os, argparse
from typing import List, Dict, Any, Set

def reflection_checker(datum: Dict[str, str], verbose: bool = False):
    if "reflection" not in datum or datum["reflection"] == "":
        try:
            output = datum["test_output"]
        except:
            print("Reflection not found:", datum)
            return False
    else:
        output = datum["reflection"]
    output = output.lower()
    if "yes" in output:
        return True
    if "accept" in output:
        return True
    if "agree" in output:
        return True
    if "the critique is valid" in output:
        return True
    if "the critique is helpful" in output:
        return True
    if "the critique is reasonable" in output:
        return True
    if "the critique is more ethically reasonable" in output:
        return True
    if "the critique is very helpful" in output:
        return True
    if verbose: print("Reflection not found:", output)
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", "-o", type=str, required=True)
    args = parser.parse_args()
    
    # Load the output file
    with jsonlines.open(args.output_file, "r") as f:
        lines = list(f)
    
    # Check if the reflection is there
    reflection_found = []
    for line in lines:
        reflection_found.append(reflection_checker(line))
    print(f"Reflection found: {sum(reflection_found)} / {len(reflection_found)}")

