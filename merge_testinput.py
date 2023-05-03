import random
import jsonlines
import argparse
import random
import os


def merge(args):
    with jsonlines.open(args.input1) as reader:
        input1 = list(reader)
        if "text" in input1[0]:
            for d in input1:
                d["test_input"] = d["text"]
                del d["text"]
    with jsonlines.open(args.input2) as reader:
        input2 = list(reader)
        if "text" in input2[0]:
            for d in input2:
                d["test_input"] = d["text"]
                del d["text"]
    outputs = input1 + input2
    random.shuffle(outputs)
    with jsonlines.open(args.output, "w") as writer:
        writer.write_all(outputs)

def recover(args):
    with jsonlines.open(args.merged_output) as reader:
        merged_output = list(reader)
        if "text" in merged_output[0]:
            for d in merged_output:
                d["test_input"] = d["text"]
                del d["text"]
    with jsonlines.open(args.input1) as reader:
        input1 = list(reader)
        input1_test_inputs = set()
        if "text" in input1[0]:
            for d in input1:
                d["test_input"] = d["text"]
                del d["text"]
                input1_test_inputs.add(d["test_input"])
    with jsonlines.open(args.input2) as reader:
        input2 = list(reader)
        input2_test_inputs = set()
        if "text" in input2[0]:
            for d in input2:
                d["test_input"] = d["text"]
                del d["text"]
                input2_test_inputs.add(d["test_input"])
    output1 = []
    output2 = []
    for d in merged_output:
        if d["test_input"] in input1_test_inputs:
            output1.append(d)
        elif d["test_input"] in input2_test_inputs:
            output2.append(d)
        else:
            print(d)
            raise ValueError("Test input not found in either input1 or input2.")
    
    # create directories if they don't exist
    os.makedirs(os.path.dirname(args.output1), exist_ok=True)
    os.makedirs(os.path.dirname(args.output2), exist_ok=True)
    with jsonlines.open(args.output1, "w") as writer:
        writer.write_all(output1)
    with jsonlines.open(args.output2, "w") as writer:
        writer.write_all(output2)

def check_completion(args):
    with jsonlines.open(args.input) as reader:
        input_data = list(reader)
        input_test_inputs = set()
        for d in input_data:
            if "text" in d:
                d["test_input"] = d["text"]
                del d["text"]
            input_test_inputs.add(d["test_input"])

    with jsonlines.open(args.output) as reader:
        output_data = list(reader)
        output_test_inputs = set()
        for d in output_data:
            if "text" in d:        
                d["test_input"] = d["text"]
                del d["text"]
            output_test_inputs.add(d["test_input"])
    
    print(f"Number of test inputs: {len(input_test_inputs)}")
    print(f"Number of test outputs: {len(output_test_inputs)}")

    if input_test_inputs == output_test_inputs:
        print("All test inputs have been processed.")
    else:
        missing_test_inputs = input_test_inputs - output_test_inputs
        print(f"Error: {len(missing_test_inputs)} test inputs have not been processed.")
    
        # if new_input is provided, create a new output file for missing inputs\
        if args.new_input:
            with jsonlines.open(args.new_input, "w") as writer:
                random.shuffle(input_data)
                for d in input_data:
                    if d["test_input"] in missing_test_inputs:
                        writer.write(d)



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input1", '-i1', type=str, required=True)
    merge_parser.add_argument("--input2", '-i2', type=str, required=True)
    merge_parser.add_argument("--output", '-o', type=str, required=True)

    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("--input1", '-i1', type=str, required=True)
    recover_parser.add_argument("--input2", '-i2', type=str, required=True)
    recover_parser.add_argument("--merged_output", '-m', type=str, required=True)
    recover_parser.add_argument("--output1", '-o1', type=str, required=True)
    recover_parser.add_argument("--output2", '-o2', type=str, required=True)

    check_completion_parser = subparsers.add_parser("check_completion")
    check_completion_parser.add_argument("--input", '-i', type=str, required=True)
    check_completion_parser.add_argument("--output", '-o', type=str, required=True)
    check_completion_parser.add_argument("--new_input", '-ni', type=str)

    args = parser.parse_args()

    if args.command == "merge":
        merge(args)
    elif args.command == "recover":
        recover(args)
    elif args.command == "check_completion":
        check_completion(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
