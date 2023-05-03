import json
import jsonlines
import os
import sys
import argparse
from typing import List, Dict


def jsonl2lmflow_input(input_path: str, output_path: str) -> None:
    lmflow_output: Dict[str, List[Dict[str, str]]] = {"type": "text_only", "data": []}
    # Open the input file in read mode.
    with jsonlines.open(input_path, "r") as reader:
        # For each line in the input file, process the data and write it to the output file.
        for datum in reader:
            lmflow_output["instances"].append({"text": datum["text"]})
    # Open the output file in write mode.
    with open(output_path, "w") as writer:
        json.dump(lmflow_output, writer)

def post_process_llama_output(input_path: str, output_path:str) -> None:
    # Open the input file in read mode.
    with jsonlines.open(input_path, "r") as reader:
        # Open the output file in write mode.
        with jsonlines.open(output_path, "w") as writer:
            # For each line in the input file, process the data and write it to the output file.
            for datum in reader:
                writer.write({"test_output": datum["test_output"].replace("<unk>", "")})

def merge_jsonl_files(input_paths: Dict[str, str], output_path: str) -> None:
    lines = []
    readers = {stage: jsonlines.open(input_paths[stage], "r") for stage in input_paths}
    readers_lines = {stage: [] for stage in readers}
    for stage in readers:
        for line in readers[stage]:
            readers_lines[stage].append(line)
        readers[stage].close()
    for i in range(len(readers_lines["source"])):
        merged_line = {"text": "", "suggestion": "", "critique": "", "reflection": ""}
        for reader_key in readers_lines:
            line = readers_lines[reader_key][i]
            if reader_key == "source":
                merged_line["text"] = line["text"]
            elif reader_key == "suggestion":
                merged_line["suggestion"] = line["test_output"]
            elif reader_key == "critique":
                merged_line["critique"] = line["test_output"]
            elif reader_key == "reflection":
                merged_line["reflection"] = line["test_output"]
            else:
                raise ValueError(f"Invalid reader key: {reader_key}")
        lines.append(merged_line)
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(lines)

if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("function", type=str, help="Function to run")
    parser.add_argument("--input_path", "-i", type=str, help="Path to the input file")
    parser.add_argument("--output_path", "-o", type=str, help="Path to the output file")
    parser.add_argument("--source_path", "-src", type=str, help="Path to the source dataset file")
    parser.add_argument("--suggestion_path", "-sug", type=str, help="Path to the suggestion dataset file")
    parser.add_argument("--critique_path", "-c", type=str, help="Path to the critique dataset file")
    parser.add_argument("--reflection_path", "-r", type=str, help="Path to the reflection dataset file")
    args = parser.parse_args()
    if args.function == "jsonl2lmflow":
        if not args.input_path:
            raise ValueError("Input path not provided")
        if not args.output_path:
            raise ValueError("Output path not provided")
        jsonl2lmflow_input(args.input_path, args.output_path)
    if args.function == "post_process_llama_output":
        if not args.input_path:
            raise ValueError("Input path not provided")
        if not args.output_path:
            raise ValueError("Output path not provided")
        post_process_llama_output(args.input_path, args.output_path)
    if args.function == "merge_jsonl_files":
        input_files = {}
        if not args.source_path:
            raise ValueError("Source path not provided")
        input_files["source"] = args.source_path
        if args.suggestion_path:
            input_files["suggestion"] = args.suggestion_path
        if args.critique_path:
            input_files["critique"] = args.critique_path
        if args.reflection_path:
            input_files["reflection"] = args.reflection_path
        if not args.output_path:
            raise ValueError("Output path not provided")
        merge_jsonl_files(
            input_files,
            args.output_path,
        )
    else:
        raise ValueError("Invalid function")
