import jsonlines
import argparse


def convert_to_fastchat(input_file, output_file):
    with jsonlines.open(input_file) as reader:
        data = list(reader)
        for idx, d in enumerate(data):
            if "text" not in d:
                d["text"] = d["test_input"]
            d["question_id"] = idx
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, required=True)
    parser.add_argument("--output_file", "-o", type=str, required=True)
    args = parser.parse_args()
    convert_to_fastchat(args.input_file, args.output_file)