import os, argparse, jsonlines, pandas, random, json
from typing import Dict, List
from reflection_checker import reflection_checker
import pandas as pd

def situation_realism(input_file: str, output_file: str):
    with jsonlines.open(input_file, "r") as reader:
        data = list(reader)
    random.shuffle(data)
    subset = data[:300]
    # df = pandas.DataFrame(subset)
    # df.loc[:, "text"] = df["text"].apply(lambda x : x.replace('\n', ' '))
    # df = df[["text"]]
    # df.to_json(output_file, orient="records")
    json_data = []
    for datum in subset:
        json_data.append(
            {
                "input_values": {
                "text": datum["text"].replace('\n', ' ')
                }
            }
        )
    with open(output_file, "w") as f:
        json.dump(json_data, f)

def analyze_situation_realism(input_file: str):

    # read in the TSV file as a pandas dataframe
    df = pd.read_csv(input_file, sep="\t")

    results = {}
    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # print the text and the label
        if row["INPUT:text"] not in results:
            results[row["INPUT:text"]] = {
                "completely realistic": 0,
                "somewhat realistic": 0,
                "unrealistic": 0,
            }
        results[row["INPUT:text"]][row["OUTPUT:realism"]] += 1

    completely_realistic = 0
    somewhat_realistic = 0
    unrealistic = 0
    non_consensus = 0
    for text in results:
        if results[text]["completely realistic"] > results[text]["somewhat realistic"] and results[text]["completely realistic"] > results[text]["unrealistic"]:
            completely_realistic += 1
        elif results[text]["somewhat realistic"] > results[text]["completely realistic"] and results[text]["somewhat realistic"] > results[text]["unrealistic"]:
            somewhat_realistic += 1
        elif results[text]["unrealistic"] > results[text]["completely realistic"] and results[text]["unrealistic"] > results[text]["somewhat realistic"]:
            unrealistic += 1
        else:
            non_consensus += 1
    print(f"completely realistic: {completely_realistic}")
    print(f"somewhat realistic: {somewhat_realistic}")
    print(f"unrealistic: {unrealistic}")
    print(f"non consensus: {non_consensus}")


def response_plausibility(input_files: List[str], output_file: str):
    data = []
    selected_contexts = None
    for input_file in input_files:
        with jsonlines.open(input_file, "r") as reader:
            curr_data = list(reader)
            if selected_contexts is None:
                random.shuffle(curr_data)
                curr_data = curr_data[:100]
                curr_data = [{"source": input_file, **datum} for datum in curr_data]
                data.extend(curr_data)
                selected_contexts = set([datum["text"].replace('\n', ' ') if "text" in datum else datum["test_input"].replace('\n', ' ') for datum in curr_data])
            else:
                for datum in curr_data:
                    ctx = datum["text"].replace('\n', ' ') if "text" in datum else datum["test_input"].replace('\n', ' ')
                    if ctx in selected_contexts:
                        data.append(
                            {
                                "source": input_file,
                                **datum
                            }
                        )
        print(input_file)
        print(len(data))
        print(len(selected_contexts))

    print(len(data))
    print(len(selected_contexts))
    json_data = []
    for datum in data:
        # print(datum)
        json_data.append(
            {
                "input_values": {
                "context": datum["text"].replace('\n', ' ') if "text" in datum else datum["test_input"].replace('\n', ' '),
                "suggestion": datum["test_output"].replace('\n', ' ')
                }
            }
        )
    with open(output_file, "w") as f:
        json.dump(json_data, f)

def analyze_plausibility(input_file: str, reference_files: List[str]):
    reference_data = {}
    data = []
    for reference_file in reference_files:
        with jsonlines.open(reference_file, "r") as reader:
            curr_data = list(reader)
            curr_data = [{"reference": reference_file, **datum} for datum in curr_data]
            data.extend(curr_data)
    print(len(data))
    for datum in data:
        context = datum["text"].replace('\n', ' ') if "text" in datum else datum["test_input"].replace('\n', ' ') 
        suggestion = datum["test_output"].replace('\n', ' ')
        if context not in reference_data:
            reference_data[context] = {
                suggestion: [datum["reference"]]
            }
        else:
            if suggestion not in reference_data[context]:
                reference_data[context][suggestion] = [datum["reference"]]
            else:
                reference_data[context][suggestion].append(datum["reference"])

    print(len(reference_data))
    sources = reference_files
    # read in the TSV file as a pandas dataframe
    df = pd.read_csv(input_file, sep="\t")

    results = {source: {"completely plausible": 0, "somewhat plausible":0, "implausible": 0} for source in sources}
    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        sources = reference_data[row["INPUT:context"]][row["INPUT:suggestion"]]
        for source in sources:
            if "gpt-4" in source and "implau" in row["OUTPUT:plausibility"]: print(row["INPUT:suggestion"])
            results[source][row["OUTPUT:plausibility"]] += 1

    # print results
    for source in results:
        print(f"Source: {source}")
        print(f"completely plausible: {results[source]['completely plausible']}")
        print(f"somewhat plausible: {results[source]['somewhat plausible']}")
        print(f"implausible: {results[source]['implausible']}")
        # print sum
        print(f"sum: {results[source]['completely plausible'] + results[source]['somewhat plausible'] + results[source]['implausible']}")
        print()

def isEnglish(datum):
    for key in datum:
        s = datum[key]
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
    return True

def generate_positive(input_files: str, output_file: str):
    
    full_data = []
    for input_file in input_files:
        with jsonlines.open(input_file, "r") as reader:
            data = [
                {
                "context": datum["context"].replace('\n', ' '),
                "suggestion": datum["suggestion"].replace('\n', ' '),
                "critique": datum["critique"].replace('\n', ' '),
                "source": input_file
                }
                for datum in reader if "none" not in datum["critique"].lower() and reflection_checker({"reflection": datum["test_output"]})
            ]
            print(len(data))
            data = [datum for datum in data if isEnglish(datum)]
            print("After Excluding Non-English data", len(data))
            random.shuffle(data)
            data = data[:100]
            full_data.extend(data)
    json_data = []
    json_data_ref = []
    for datum in full_data:
        # print(datum)
        json_data.append(
            {
                "input_values": {
                "context": datum["context"],
                "suggestion": datum["suggestion"],
                "critique": datum["critique"],
                }
            }
        )
        json_data_ref.append(
            {
                "input_values": {
                "context": datum["context"],
                "suggestion": datum["suggestion"],
                "critique": datum["critique"],
                "source": datum["source"]
                }
            }
        )
    with open(output_file, "w") as f:
        json.dump(json_data, f)
    with open(output_file+".ref", "w") as f:
        json.dump(json_data_ref, f)            

def generate_negative(input_files: str, output_file: str):
    
    full_data = []
    for input_file in input_files:
        with jsonlines.open(input_file, "r") as reader:
            data = [
                {
                "context": datum["context"].replace('\n', ' '),
                "suggestion": datum["suggestion"].replace('\n', ' '),
                "critique": datum["critique"].replace('\n', ' '),
                "source": input_file
                }
                for datum in reader if not ("none" not in datum["critique"].lower() and reflection_checker({"reflection": datum["test_output"]}))
            ]
            print(len(data))
            data = [datum for datum in data if isEnglish(datum)]
            print("After Excluding Non-English data", len(data))
            random.shuffle(data)
            data = data[:100]
            full_data.extend(data)
    json_data = []
    json_data_ref = []
    for datum in full_data:
        # print(datum)
        json_data.append(
            {
                "input_values": {
                "context": datum["context"],
                "suggestion": datum["suggestion"],
                }
            }
        )
        json_data_ref.append(
            {
                "input_values": {
                "context": datum["context"],
                "suggestion": datum["suggestion"],
                "source": datum["source"]
                }
            }
        )
    with open(output_file, "w") as f:
        json.dump(json_data, f)
    with open(output_file+".ref", "w") as f:
        json.dump(json_data_ref, f)  

def analyze_positive(input_file: str, reference_file: str):
    # read in the TSV file as a pandas dataframe
    df = pd.read_csv(input_file, sep="\t")

    reference = {}
    sources = set()
    with open(reference_file, "r") as reader:
        data = json.load(reader)
        for datum in data:
            datum = datum["input_values"]
            ref_key = datum["context"] + datum["suggestion"] + datum["critique"]
            print(datum)
            reference[ref_key] = datum["source"]
            sources.add(datum["source"])
    
    count = {
        source :{
        "reasonable": 0,
        "unreasonable": 0,
        }
        for source in sources
    }

    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        ref_key = row["INPUT:context"] + row["INPUT:suggestion"] + row["INPUT:critique"]
        count[reference[ref_key]][row["OUTPUT:is_reasonable"]] += 1

    # print results
    for source in sources:
        print(f"Source: {source}")
        print(f"reasonable: {count[source]['reasonable']}")
        print(f"unreasonable: {count[source]['unreasonable']}")
        # print sum
        print(f"sum: {count[source]['reasonable'] + count[source]['unreasonable']}")
        print()

def analyze_negative(input_file: str, reference_file: str):
    # read in the TSV file as a pandas dataframe
    df = pd.read_csv(input_file, sep="\t")

    reference = {}
    sources = set()
    with open(reference_file, "r") as reader:
        data = json.load(reader)
        for datum in data:
            datum = datum["input_values"]
            ref_key = datum["context"] + datum["suggestion"]
            reference[ref_key] = datum["source"]
            sources.add(datum["source"])
    
    count = {
        source :{
        "ethical": 0,
        "unethical": 0,
        }
        for source in sources
    }

    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        ref_key = row["INPUT:context"] + row["INPUT:suggestion"]
        count[reference[ref_key]][row["OUTPUT:is_ethical"]] += 1

    # print results
    for source in sources:
        print(f"Source: {source}")
        print(f"ethical: {count[source]['ethical']}")
        print(f"unethical: {count[source]['unethical']}")
        # print sum
        print(f"sum: {count[source]['ethical'] + count[source]['unethical']}")
        print()


def analyze_positive2(input_file: str, reference_files: List[str]):
    # read in the TSV file as a pandas dataframe
    df = pd.read_csv(input_file, sep="\t")

    reference = {}
    sources = set(reference_files)

    for reference_file in reference_files:
        with jsonlines.open(reference_file, "r") as reader:
            data = list(reader)
            for datum in data:
                ref_key = datum["context"].replace('\n', ' ') + datum["suggestion"].replace('\n', ' ') + datum["critique"].replace('\n', ' ')
                reference[ref_key] = reference_file
    
    count = {
        source :{
        "reasonable": 0,
        "unreasonable": 0,
        }
        for source in sources
    }

    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        ref_key = row["INPUT:context"] + row["INPUT:suggestion"] + row["INPUT:critique"]
        count[reference[ref_key]][row["OUTPUT:is_reasonable"]] += 1

    # print results
    for source in sources:
        print(f"Source: {source}")
        print(f"reasonable: {count[source]['reasonable']}")
        print(f"unreasonable: {count[source]['unreasonable']}")
        # print sum
        print(f"sum: {count[source]['reasonable'] + count[source]['unreasonable']}")
        print()

def generate_refinement(input_files: List[str], output_file: str):
    
    full_data = []
    for input_file in input_files:
        with jsonlines.open(input_file, "r") as reader:
            data = [
                {
                "context": datum["context"].replace('\n', ' '),
                "suggestion": datum["suggestion"].replace('\n', ' '),
                "refine": datum["test_output"].replace('\n', ' '),
                "source": input_file
                }
                for datum in reader
            ]
            print(len(data))
            data = [datum for datum in data if isEnglish(datum)]
            print("After Excluding Non-English data", len(data))
            random.shuffle(data)
            data = data
            full_data.extend(data)
    json_data_ref = []
    for datum in full_data:
        json_data_ref.append(
            {
                "input_values": {
                "context": datum["context"],
                "suggestion": datum["suggestion"],
                "refine": datum["refine"],
                "source": datum["source"]
                }
            }
        )
    with open(output_file, "w") as f:
        json.dump(json_data_ref, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--input-files", type=list)
    parser.add_argument("--output-file", "-o", type=str)
    args = parser.parse_args()
    # situation_realism(args.input_file, args.output_file)
    # analyze_situation_realism(args.input_file)
    # analyze_plausibility(args.output_file, [
    #     "llama-7b/suggestion_output_all.jsonl",
    #     "gpt-neo/gptneo_suggestion_output.jsonl", 
    #     "chatglm/chatglm_suggestion_output.jsonl",
    #     "llama-13b/suggestion_output_all.jsonl",
    #     "vicuna/vicuna_suggestion_output.jsonl",
    #     "chatgpt/suggestion_output.jsonl",
    #     "opt-1.3B/suggestion_output_all.jsonl",
    #     "gpt-4/suggestion_output.jsonl"
    # ])
    # response_plausibility([
    #     "chatgpt/suggestion_output.jsonl",
    #     "llama-7b/suggestion_output_all.jsonl",
    #     "gpt-neo/gptneo_suggestion_output.jsonl", 
    #     "chatglm/chatglm_suggestion_output.jsonl",
    #     "llama-13b/suggestion_output_all.jsonl",
    #     "vicuna/vicuna_suggestion_output.jsonl",
    #     # "opt-1.3B/suggestion_output_all.jsonl"
    # ], args.output_file)

    # generate_positive(
    #     [
    #         # "chatglm/chatglm_reflection_output.jsonl",
    #         "gpt-neo/gptneo_reflection_output.jsonl",
    #         "chatgpt/vicuna_chatgpt_reflection_output.jsonl",
    #         # "llama-7b/llama7B_reflection_output.jsonl",
    #         # "llama-13b/vicuna_llama13B_reflection_output.jsonl",
    #         # "vicuna/chatgpt_vicuna_reflection_output.jsonl",
    #         # "vicuna/chatglm_vicuna_reflection_output.jsonl",
    #         # "vicuna/vicuna_self_reflection_output.jsonl",
    #     ], args.output_file
    # )

    # analyze_positive2("crowdsourced/positive-all.tsv",  [
    #         "chatglm/chatglm_reflection_output.jsonl",
    #         "gpt-neo/gptneo_reflection_output.jsonl",
    #         "chatgpt/vicuna_chatgpt_reflection_output.jsonl",
    #         "llama-7b/llama7B_reflection_output.jsonl",
    #         "llama-13b/vicuna_llama13B_reflection_output.jsonl",
    #         "vicuna/chatgpt_vicuna_reflection_output.jsonl",
    #         "vicuna/chatglm_vicuna_reflection_output.jsonl",
    #         "vicuna/vicuna_self_reflection_output.jsonl",
    #     ])

    # generate_negative(
    #     [
    #         # "chatglm/chatglm_reflection_output.jsonl",
    #         "gpt-neo/gptneo_reflection_output.jsonl",
    #         "chatgpt/vicuna_chatgpt_reflection_output.jsonl",
    #         # "llama-7b/llama7B_reflection_output.jsonl",
    #         # "llama-13b/vicuna_llama13B_reflection_output.jsonl",
    #         # "vicuna/chatgpt_vicuna_reflection_output.jsonl",
    #         # "vicuna/chatglm_vicuna_reflection_output.jsonl",
    #         # "vicuna/vicuna_self_reflection_output.jsonl",
    #     ], args.output_file
    # )

    # analyze_negative("crowdsourced/negative-gpt.tsv", "crowdsourced/negative-gpt.json.ref")

    generate_refinement([
        "vicuna/chatglm_vicuna_refine_output.jsonl"
    ], "crowdsourced/refinement-vicuna.json")