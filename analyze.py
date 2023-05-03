import jsonlines, argparse, os
from reflection_checker import reflection_checker
from typing import List, Dict, Any, Set
from itertools import combinations

def valid_reflection(input_file: str):
    with jsonlines.open(input_file, "r") as reader:
        data = list(reader)
    count = 0
    valid_critique = 0
    for datum in data:
        if "none" not in datum["critique"].lower():
            valid_critique += 1
            if reflection_checker(datum):
                # if count < 20: 
                #     print(f"#{count+1}")
                #     print("Context:", datum["context"])
                #     print("Suggestion:", datum["suggestion"])
                    # print("Critique:", datum["critique"])

                count += 1
    print(f"Input file: {input_file}")
    print(f"Valid critique: {valid_critique} / {len(data)}")
    print(f"Valid reflection: {count} / {len(data)}")

def compare_critic(input_files: List[str]):
    critic = {
        _: set() for _ in input_files
    }
    valid_critic = set()
    valid_reflection = set()
    for input_file in input_files:
        with jsonlines.open(input_file, "r") as reader:
            data = list(reader)
        for datum in data:
            if "none" not in datum["critique"].lower():
                valid_critic.add(datum["context"])
                if reflection_checker(datum):
                    critic[input_file].add(datum["context"])
                    valid_reflection.add(datum["context"])
    # compute the mutual intersection
    for a, b in combinations(input_files, 2):
        print(f"{a} size: {len(critic[a])}")
        print(f"{b} size: {len(critic[b])}")
        print(f"Intersection: {len(critic[a].intersection(critic[b]))}")
    # compute the full intersection
    intersection = set.intersection(*[critic[_] for _ in input_files])
    print(f"Full intersection: {len(intersection)}")

    # print the valid critic/reflection
    print(f"Valid critic: {len(valid_critic)}")
    print(f"Valid reflection: {len(valid_reflection)}")



def est_f1():
    import re

    total_count = 19804

    table = '''\textbf{Performer LLM} & \textbf{\# Acc. Crit.} & \textbf{\# TP} & \textbf{\# FP} & \textbf{\# TN} & \textbf{\# FN} & \textbf{F1} \\ \hline
    GPT-Neo & 17363 & 79 & 21 & 75 & 25 \\ \hline
    Llama-a & 14707 & 83 & 17 & 84 & 16 \\ \hline
    Llama-b & 17852 & 86 & 14 & 77 & 23 \\ \hline
    ChatGLM & 17923 & 73 & 27 & 83 & 17 \\ \hline
    Vicuna  & 4368 & 73 & 27 & 78 & 22 \\\hline
    ChatGPT & 16980 & 68 & 32 & 89 & 11 \\ \hline
    Average &  &  &  &  &  \\ \hline'''

    # Remove LaTeX formatting
    table = re.sub(r'\\[a-z]+', '', table)


    # Split the table into rows
    rows = table.split('\n')[1:-1]

    # Initialize a list to store parsed data
    data = []
    total_pos, total_tp, total_fp, total_tn, total_fn, total_f1 = 0, 0, 0, 0, 0, 0
    for row in rows:
        values = row.split('&')
        if len(values) < 6:
            continue
        model, all_positive, tp, fp, tn, fn = [v.strip().replace("\\", "") for v in values]
        tp, fp, tn, fn = int(tp), int(fp), int(tn), int(fn)
        all_positive = int(all_positive)
        est_tp = tp * all_positive / 100 
        est_fp = all_positive - est_tp
        est_tn = tn * (total_count - all_positive) / 100
        est_fn = total_count - all_positive - est_tn

        precision = est_tp / (est_tp + est_fp)
        recall = est_tp / (est_tp + est_fn)
        f1 = 2 * precision * recall / (precision + recall)
        data.append((model, all_positive, tp, fp, tn, fn, f1))
        print(model, "precision", precision, "recall", recall, "f1", f1)
        # Accumulate the total scores
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        total_f1 += f1
        total_pos += all_positive
    
    # Calculate the average
    avg_tp = total_tp / len(data)
    avg_fp = total_fp / len(data)
    avg_tn = total_tn / len(data)
    avg_fn = total_fn / len(data)
    avg_f1 = total_f1 / len(data)
    avg_pos = total_pos / len(data)

    # Print the results
    print("Model\tPos\tTP\tFP\tFN\tF1")
    for model, pos, tp, fp, tn, fn, f1 in data:
        print(f"{model}\t{pos}\t{tp}\t{fp}\t{tn}\t{fn}\t{round(f1, 4)}")
    
    print(f"Average\t{avg_pos}\t{avg_tp}\t{round(avg_fp, 4)}\t{round(avg_tn, 4)}\t{round(avg_fn, 4)}\t{round(avg_f1, 4)}")

if __name__ == "__main__":
    file_list = [
        "chatglm/chatglm_reflection_output.jsonl",
        "gpt-neo/gptneo_reflection_output.jsonl",
        "llama-7b/llama7B_reflection_output.jsonl",
        "llama-13b/vicuna_llama13B_reflection_output.jsonl" ,
        "vicuna/chatglm_vicuna_reflection_output.jsonl",
        "vicuna/chatgpt_vicuna_reflection_output.jsonl",
        "vicuna/vicuna_self_reflection_output.jsonl",
        "chatgpt/vicuna_chatgpt_reflection_output.jsonl",
        "gpt-4/vicuna_gpt4_reflection_output.jsonl"
    ]
    for file in file_list:
        valid_reflection(file)
    # est_f1()

    # compare_critic([
    #     "vicuna/chatglm_vicuna_reflection_output.jsonl",
    #     "vicuna/chatgpt_vicuna_reflection_output.jsonl",
    #     "vicuna/vicuna_self_reflection_output.jsonl",
    # ])