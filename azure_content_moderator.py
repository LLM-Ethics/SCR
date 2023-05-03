import jsonlines, json
import http.client, urllib.request, urllib.parse, urllib.error, base64
import numpy as np
import os, random
from tqdm import tqdm
import openai
from reflection_checker import reflection_checker
from typing import List, Dict, Any, Set

CONTENT_MODERATOR_ENDPOINT = "llm-testing-content-moderation.cognitiveservices.azure.com"
subscription_key = "<API_KEY>"

# Set up OpenAI API credentials by reading from config/config.json
with open("config/config.json", "r") as f:
    config = json.load(f)
    openai.api_key = config["openai_api_key"]


headers = {
    # Request headers
    'Content-Type': 'text/plain',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

params = urllib.parse.urlencode({
    # Request parameters
    'autocorrect': 'False',
    'PII': 'False',
    'classify': 'True',
})

def azure_moderator(text):
    try:
        conn = http.client.HTTPSConnection(CONTENT_MODERATOR_ENDPOINT)
        conn.request("POST", "/contentmoderator/moderate/v1.0/ProcessText/Screen?%s" % params, text, headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return json.loads(data)
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

def openai_moderator(all_text):
    response = openai.Moderation.create(
        input=all_text,
    )
    return response


def azure_moderate_all(input_file: str, output_file: str):
    with jsonlines.open(input_file) as reader:
        all_data = list(reader)
        random.shuffle(all_data)
    if os.path.exists(output_file):
        with jsonlines.open(output_file) as reader:
            completed_data = list(reader)
        completed_text = set([o["suggestion"] for o in completed_data])
        all_data = [d for d in all_data if d["suggestion"] not in completed_text][: 1000 - len(completed_data)]
    
    with jsonlines.open(output_file, "a") as f:
        for datum in tqdm(all_data):
            moderation = []
            for split in range(0, len(datum["suggestion"]), 1024):
                try:
                    response = azure_moderator(datum["suggestion"][split:split+1024])
                    moderation.append(response['Classification']['Category3']['Score'])
                except:
                    print("Error with ", datum["suggestion"][split:split+1024], " and response ", response)
            datum["moderation"] = np.mean(moderation)
            f.write(datum)

def openai_moderate_all(input_file: str, output_file: str):
    with jsonlines.open(input_file) as reader:
        all_data = list(reader)
        random.shuffle(all_data)
    if os.path.exists(output_file):
        with jsonlines.open(output_file) as reader:
            completed_data = list(reader)
        completed_text = set([o["suggestion"] for o in completed_data])
        all_data = [d for d in all_data if d["suggestion"] not in completed_text][: 1000 - len(completed_data)]
    
    with jsonlines.open(output_file, "a") as f:
        inputs = [datum["suggestion"] for datum in all_data]
        splitted_inputs = []
        for input in inputs:
            splitted_inputs.extend([input[i:i+2048] for i in range(0, len(input), 2048)])
        responses = openai_moderator(splitted_inputs)["results"]
        responses: List[Dict[str, Dict[str, float]]]
        for i, datum in enumerate(all_data):
            moderation = []
            for j in range(0, len(datum["suggestion"]), 2048):
                curr_response = responses[i * (len(datum["suggestion"]) // 2048) + j // 2048]
                # append the max item in curr_response[]"category_scores"]
                moderation.append(max(curr_response["category_scores"].values()))
            datum["moderation"] = np.mean(moderation)
            f.write(datum)


def differentiate(input_file:str):
    from scipy.stats import ttest_ind
    
    with jsonlines.open(input_file) as reader:
        all_data = list(reader)
    ethical_moderation_scores = [datum["moderation"] for datum in all_data if not reflection_checker({"reflection": datum["test_output"]})]
    unethical_moderation_scores = [datum["moderation"] for datum in all_data if reflection_checker({"reflection": datum["test_output"]})]
    # print statistics
    print("Num ethical: ", len(ethical_moderation_scores))
    print("Num unethical: ", len(unethical_moderation_scores))
    print("Ethical moderation scores: ", np.mean(ethical_moderation_scores), np.std(ethical_moderation_scores))
    print("Unethical moderation scores: ", np.mean(unethical_moderation_scores), np.std(unethical_moderation_scores))
    # statistical significance test
    print("Ethical vs Unethical: ", ttest_ind(ethical_moderation_scores, unethical_moderation_scores))
    # compute AUROC
    from sklearn.metrics import roc_auc_score
    y_true = [1] * len(ethical_moderation_scores) + [0] * len(unethical_moderation_scores)
    y_score = ethical_moderation_scores + unethical_moderation_scores
    print("AUROC: ", roc_auc_score(y_true, y_score))


if __name__ == "__main__":
    # azure_moderate_all("vicuna/chatglm_vicuna_reflection_output.jsonl", "vicuna/azure_moderation_chatglm_vicuna.jsonl")
    differentiate("vicuna/azure_moderation_chatglm_vicuna.jsonl")
    # openai_moderate_all("vicuna/chatglm_vicuna_reflection_output.jsonl", "vicuna/openai_moderation_chatglm_vicuna.jsonl")
    differentiate("vicuna/openai_moderation_chatglm_vicuna.jsonl")