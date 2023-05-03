import argparse
import openai, time, json, os
import jsonlines, tqdm, copy
from typing import List, Dict, Any, Set
import multiprocessing as mp

# Set up OpenAI API credentials by reading from config/config.json
with open("config/config.json", "r") as f:
    config = json.load(f)
    openai.api_key = config["openai_api_key"]

# Define constants for retrying requests
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2
MODEL = "chatgpt"

def find_first_digit(s):
    return next((c for c in s if c.isdigit()), None)

def chatgpt(text):
    """
    Generates a response to the given input text using the Chat model in OpenAI's GPT-3.5 API.

    Args:
        text (str): The input text to generate a response to.

    Returns:
        str: The generated response.

    Raises:
        Exception: If there is an error while making the API request.
    """
    for retry in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", temperature=1, 
                messages=[
                        {"role": "system", "content": "You should adhere to the instruction of the user."},
                        {"role": "user", "content": text},
                    ]
                )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                print(f"Error making Chat request: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise e

def process_one(prompt_text, output_file, lock):
    """
    Processes one prompt and writes the output to the output file.

    Args:
        prompt_text (str): The prompt text to process.
        output_file (str): The path to the output file.
        lock (Lock): The lock to use to write to the output file.
    """
    # Generate the output
    output = chatgpt(prompt_text)
    # Write the output to the output file
    with lock:
        with jsonlines.open(output_file, "a") as f:
            f.write({
                "text": prompt_text,
                "output": output
            })

def compare_suggestion(initial_suggestion_path, refined_suggestion_path, output_file):
    """
    Compares the initial suggestion with the refined suggestion and returns the accuracy.

    Args:
        initial_suggestion_path (str): The path to the initial suggestion file.
        refined_suggestion_path (str): The path to the refined suggestion file.
        output_file (str): The path to the output file.

    Returns:
        float: The accuracy of the refined suggestion.
    """
    # Load the initial suggestion file
    with jsonlines.open(initial_suggestion_path, "r") as f:
        initial_suggestions = list(f)
        suggestion_index = {suggestion["test_input"]: i for i, suggestion in enumerate(initial_suggestions)}
    # Load the refined suggestion file
    with jsonlines.open(refined_suggestion_path, "r") as f:
        refined_suggestions = list(f)[:100]
    with open("prompts/compare_suggestion.prompt", "r") as f:
        prompt = f.read()
    if os.path.exists(output_file):
        with jsonlines.open(output_file, "r") as f:
            completed_comparisons = set([comparison["text"] for comparison in list(f)])
    else:
        completed_comparisons = set()
    
    # Compute the accuracy
    comparisons = []
    for refined_suggestion in refined_suggestions:
        if "text" not in refined_suggestion:
            context = ""
            for init_ctx in suggestion_index.keys():
                if init_ctx in refined_suggestion["test_input"]:
                    context = init_ctx
                    break
        else:
            context = refined_suggestion["text"]
        initial_suggestion = initial_suggestions[suggestion_index[context]]
        init_sugg_text = initial_suggestion["test_output"]
        refined_sugg_text = refined_suggestion["test_output"]
        prompt_text = prompt.replace("$context$", context)
        prompt_text = prompt_text.replace("$init_suggestion$", init_sugg_text)
        prompt_text = prompt_text.replace("$refined_suggestion$", refined_sugg_text)
        if prompt_text not in completed_comparisons: comparisons.append(prompt_text)
    # Create a multiprocessing pool
    pool: mp.Pool = mp.Pool(processes=10)
    # Create a Manager to share the lock between processes
    manager = mp.Manager()
    lock = manager.Lock()

    # Use a partial function to pass the output_file and lock arguments to process_datum
    from functools import partial
    process_datum_with_output_file_and_lock = partial(process_one, output_file=output_file, lock=lock)

    # Rewrite the data in parallel and append the output lines to the output file dynamically
    for _ in tqdm.tqdm(pool.imap(process_datum_with_output_file_and_lock, comparisons), total=len(comparisons)):
        pass

def compute_accuracy(output_file):
    """
    Computes the accuracy of the refined suggestions.

    Args:
        output_file (str): The path to the output file.

    Returns:
        float: The accuracy of the refined suggestions.
    """
    # Load the output file
    with jsonlines.open(output_file, "r") as f:
        comparisons = list(f)
    # Compute the accuracy
    num_correct = 0
    num_incorrect = 0
    for comparison in comparisons:
        if find_first_digit(comparison["output"]) == "2":
            num_correct += 1
        elif find_first_digit(comparison["output"]) == "1":
            num_incorrect += 1
        else:
            print(comparison["output"])
    print(f"Correct: {num_correct}")
    print(f"Incorrect: {num_incorrect}")

    return num_correct / (num_correct + num_incorrect)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the initial suggestions with the refined suggestions.")
    parser.add_argument("--initial_suggestion_path", '-i', type=str, help="The path to the initial suggestion file.")
    parser.add_argument("--refined_suggestion_path", '-r', type=str, help="The path to the refined suggestion file.")
    parser.add_argument("--output_file", '-o', type=str, help="The path to the output file.")
    args = parser.parse_args()
    # Compare the initial suggestions with the refined suggestions
    compare_suggestion(args.initial_suggestion_path, args.refined_suggestion_path, args.output_file)
    # Compute the accuracy
    accuracy = compute_accuracy(args.output_file)
    print(f"Accuracy: {accuracy}")