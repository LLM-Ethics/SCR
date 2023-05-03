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

def gpt3(text):
    """
    Generates text completion for the given prompt using OpenAI's GPT-3 API.

    Args:
        text (str): The prompt to generate text completion for.

    Returns:
        str: The generated text completion.

    Raises:
        Exception: If there is an error while making the API request.
    """
    for retry in range(MAX_RETRIES):
        try:
            response = openai.Completion.create(engine="text-davinci-003", temperature=1.0, prompt=text, max_tokens=1024)
            return response["choices"][0]["text"]
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                print(f"Error making GPT-3 request: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise e

def process_datum(obj, output_file, lock):
    if MODEL == "chatgpt":
        response = chatgpt(obj["test_input"])
    elif MODEL == "gpt3":
        response = gpt3(obj["test_input"])
    else:
        raise Exception(f"Error: model {MODEL} is not supported.")

    # print(f"Generated response: {response}")    
    output_line = {"model": MODEL, **obj}
    output_line["test_output"] = response
    
    # Append the output line to the output file with the lock
    with lock:
        with jsonlines.open(output_file, mode='a') as writer:
            writer.write(output_line)

def critique(input_file: str, output_file: str, continue_critique: bool = True):
    """
    Generates a response to each input in the given input file using the specified model and writes the responses to the given output file.

    Args:
        input_file (str): The path to the input file.
        output_file (str): The path to the output file.
        continue_critique (bool, optional): Whether to continue the critique if the output file already exists. Defaults to True.
    Raises:
        Exception: If the input file does not exist.
    """
    # Check that the input file exists
    if not os.path.exists(input_file):
        raise Exception(f"Error: input file {input_file} does not exist.")

    # If the output file exists and the user wants to continue the critique, only use lines that haven't been processed yet
    if continue_critique and os.path.exists(output_file):
        with jsonlines.open(output_file) as reader:
            # Get the set of processed inputs
            output_lines: List[Dict[str, Any]] = list(reader)
        completed_text: Set[str] = set([o["test_input"] for o in output_lines])
        with jsonlines.open(input_file) as reader:
            # Get the set of inputs that need to be processed
            input_lines: List[Dict[str, Any]] = list(reader)
        input_lines = [l for l in input_lines if l["test_input"] not in completed_text]
        print(f"Continuing critique from {len(output_lines)} processed lines to {len(input_lines)} unprocessed lines.")
    else:
        # Otherwise, just use the first 10 lines of the input file
        with jsonlines.open(input_file) as reader:
            input_lines: List[Dict[str, Any]] = list(reader)
    
    # Create a multiprocessing pool
    pool: mp.Pool = mp.Pool(processes=15)
    
    # Create a lock for writing to the output file
    manager = mp.Manager()
    lock = manager.Lock()

    # Use a partial function to pass the output_file and lock arguments to process_datum
    from functools import partial
    process_datum_with_output_file_and_lock = partial(process_datum, output_file=output_file, lock=lock)

    # Rewrite the data in parallel and append the output lines to the output file dynamically
    for _ in tqdm.tqdm(pool.imap_unordered(process_datum_with_output_file_and_lock, input_lines), total=len(input_lines)):
        pass

def suggestion(input_file: str, output_file: str, continue_critique: bool = True):
    """
    Generates a response to each input in the given input file using the specified model and writes the responses to the given output file.

    Args:
        input_file (str): The path to the input file.
        output_file (str): The path to the output file.
        continue_critique (bool, optional): Whether to continue the critique if the output file already exists. Defaults to True.
    Raises:
        Exception: If the input file does not exist.
    """
    # Check that the input file exists
    if not os.path.exists(input_file):
        raise Exception(f"Error: input file {input_file} does not exist.")

    # If the output file exists and the user wants to continue the critique, only use lines that haven't been processed yet
    if continue_critique and os.path.exists(output_file):
        with jsonlines.open(output_file) as reader:
            # Get the set of processed inputs
            output_lines: List[Dict[str, Any]] = list(reader)
        completed_text: Set[str] = set([o["text"] for o in output_lines])
        with jsonlines.open(input_file) as reader:
            # Get the set of inputs that need to be processed
            input_lines: List[Dict[str, Any]] = list(reader)
        input_lines = [l for l in input_lines if l["text"] not in completed_text]
        print(f"Continuing critique from {len(output_lines)} processed lines to {len(input_lines)} unprocessed lines.")
    else:
        # Otherwise, just use the first 10 lines of the input file
        with jsonlines.open(input_file) as reader:
            input_lines: List[Dict[str, Any]] = list(reader)
    input_lines = [{"test_input":l["text"], **l} for l in input_lines]
    # Create a multiprocessing pool
    pool: mp.Pool = mp.Pool(processes=15)
    
    # Create a lock for writing to the output file
    manager = mp.Manager()
    lock = manager.Lock()

    # Use a partial function to pass the output_file and lock arguments to process_datum
    from functools import partial
    process_datum_with_output_file_and_lock = partial(process_datum, output_file=output_file, lock=lock)

    # Rewrite the data in parallel and append the output lines to the output file dynamically
    for _ in tqdm.tqdm(pool.imap_unordered(process_datum_with_output_file_and_lock, input_lines), total=len(input_lines)):
        pass

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Generate responses to input data using OpenAI GPT models')
    parser.add_argument('--input_file', '-i', type=str, required=True, help='path to input file')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='path to output file')
    parser.add_argument('--model', '-m', type=str, choices=["chatgpt", "gpt3"], default="chatgpt", help='model to use (default: chatgpt)')

    # Parse the input arguments
    args = parser.parse_args()
    MODEL = args.model
    # Run the inference
    suggestion(args.input_file, args.output_file)