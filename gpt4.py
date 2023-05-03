import argparse
import openai, time, json, os
import jsonlines, tqdm, copy, random
from typing import List, Dict, Any, Set
import multiprocessing as mp
from functools import wraps

# Set up OpenAI API credentials by reading from config/config.json
with open("config/config.json", "r") as f:
    config = json.load(f)
    openai.api_key = config["azure_api_key"]

openai.api_type = "azure"
openai.api_base = "https://llm-testing.openai.azure.com/"
openai.api_version = "2023-03-15-preview"

def limit_calls(max_calls=12, interval=60):
    """A decorator function that limits the number of function calls per time interval."""
    count = 0
    last_called = 0

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal count, last_called
            now = time.time()
            if now - last_called > interval:
                # reset count if the interval has passed
                count = 0
            if count >= max_calls:
                # wait until the interval has passed
                time.sleep(interval - (now - last_called))
            count += 1
            last_called = now
            result = fn(*args, **kwargs)
            return result
        return wrapper

    return decorator

@limit_calls(max_calls=12, interval=60)
def gpt4(text):
    """
    Generates a response to the given input text using the Chat model in OpenAI's GPT-3.5 API.

    Args:
        text (str): The input text to generate a response to.

    Returns:
        str: The generated response.

    Raises:
        Exception: If there is an error while making the API request.
    """

    # check tally counter whether we have reached the limit for 12 requests per minute
    

    try:
        response = openai.ChatCompletion.create(
            engine="llm-testing-gpt4", temperature=1, 
            messages=[
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": text},
                ]
            )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        if "Please retry after" in str(e):
            print("Waiting for 10 seconds...")
            time.sleep(10)
            return gpt4(text)
        return None


def process_datum(obj, output_file, lock):
    """
    Generates a response to the given input and writes the response to the given output file.
    """

    response = gpt4(obj["test_input"])
    if response is None:
        return

    # print(f"Generated response: {response}")    
    output_line = {"model": "gpt-4", **obj}
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
    random.shuffle(input_lines)
    input_lines = [{"test_input":l["text"], **l} for l in input_lines][:200]
    # Create a multiprocessing pool
    pool: mp.Pool = mp.Pool(processes=1)
    
    # Create a lock for writing to the output file
    manager = mp.Manager()
    lock = manager.Lock()

    # Use a partial function to pass the output_file and lock arguments to process_datum
    from functools import partial
    process_datum_with_output_file_and_lock = partial(process_datum, output_file=output_file, lock=lock)

    # Rewrite the data in parallel and append the output lines to the output file dynamically
    for _ in tqdm.tqdm(pool.imap(process_datum_with_output_file_and_lock, input_lines), total=len(input_lines)):
        pass

def generate_plausibility_data(input_file, output_file):

    with jsonlines.open(input_file) as f:
        data = list(f)
    
    with open("crowdsourced/plausibility_all.json") as f:
        references_json = json.load(f)
        reference_contexts = set([r['input_values']["context"] for r in references_json])

    json_data = []
    for datum in data:
        # print(datum)
        if datum["text"] not in reference_contexts:
            continue
        json_data.append(
            {
                "input_values": {
                "context": datum["text"].replace('\n', ' ') if "text" in datum else datum["test_input"].replace('\n', ' '),
                "suggestion": datum["test_output"].replace('\n', ' ')
                }
            }
        )
    print(len(json_data))
    with open(output_file, "w") as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Generate responses to input data using OpenAI GPT models')
    parser.add_argument('--input_file', '-i', type=str, required=True, help='path to input file')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='path to output file')

    # Parse the input arguments
    args = parser.parse_args()
    # Run the inference
    generate_plausibility_data(args.input_file, args.output_file)