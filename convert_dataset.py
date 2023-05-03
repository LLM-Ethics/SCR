import openai
import pandas as pd
import os, time
import jsonlines
import tqdm, json
from typing import List, Dict, Any
import multiprocessing as mp

# Set up OpenAI API credentials by reading from config/config.json

def change_backend(backend: str="azure"):

    if backend == "azure":
        import openai
        openai.api_type = "azure"
        openai.api_base = "https://llm-testing.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = "<API_KEY>"
    else:
        import openai
        with open("config/config.json", "r") as f:
            config = json.load(f)
            openai.api_key = config["openai_api_key"]

change_backend("")

def statement_rewriter(statement: str, model: str="chatgpt") -> str:
    # open situation2context.prompt file and read the prompt
    
    try:
        with open("prompts/situation2context.prompt", "r") as f:
            prompt: str = f.read()
    except:
        raise Exception("Error: could not read the prompt file.")

    # Use GPT-3 to rewrite the statement
    if model == "gpt3":
        try:
            response: str = openai.Completion.create(engine="text-davinci-003", temperature=1.0, prompt=prompt + "\n" + statement + "\n", max_tokens=1024)["choices"][0]["text"]
        except:
            raise Exception("Error: could not get response from OpenAI API.")

    # Use GPT-3.5 Turbo to rewrite the statement
    if model == "chatgpt":
        retry = 0
        while retry < 10:
            try:
                if openai.api_type == "azure":
                    response = openai.ChatCompletion.create(
                      engine="llm-testing",model="gpt-3.5-turbo", temperature=1, 
                      messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": statement},
                        ]
                    )["choices"][0]["message"]["content"]
                else:
                    response: str = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", temperature=1, 
                        messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": statement},
                            ]
                        )["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if "The response was filtered due to the prompt" in str(e):
                    change_backend("openai")
                    resp = statement_rewriter(statement, model)
                    print("Resort to openai backend", repr(resp))
                    change_backend("azure")
                    return resp
                time.sleep(5)
                # raise Exception("Error: could not get response from OpenAI API.")
                retry += 1
    if retry == 10:
        print("The response was failed after ten attempts", repr(statement))
        return None

    return response

def read_dataset(csv_file: str) -> List[Dict[str, str]]:
    # use file name without extension as source
    source = os.path.splitext(os.path.basename(csv_file))[0]
    df = pd.read_csv(csv_file)
    
    # print csv file name, column names, and shape
    print(csv_file, df.columns, df.shape)
    
    # if input column exists, rename as text
    if "input" in df.columns:
        df["text"] = df["input"]
    # if scenario and excuse columns exist, concatenate and rename as text
    elif "scenario" in df.columns and "excuse" in df.columns:
        df["text"] = df["scenario"] + "," + df["excuse"]
    # if only scenario column exists, rename as text
    elif "scenario" in df.columns and "excuse" not in df.columns:
        df["text"] = df["scenario"]
    # if no supported columns exist, return empty list
    else:
        print("Dataset not supported")
        return []
    
    # add source column
    df["source"] = source
    
    # only keep text and source columns
    df = df[["text", "source"]]
    
    # return list of dictionaries, where each dictionary represents a single JSON object
    return df.to_dict("records")

def convert2jsonl():
    # get list of all source folders
    dataset_source_folders: List[str] = ["commonsense", "deontology", "justice", "virtue", "utilitarianism"]
    
    # create empty lists to hold dataframes
    train_df: List[Dict[str, str]] = []
    test_df: List[Dict[str, str]] = []
    test_hard_df: List[Dict[str, str]] = []
    
    # for each source folder
    for source_folder in dataset_source_folders:
        # for each file in folder
        for csv_file in os.listdir(f"ethics/{source_folder}"):
            # if file is train
            if "train" in csv_file: 
                # read data and add to train dataframe
                train_df += read_dataset(f"ethics/{source_folder}/{csv_file}")
            # if file is test
            elif "test" in csv_file and "hard" not in csv_file:
                # read data and add to test dataframe
                test_df += read_dataset(f"ethics/{source_folder}/{csv_file}")
            # if file is test hard
            elif "test" in csv_file and "hard" in csv_file:
                # read data and add to test hard dataframe
                test_hard_df += read_dataset(f"ethics/{source_folder}/{csv_file}") 
    
    # write train, test and test_hard datasets to jsonl files
    # print number of records in each dataset
    print("Train dataset size:", len(train_df))
    print("Test dataset size:", len(test_df))
    print("Test hard dataset size:", len(test_hard_df))
    with jsonlines.open("ethics/processed/train.jsonl", "w") as writer:
        writer.write_all(train_df)
    with jsonlines.open("ethics/processed/test.jsonl", "w") as writer:
        writer.write_all(test_df)
    with jsonlines.open("ethics/processed/test_hard.jsonl", "w") as writer:
        writer.write_all(test_hard_df)
    print("Done writing train, test and test_hard datasets to jsonl files")

def process_datum(datum: dict, output_file, lock) -> dict:
    datum["original_text"] = datum["text"]
    datum["text"] = statement_rewriter(datum["text"])

    if datum["text"] is None:
        return
        
    # Append the output line to the output file with the lock
    with lock:
        with jsonlines.open(output_file, mode='a') as writer:
            writer.write(datum)

def rewrite_dataset(jsonl_file: str, rewrite_num: int, continue_rewrite: bool=True):
    """
    This function takes in a jsonl file and a number of
    data points to rewrite. It opens the file, reads in
    the data, rewrites the data, and writes the rewritten
    data to a new file.
    """
    # Open the jsonl file
    with jsonlines.open(jsonl_file, "r") as reader:
        # Read the data into a list
        data: List[Dict[str, Any]] = list(reader)
    
    basename = os.path.basename(jsonl_file)
    rewrite_file_path = f"ethics/processed/rewritten_{basename}"

    # If we are continuing a rewrite, we need to read in the
    # existing rewritten data and remove it from the data
    # that we will rewrite.
    print("Continuing rewrite:", continue_rewrite)
    print("Rewrite file path:", rewrite_file_path, "exists:", os.path.exists(rewrite_file_path))
    if continue_rewrite and os.path.exists(rewrite_file_path):
        # Open the existing rewritten jsonl file
        with jsonlines.open(rewrite_file_path, "r") as reader:
            # Read the data into a list
            rewritten_data: List[Dict[str, Any]] = list(reader)
        completed_data = set([datum["original_text"] for datum in rewritten_data])
        data = [datum for datum in data if datum["text"] not in completed_data]
        print(f"Continuing rewrite. A total {len(data)} data points to rewrite.")
    # Create a multiprocessing pool
    pool: mp.Pool = mp.Pool(processes=15)
    # Rewrite the data in parallel
    if rewrite_num != -1:
        data = data[:rewrite_num]
    # Create a Manager to share the lock between processes
    manager = mp.Manager()
    lock = manager.Lock()

    # Use a partial function to pass the output_file and lock arguments to process_datum
    from functools import partial
    process_datum_with_output_file_and_lock = partial(process_datum, output_file=rewrite_file_path, lock=lock)

    # Rewrite the data in parallel and append the output lines to the output file dynamically
    for _ in tqdm.tqdm(pool.imap(process_datum_with_output_file_and_lock, data), total=len(data)):
        pass

if __name__ == "__main__":
    # main()
    # print(statement_rewriter("I told my baby I hated her when she cried."))
    rewrite_dataset("ethics/processed/train-mini.jsonl", -1, True)
    # df = pd.read_csv("data/processed/processed_dataset.csv")
    # df["context"] = df["statement"].apply(statement_rewriter)
    # df.to_csv("data/processed/processed_dataset.csv", index=False)