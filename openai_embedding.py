import argparse
import openai, time, json, os
import jsonlines, tqdm, copy
from typing import List, Dict, Any, Set
import multiprocessing as mp
import numpy as np



# Define constants for retrying requests
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2

def get_embedding(text, model="text-embedding-ada-002"):
    import openai
    # Set up OpenAI API credentials by reading from config/config.json
    with open("config/config.json", "r") as f:
        config = json.load(f)
        openai.api_key = config["openai_api_key"]
    for retry in range(MAX_RETRIES):
            try:
                response = openai.Embedding.create(input = [text], model=model)
                return response['data'][0]['embedding']
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    print(f"Error making Chat request: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    raise e

def get_cosine_distance(initial_suggestion: str, refined_suggestion: str):
    # Get embeddings for the initial and refined suggestion
    initial_embed = get_embedding(initial_suggestion)
    refined_embed = get_embedding(refined_suggestion)
    
    # Calculate cosine distance between embeddings using numpy
    cosine_distance = np.dot(initial_embed, refined_embed)/(np.linalg.norm(initial_embed)*np.linalg.norm(refined_embed))
    
    return 1 - cosine_distance