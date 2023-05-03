import jsonlines
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List
import argparse
import dill
from tqdm import tqdm

SENTENCE_KEYS = {"test_input", "test_output", "suggestion", "context"}
BATCH_SIZE = 32

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings(sentences: List[str]) -> List[List[float]]:
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    all_embeddings = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
        current_batch = sentences[i:i+BATCH_SIZE]
        # Tokenize sentences
        encoded_input = tokenizer(current_batch, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Add to result
        all_embeddings.extend(sentence_embeddings.tolist())

    return all_embeddings

def main(input_file: str, output_file: str):
    # Load data
    with jsonlines.open(input_file) as reader:
        data = list(reader)
    # exampple data

    sentences = []
    
    for datum in data:
        for key in SENTENCE_KEYS:
            if key in datum:
                sentences.append(datum[key])

    # Generate embeddings
    embeddings = generate_embeddings(sentences)

    for datum in data:
        for key in SENTENCE_KEYS:
            if key in datum:
                datum[f"{key}_embedding"] = embeddings.pop(0)

    # Write embeddings to file
    with open(output_file, "wb") as f:
        dill.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    main(args.input_file, args.output_file)