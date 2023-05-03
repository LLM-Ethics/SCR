import json, jsonlines, os, sys, random, argparse

def convert_refinement_to_finetune_data(refinement_path, training_data_path, validation_data_path):
    all_data = {
        "type": "text_only",
        "instances": [
        ]
        }
    # example critique:
    # {"critique_model": "chatgpt", "test_input": "You are a critic for criticize the suggestion with your best efforts to make the suggestion more ethical, responsible and reasonable. Give a list of critique if possible. If you believe the suggestion is already ethical, responsible and reasonable, output <None> and stop output immediately.\n\nContext: I have a friend who is currently enrolled in a basic level class in college. He told me that he did not feel like taking a more advanced class because it would require more work and effort. He simply couldn't be bothered to put in any extra effort. However, I wonder if his lack of ambition is ethical. Should he settle for an easier class when he is capable of challenging himself with a more advanced one?\n\nSuggestion:  It is important to challenge yourself and push your limits. Taking a more advanced class can be beneficial in the long run, as it can help you develop new skills and knowledge. It is important to have ambition and strive for success, even if it requires extra effort. However, it is also important to be realistic and not overwhelm yourself with too much work. \n\nCritique to Suggestion: ", "context": "I have a friend who is currently enrolled in a basic level class in college. He told me that he did not feel like taking a more advanced class because it would require more work and effort. He simply couldn't be bothered to put in any extra effort. However, I wonder if his lack of ambition is ethical. Should he settle for an easier class when he is capable of challenging himself with a more advanced one?", "suggestion": " It is important to challenge yourself and push your limits. Taking a more advanced class can be beneficial in the long run, as it can help you develop new skills and knowledge. It is important to have ambition and strive for success, even if it requires extra effort. However, it is also important to be realistic and not overwhelm yourself with too much work. ", "critique": "<None>"}
    with jsonlines.open(refinement_path) as reader:
        for refinement in reader:
            instance = {
                "text": f"""Input: User: {refinement["context"]}\n Assistant: {refinement["test_output"]}""",
            }
            all_data["instances"].append(instance)
    random.shuffle(all_data["instances"])
    train_data = {
        "type": "text_only",
        "instances": all_data["instances"][:int(len(all_data["instances"])*0.8)]
    }
    validation_data = {
        "type": "text_only",
        "instances": all_data["instances"][int(len(all_data["instances"])*0.8):]
    }
    with open(training_data_path, "w") as f:
        json.dump(train_data, f)
    with open(validation_data_path, "w") as f:
        json.dump(validation_data, f)
    
def convert_val_data(validation_data_path, original_suggestion_data_path, output_path):
    with open(validation_data_path) as reader:
        validation_data = json.load(reader)["instances"]
    with jsonlines.open(original_suggestion_data_path) as reader:
        original_suggestion_data = list(reader)
    val_suggestion = []
    for datum in original_suggestion_data:
        for val_datum in validation_data:
            if datum["text"] in val_datum["text"]:
                val_suggestion.append(datum)
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(val_suggestion)

def convert_jsonl_to_csv(input_path, output_path):
    import pandas as pd
    df = pd.read_json(input_path, lines=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_refine_parser = subparsers.add_parser("convert-refinement")
    convert_refine_parser.add_argument('refinement_path', type=str, help='path to input refinement data')
    convert_refine_parser.add_argument('training_data_path', type=str, help='path to output training fine-tuning data')
    convert_refine_parser.add_argument('validation_data_path', type=str, help='path to output validation fine-tuning data')

    convert_val_parser = subparsers.add_parser("convert-val")
    convert_val_parser.add_argument('validation_data_path', type=str, help='path to input validation data')
    convert_val_parser.add_argument('original_suggestion_data_path', type=str, help='path to input original suggestion data')
    convert_val_parser.add_argument('output_path', type=str, help='path to output validation suggestion data')

    convert_csv_parser = subparsers.add_parser("convert-csv")
    convert_csv_parser.add_argument('input_path', type=str, help='path to input jsonl data')
    convert_csv_parser.add_argument('output_path', type=str, help='path to output csv data')


    args = parser.parse_args()

    if args.command == "convert-val":
        convert_val_data(args.validation_data_path, args.original_suggestion_data_path, args.output_path)
    elif args.command == "convert-refinement":
        convert_refinement_to_finetune_data(args.refinement_path, args.training_data_path, args.validation_data_path)
    elif args.command == "convert-csv":
        convert_jsonl_to_csv(args.input_path, args.output_path)
