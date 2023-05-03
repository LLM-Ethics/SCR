import jsonlines
from typing import Dict, List
import argparse, random

from reflection_checker import reflection_checker

STAGE_SUGGESTION = "suggestion"
STAGE_CRITIQUE = "critique"
STAGE_REFLECTION = "reflection"
STAGE_REFLECTION_EXPLAIN = "ref-exp"
STAGE_REFINE = "refine"
STAGE_CONSOLIDATE = "consolidate"

CRITIQUE_PROMPT_PATH = "prompts/critique.prompt"
REFLECTION_PROMPT_PATH = "prompts/reflection.prompt"
REFLECTION_EXPLAIN_PROMPT_PATH = "prompts/reflection-explain.prompt"
REFLECTION_REFINE_PROMPT_PATH = "prompts/refine.prompt"
CONSOLIDATE_PROMPT_PATH = "prompts/consolidate.prompt"

PRECEEDING_STAGE = {
    STAGE_SUGGESTION: None,
    STAGE_CRITIQUE: STAGE_SUGGESTION,
    STAGE_REFLECTION: STAGE_CRITIQUE,
    STAGE_REFLECTION_EXPLAIN: STAGE_CRITIQUE,
    STAGE_REFINE: STAGE_REFLECTION
}

def has_critique(datum: Dict[str, str]):
    assert "critique" in datum
    return "none" not in datum["critique"].lower()


def testinput_generation(datum: Dict[str, str], stage: str) -> Dict[str, str]:
    """
    Generates test input for a machine learning model based on the given input data and stage.

    Args:
        datum (Dict[str, str]): A dictionary containing the input data fields. This dictionary should contain the "text"
            field, and "suggestion" and "critique" fields depending on the stage parameter.
        stage (str): A string indicating the stage for which to generate a test input. This parameter should be one of
            "suggestion", "critique", or "reflection".

    Returns:
        Dict[str, str]: A dictionary containing the "test_input" field and any other fields from the "datum" parameter.

    Raises:
        ValueError: If the input data does not contain the required fields, or if the stage parameter is invalid.
        IOError: If there is an error while reading the prompt file.
    """

    if stage == STAGE_SUGGESTION:
        test_input = datum["text"] if "text" in datum else datum["test_input"]
    else:
        if "context" not in datum:
            raise ValueError("Missing 'context' field in input data")
        if "suggestion" not in datum and (stage == STAGE_CRITIQUE or stage == STAGE_REFLECTION or stage == STAGE_REFLECTION_EXPLAIN or stage == STAGE_REFINE):
            raise ValueError("Missing 'suggestion' field in input data")
        if "critique" not in datum and (stage == STAGE_REFLECTION  or stage == STAGE_REFLECTION_EXPLAIN or stage == STAGE_REFINE):
            raise ValueError("Missing 'critique' field in input data")
        if "reflection" not in datum and stage == STAGE_REFINE:
            raise ValueError("Missing 'reflection' field in input data")
        if "refine" not in datum and stage == STAGE_CONSOLIDATE:
            raise ValueError("Missing 'refine' field in input data")

        if stage == STAGE_CRITIQUE:
            prompt_path = CRITIQUE_PROMPT_PATH
            replacement_fields = {"$context$": datum["context"], "$suggestion$": datum["suggestion"]}
        elif stage == STAGE_REFLECTION:
            prompt_path = REFLECTION_PROMPT_PATH
            replacement_fields = {"$context$": datum["context"], "$suggestion$": datum["suggestion"], "$critique$": datum["critique"]}
        elif stage == STAGE_REFLECTION_EXPLAIN:
            prompt_path = REFLECTION_EXPLAIN_PROMPT_PATH
            replacement_fields = {"$context$": datum["context"], "$suggestion$": datum["suggestion"], "$critique$": datum["critique"]}
        elif stage == STAGE_REFINE:
            prompt_path = REFLECTION_REFINE_PROMPT_PATH
            replacement_fields = {"$context$": datum["context"], "$suggestion$": datum["suggestion"], "$critique$": datum["critique"]}
        elif stage == STAGE_CONSOLIDATE:
            prompt_path = CONSOLIDATE_PROMPT_PATH
            initial_suggestion_first = random.random() < 0.5
            if initial_suggestion_first:
                replacement_fields = {"$context$": datum["context"], "$suggestion1$": datum["suggestion"], "$suggestion2$": datum["refine"]}
            else:
                replacement_fields = {"$context$": datum["context"], "$suggestion1$": datum["refine"], "$suggestion2$": datum["suggestion"]}
            datum["initial_suggestion_first"] = initial_suggestion_first
        else:
            raise ValueError("Invalid stage value")

        try:
            with open(prompt_path, "r") as f:
                prompt = f.read()
        except IOError:
            raise IOError(f"Could not read {prompt_path}")

        for field, value in replacement_fields.items():
            prompt = prompt.replace(field, value)

        if False: # (stage == STAGE_REFLECTION or stage == STAGE_REFLECTION_EXPLAIN) and not has_critique(datum):
            test_input = ""
        elif stage == STAGE_REFINE and not reflection_checker({"test_output": datum["reflection"]}):
            test_input = ""
        elif stage == STAGE_CONSOLIDATE and datum["refine"].strip() == "":
            test_input = ""
        else:
            test_input = prompt

    return {"test_input": test_input, **datum}

def generate_test_inputs_from_jsonl(source_path: str, input_path: str, output_path: str, stage: str, is_merged: bool = False):
    # Open the source file in read mode.
    with jsonlines.open(source_path, "r") as source_reader:
        data1 = list(source_reader)
        # Open the input file in read mode.
        with jsonlines.open(input_path, "r") as reader:
            data2 = list(reader)
            if STAGE_CONSOLIDATE == stage:
                data2 = data2[:1000]
            # Open the output file in write mode.
            with jsonlines.open(output_path, "w") as writer:
                # For each line in the input file, process the data and write it to the output file.
                for idx, datum in enumerate(data2):
                    # Process the data and write it to the output file.
                    # this is a merged jsonl file
                    # if is_merged and PRECEEDING_STAGE[stage] in datum:
                    #     data2[idx][PRECEEDING_STAGE[stage]] = datum[PRECEEDING_STAGE[stage]]
                    #     if stage == STAGE_REFLECTION or stage == STAGE_REFLECTION_EXPLAIN:
                    #         data2[idx]["suggestion"] = datum["suggestion"]
                    # else: data1[idx][PRECEEDING_STAGE[stage]] = datum["test_output"]
                    # if is_merged:
                    #     data2[idx][PRECEEDING_STAGE[stage]] = datum["test_output"]
                    #     if stage == STAGE_REFLECTION or stage == STAGE_REFLECTION_EXPLAIN:
                    #         data2[idx]["suggestion"] = datum["suggestion"]
                    new_datum = {
                        "context": datum["test_input"] if  (stage == STAGE_CRITIQUE and "test_input" in datum) else datum["context"] if "context" in datum else datum["text"],
                        "suggestion": datum["test_output"] if stage == STAGE_CRITIQUE else datum["suggestion"],
                        "critique": "" if stage == STAGE_CRITIQUE else datum["test_output"] if datum["critique"] == "" else datum["critique"],
                        "reflection": datum["test_output"] if stage == STAGE_REFINE else "",
                        "refine": datum["test_output"] if stage == STAGE_CONSOLIDATE else "",
                    }
                    result = testinput_generation(new_datum, stage)
                    if result is not None and result["test_input"] != "": writer.write(result)

if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", "-src", default="ethics/processed/rewritten_test.jsonl", type=str, help="Path to the source dataset JSONL file")
    parser.add_argument("--input_path", "-i", type=str, help="Path to the input JSONL file")
    parser.add_argument("--output_path", "-o", type=str, help="Path to the output JSONL file")
    parser.add_argument("--merged_path", "-m", type=str, help="Path to the merged JSONL file")
    parser.add_argument("--stage", "-s", type=str, required=True, help="Stage for which to generate test inputs", choices=[STAGE_SUGGESTION, STAGE_CRITIQUE, STAGE_REFLECTION, STAGE_REFLECTION_EXPLAIN, STAGE_REFINE, STAGE_CONSOLIDATE])
    args = parser.parse_args()
    print(args)
    # Generate test inputs.
    generate_test_inputs_from_jsonl(args.source_path, args.merged_path if args.merged_path else args.input_path, args.output_path, args.stage, is_merged=args.merged_path is not None)