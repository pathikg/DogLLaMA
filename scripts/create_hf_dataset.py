import argparse
import json
from datasets import Dataset

def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def convert_to_llama2_prompt(conversation):
    llama2_template = "<s>[INST] {user_msg} [/INST] {model_answer} </s>\n"
    return llama2_template.format(user_msg=conversation["Human"], model_answer=conversation["DogGPT"])

def main(args):
    # Load JSON data
    json_data = load_json(args.json_file)

    # Convert JSON data to Llama 2 compatible prompt format
    llama2_prompts = [convert_to_llama2_prompt(conv) for conv in json_data if conv.get("Human", "")]

    # Create Hugging Face dataset
    dataset = Dataset.from_dict({"text": llama2_prompts})

    # Split dataset into train and test
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Save dataset
    dataset.save_to_disk(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON data to Llama 2 compatible prompt format and create a train-test split")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the Hugging Face dataset")

    args = parser.parse_args()
    main(args)
