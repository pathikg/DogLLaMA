import os
from datasets import load_from_disk
from dotenv import load_dotenv
from argparse import ArgumentParser

load_dotenv()

def push_to_huggingface(args):
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path

    # Load the dataset
    dogllama_dataset = load_from_disk(dataset_path)

    # Set up Hugging Face credentials
    huggingface_username = os.getenv("HF_USERNAME")
    huggingface_api_key = os.getenv("HF_API_KEY")

    if not huggingface_username or not huggingface_api_key:
        raise ValueError("Hugging Face credentials (HF_USERNAME and HF_API_KEY) are missing.")


    # Set the dataset name on the Hugging Face Model Hub
    full_dataset_name = f"{huggingface_username}/{dataset_name}"

    # Push the dataset to the Hugging Face Model Hub
    dogllama_dataset.push_to_hub(repo_id=full_dataset_name, token=huggingface_api_key)

    # Output the Hugging Face dataset ID
    print(f"DogLLaMA dataset uploaded to Hugging Face Model Hub with name: {full_dataset_name}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Push DogLLaMA dataset to Hugging Face Model Hub")
    parser.add_argument("--dataset_name", type=str, help="Name for the dataset on the Hugging Face Model Hub", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path of dataset stored locally", required=True)
    args = parser.parse_args()
    
    push_to_huggingface(args)
