# DogLLaMA: Dog Language Model and Dataset

## Overview
Welcome to DogLLaMA, a fun project which presents a language model that translates user queries into the language of dogs. Whether you want to know a dog's perspective on jokes, treats, or the weather, DogLLaMA has got you covered! This repository also includes scripts to create the dataset, fine-tune the model, and push the dataset to the Hugging Face Model Hub.

## Project Structure
- data: Directory containing the DogLLaMA dataset in JSON format and few prompts.
- scripts: Collection of utility scripts for dataset creation, Hugging Face compatibility, and model deployment.
- DogLLaMA_2_Fine_Tuning_Lora.ipynb: Jupyter notebook containing the model fine-tuning code.
- README.md: This readme file.

## Usage

### 1. create_dataset.py  
Generate a datasets of Human-DogGPT chat pair using OpenAI GPT model for fine-tuning the Dog Language Model.

`python .\scripts\create_dataset.py -h`

Options:

- --model MODEL: GPT model to use.
- --system_prompt_file SYSTEM_PROMPT_FILE: Path to the system prompt file.
- --generation_prompt_file GENERATION_PROMPT_FILE: Path to the dataset generation prompt file.
- --output_path OUTPUT_PATH: Path to save the dataset in JSON format.
- --temperature TEMPERATURE: Parameter to model creativity, -1 for random temperature in each iteration.
- --num_samples NUM_SAMPLES: Number of samples to generate.

### 2. create_hf_dataset.py
Convert the JSON dataset to Llama 2 compatible prompt format and create a train-test split.

Usage:

`python .\scripts\create_hf_dataset.py -h`

Options:

- --json_file JSON_FILE: Path to the input JSON file.
- --output_dir OUTPUT_DIR: Directory to save the Hugging Face dataset.

### 3. push_to_hub.py
Push the DogLLaMA dataset to the Hugging Face Model Hub.

Usage:

`python .\scripts\push_to_hub.py -h`

Options:

--dataset_name DATASET_NAME: Name for the dataset on the Hugging Face Model Hub.
--dataset_path DATASET_PATH: Path of the dataset stored locally.

### 4. Fine-Tuning Notebook
Explore the DogLLaMA_2_Fine_Tuning_Lora.ipynb notebook for detailed code and instructions on fine-tuning the Dog Language Model. referred from [here](https://deci.ai/blog/fine-tune-llama-2-with-lora-for-question-answering/)

## Next Steps:

- Inference Code
- HuggingFace Space 
 

## Contributing
Feel free to contribute to the project by creating issues or pull requests. We welcome your ideas and improvements!

Happy translating to Dog Language with DogLLaMA! 🐾