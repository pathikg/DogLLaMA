import os
import json
import random
import argparse

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


def generate_response(client, model, system_prompt, generation_prompt, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": generation_prompt}],
        temperature=temperature,
    )
    json_string = response.choices[0].message.content
    tokens = response.usage.total_tokens
    
    return tokens, json.loads(json_string)

def load_prompts(filename):
    with open(filename, "r") as f:
        return f.read()

def generate_samples_chunk(chunk_args, pbar):
    client, model, system_prompt, generation_prompt, temperature, chunk_size = chunk_args
    chunk_samples = []
    
    if temperature == -1: # dawg goes wild
        temperature = random.uniform(0, 0.9)

    tokens, response = generate_response(client, model, system_prompt, generation_prompt, temperature)
    response.append(
        {
            "tokensUsed": tokens
        }
    )
    chunk_samples.extend(response)
    pbar.update(chunk_size)

    return chunk_samples

def main(args):
    model = args.model
    temperature = args.temperature
    
    api_key = os.getenv("OPENAI_API_KEY", None)
    client = OpenAI(api_key=api_key)
    
    system_prompt = load_prompts(args.system_prompt_file)
    generation_prompt = load_prompts(args.generation_prompt_file)
    number_of_samples = args.num_samples
    tokens_used = 0
    samples = []

    # Specify the chunk size for each parallel iteration
    chunk_size = 30
    num_chunks = number_of_samples // chunk_size
    with tqdm(total=number_of_samples, desc="Generating Samples") as pbar:
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            # Use list comprehension to create a list of arguments for each parallel iteration
            chunk_args_list = [(client, model, system_prompt, generation_prompt, temperature, chunk_size) for _ in range(num_chunks)]

            # Map the generate_samples_chunk function to execute in parallel
            result_samples = list(executor.map(lambda args: generate_samples_chunk(args, pbar), chunk_args_list))

            # Flatten the list of lists into a single list
            samples = [sample for chunk_samples in result_samples for sample in chunk_samples]

    tokens_used = sum(sample.get("tokensUsed", 0) for sample in samples)

    samples.append({
        "tokensUsed": tokens_used
    })

    with open(args.output_path, "w") as f:
        json.dump(samples, f)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, help="GPT model to use", default="gpt-3.5-turbo")
    parser.add_argument("--system_prompt_file", type=str, help="Path to the system prompt file", default="./data/dataset_instructions.txt")
    parser.add_argument("--generation_prompt_file", type=str, help="Path to the dataset generation prompt file", default="./data/dataset_generation_instructions.txt")
    parser.add_argument("--output_path", type=str, help="Path to the save the dataset in JSON format", default="./data/dataset.json")
    parser.add_argument("--temperature", type=float, help="Parameter to model creativity, -1 for random temperature in each interation", default=0.5)
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate", default=1000)
    
    args = parser.parse_args()
    main(args)
    
