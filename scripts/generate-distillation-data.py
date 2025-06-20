import json
import os
import time
from functools import partial

import openai
from datasets import load_dataset
from fire import Fire
from loguru import logger
from tqdm import tqdm
from transformers import set_seed

from doge.data import batch_preprocess_fn

set_seed(233)


def append_generation(response, prompt, output_file):
    entry = {
        "response": response,
        "prompt": prompt,
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def api_generate_distillation_data_eager(
        dataset_name: str = "ServiceNow-AI/R1-Distill-SFT",
        base_url: str = "http://localhost:2334/v1",
        save_dir: str = "data/phimoe/",
        model_name: str = "microsoft/Phi-3.5-MoE-instruct",
        num_workers: int = 4,
        max_tokens: int = 8192,
        shuffle: bool = True,
        split_id: int = 0,
        num_splits: int = None,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Created directory {save_dir}")

    client = openai.Client(base_url=base_url, api_key="EMPTY")
    is_gsm_8k = "gsm8k" in dataset_name.lower()
    is_reasoning = "r1" in dataset_name.lower()

    if is_gsm_8k:
        dataset = load_dataset(
            dataset_name, "main", trust_remote_code=True
        )
    else:
        dataset = load_dataset(
            dataset_name, "v1", trust_remote_code=True
        )
    dataset = dataset["train"]

    # remove those samples with "source" is "ai2-adapt-dev/tulu_hard_coded_repeated_10"
    if not is_gsm_8k:
        dataset = dataset.filter(lambda example: example["source"] != "ai2-adapt-dev/tulu_hard_coded_repeated_10")
    if shuffle:
        dataset = dataset.shuffle(seed=42)
        if num_splits is not None:
            raise ValueError("Shuffle and split cannot be applied at the same time")
    
    if is_gsm_8k:
        preprocess_fn = partial(batch_preprocess_fn, task="chat-gen-gsm8k")
    else:
        preprocess_fn = partial(batch_preprocess_fn, task="chat-gen")
    dataset = dataset.map(preprocess_fn, batched=True, num_proc=num_workers, remove_columns=dataset.column_names)
    
    if num_splits is not None:
        # selecte the no. split_id of the dataset
        num_samples_per_split = len(dataset) // num_splits
        logger.info(f"Selecting {num_samples_per_split} samples for split {split_id} of {num_splits}")
        dataset = dataset.select(range(split_id * num_samples_per_split, (split_id + 1) * num_samples_per_split))
    
    if "qwen3" in model_name.lower():
        kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "extra_body": {
                "top_k": 20,
            },
        }
    else:
        kwargs = {}
    
    if num_splits is not None:
        save_file_name = f"distillation_data_split-{split_id}-of-{num_splits}.jsonl"
    else:
        save_file_name = "distillation_data.jsonl"

    with open(os.path.join(save_dir, save_file_name), 'a') as file:
        for j, messages in enumerate(tqdm(dataset["content"], desc="Generating distillation data via API")):
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs
            )
            reasoning_response = completion.choices[0].message.reasoning_content
            response = completion.choices[0].message.content
            prompt = messages[-1]["content"]
            result_content = json.dumps({
                "reasoning_response": reasoning_response,
                "response": response,
                "prompt": prompt,
            }, ensure_ascii=False) + "\n"
            file.write(result_content)


if __name__ == "__main__":
    Fire(api_generate_distillation_data_eager)
