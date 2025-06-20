import json
from functools import partial
from loguru import logger
from tqdm import tqdm
from fire import Fire
import os

from vllm import LLM
from vllm.sampling_params import SamplingParams
from doge.evaluation import evaluate_predictions
from doge.data import batch_preprocess_fn
from transformers import AutoTokenizer
from datasets import load_dataset


def main(
    model_path: str,
    task_name: str="gsm8k",
    batch_size: int=4,
    num_workers: int=1,
    num_gpus: int=1,
    max_tokens: int=8192,
    debugging: bool=False,
):    
    if "llama-3.2" in model_path.lower():
        tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif "r1-distill-qwen-7b" in model_path.lower() or "qwen7b-antidistill" in model_path.lower():
        tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    else:
        raise ValueError(f"Model {model_path} not supported")
    
    if task_name == "gsm8k":
        datasets = load_dataset("openai/gsm8k", "main", trust_remote_code=True)["test"]
        task_type = "math"
    elif task_name == "math":
        datasets = load_dataset("ANONYMOUS/math", trust_remote_code=True)["test"]
        task_type = "math"
    elif task_name == "tabmwp": 
        datasets = load_dataset("ANONYMOUS/tabmwp", trust_remote_code=True)["test"]
        task_type = "table"
    elif task_name == "csqa":
        datasets = load_dataset("tau/commonsense_qa", trust_remote_code=True)["validation"]
        task_type = "csqa"
    elif task_name == "arcc":
        datasets = load_dataset("allenai/ai2_arc", "ARC-Challenge", trust_remote_code=True)["test"]
        task_type = "arcc"
    else:
        raise ValueError(f"Task name {task_name} not supported")
    
    preprocess_fn = partial(
        batch_preprocess_fn, task="math-reasoning-llama-3.2-eval", tokenizer=tokenizer, task_type=task_type
    )
    
    # no cache
    datasets = datasets.map(
        preprocess_fn, 
        num_proc=num_workers, 
        batched=True
    )

    if debugging:
        datasets = datasets.select(range(4))
    
    sampling_params = SamplingParams(max_tokens=max_tokens, stop=[tokenizer.pad_token])
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        max_model_len=max_tokens,
        tensor_parallel_size=num_gpus,
    )
    
    predictions = []
    ground_truths = []
    prompt_list = []
    
    logger.info(f"Generating predictions for {len(datasets)} examples with batch size {batch_size}!")
    for batch_id in tqdm(range(len(datasets) // batch_size)):
        batch_datasets = datasets[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_prompt = [prompt.split(tokenizer.bos_token)[-1] for prompt in batch_datasets['prompt']]
        batch_answer = batch_datasets['response']
        
        outputs = llm.generate(batch_prompt, sampling_params=sampling_params)
        batch_prediction = [output.outputs[0].text for output in outputs]
        
        predictions.extend(batch_prediction)
        ground_truths.extend(batch_answer)
        prompt_list.extend(batch_prompt)
        
        for pred, gt in zip(batch_prediction, batch_answer):
            logger.debug(f"Input: {batch_prompt}")
            logger.debug(f"Prediction: {pred}")
            logger.debug(f"Ground truth: {gt}")
            logger.debug("-"*100)
    
    results = evaluate_predictions(predictions, ground_truths)
    logger.info(f"Results: {results}")
    
    results["model_name"] = model_path
    results["task_name"] = task_name
    results["content"] = []
    for sample_id in range(len(predictions)):
        results["content"].append({
            "id": sample_id,
            "prompt": prompt_list[sample_id],
            "prediction": predictions[sample_id],
            "ground_truth": ground_truths[sample_id]
        })
    
    save_dir = model_path if "outputs/" in model_path else os.path.join("outputs", model_path)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    if debugging:
        save_path = os.path.join(save_dir, f"{task_name}-results-debug.json")
    else:
        save_path = os.path.join(save_dir, f"{task_name}-results.json")
        
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {save_path}")
        
if __name__ == "__main__":
    Fire(main)
