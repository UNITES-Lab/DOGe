import logging
import math
import os
from typing import Any, Dict, List

import datasets
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from fire import Fire
from pandas import Timedelta
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, PreTrainedTokenizerBase

from doge.data import CustomDataCollatorWithPadding
from doge.models.distill_wrapper import AntiDistillWrapper

set_seed(233)
logger = get_logger(__name__)


def process_r1_thinking_data(
        examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[Any]]:
    input_ids_list = []
    labels_list = []
    num_samples = len(examples["response"])

    for i in range(num_samples):
        question = examples["prompt"][i]
        response = examples["response"][i]
        reasoning = examples["reasoning_response"][i]
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
            {"role": "user", "content": question}
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)  # end with "<think>\n"
        response_with_thinking = f"{reasoning}\n</think>{response}{tokenizer.eos_token}"
        labels = tokenizer.encode(response_with_thinking, add_special_tokens=False)

        input_ids.extend(labels)
        labels = [-100] * (len(input_ids) - len(labels)) + labels  # label -100 for prefilling tokens
        if len(input_ids) > tokenizer.model_max_length:
            # skip
            logger.info(f"Skip sample {i} with length {len(input_ids)}")
            continue

        assert len(labels) == len(input_ids)
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }


def train_antidistill(
        anti_kd_coef: float,
        kd_temperature: float,
        output_dir: str,
        teacher_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        proxy_model_name: str = "Qwen/Qwen2.5-3B",
        dataset_path: str = "data/r1-qwen-7b-gsm8k/distillation_data.jsonl",
        max_length: int = 2048,
        batch_size_per_device: int = 4,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 2,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        num_workers: int = 4,
        checkpointing_steps: int = -1,
        debugging: bool = False,
        lm_head_projector_dim: int = 0,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, project_dir=output_dir, log_with="wandb",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=Timedelta(hours=1))],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if "jsonl" in dataset_path.lower():
        logger.info(f"Loading dataset from {dataset_path} at local.")
        raw_datasets = load_dataset("json", data_files=dataset_path)["train"]
    else:
        raise NotImplementedError(f"Dataset {dataset_path} not implemented.")

    # debugging
    if debugging:
        raw_datasets = raw_datasets.select(range(100))
        checkpointing_steps = 2

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)

    tokenizer.model_max_length = max_length
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, trust_remote_code=True)
    proxy_model = AutoModelForCausalLM.from_pretrained(proxy_model_name, trust_remote_code=True)
    model = AntiDistillWrapper(
        teacher_model=teacher_model,
        proxy_model=proxy_model,
        anti_kd_coef=anti_kd_coef,
        kd_temperature=kd_temperature,
        lm_head_projector=lm_head_projector_dim>0,
        lm_head_projector_dim=lm_head_projector_dim,
    )
    for n, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f"{n} requires grad.")

    with accelerator.main_process_first():
        columns_names = raw_datasets.column_names
        sft_dataset = raw_datasets.map(
            lambda x: process_r1_thinking_data(x, tokenizer=tokenizer),
            batched=True,
            remove_columns=columns_names,
            num_proc=num_workers,
        )
        logger.info(sft_dataset[0])
        logger.info(sft_dataset[1])

    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, max_length=max_length)
    dataloader = DataLoader(
        sft_dataset,
        collate_fn=data_collator, batch_size=batch_size_per_device, num_workers=num_workers, shuffle=True
    )
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0
        },
    ]
    logger.info(optimizer_grouped_parameters)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    warmup_steps = int(warmup_ratio * num_training_steps)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_epochs * len(dataloader),
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    logger.info(f"num_training_steps = {num_training_steps}")

    accelerator.init_trackers(project_name="anti-distill")

    # Train!
    total_batch_size = batch_size_per_device * accelerator.num_processes * gradient_accumulation_steps

    if checkpointing_steps < 0:
        logger.info(f"Setting checkpointing_steps to {num_update_steps_per_epoch}")
        checkpointing_steps = num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"total_batch_size = {total_batch_size}")
    logger.info(f"gradient_accumulation_steps = {gradient_accumulation_steps}")
    logger.info(f"num_train_epochs = {num_train_epochs}")
    logger.info(f"learning_rate = {learning_rate}")
    logger.info(f"weight_decay = {weight_decay}")
    logger.info(f"warmup_ratio = {warmup_ratio}")
    logger.info(f"output_dir = {output_dir}")
    logger.info(f"num_workers = {num_workers}")
    logger.info(f"checkpointing_steps = {checkpointing_steps}")
    logger.info("***** Anti Distillation *****")
    logger.info(f"kd_temperature = {kd_temperature}")
    logger.info(f"anti_kd_coef = {anti_kd_coef}")

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    model.train()
    completed_steps = 0

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            model.train()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                lm_loss = outputs.lm_loss
                kd_loss = outputs.kd_loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
                unwrapped_model = accelerator.unwrap_model(model).teacher_model
                unwrapped_model.save_pretrained(
                    checkpointing_dir,
                    is_main_process=accelerator.is_main_process,
                    state_dict=accelerator.get_state_dict(model),
                )

            if completed_steps > num_training_steps:
                break

            accelerator.log({
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "lm_loss": lm_loss.item(),
                "kd_loss": kd_loss.item(),
            }, step=completed_steps)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model).teacher_model
    checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
    unwrapped_model.save_pretrained(
        checkpointing_dir,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model),
    )
    logger.info(f"Saving model checkpoint to {checkpointing_dir}")
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training completed")


if __name__ == "__main__":
    Fire(train_antidistill)
