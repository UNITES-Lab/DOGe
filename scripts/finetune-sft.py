import logging
import math
import os

import datasets
import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from doge.data import batch_preprocess_fn, CustomDataCollatorWithPadding
from doge.models.deepseek import DeepseekV3ForCausalLM
from fire import Fire
from pandas import Timedelta
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

set_seed(233)
logger = get_logger(__name__)


def train_sft(
        base_model_name: str = "allenai/OLMoE-1B-7B-0125",
        dataset_name: str = "ANONYMOUS/sft-dataset-original-filtered",
        dataset_filter_condition: str = None,
        max_length: int = 8192,
        batch_size_per_device: int = 4,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 2,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        output_dir: str = "./outputs/",
        num_workers: int = 4,
        checkpointing_steps: int = -1,
        logging_steps: int = 1,
        debugging: bool = False,
        enable_lora: bool = False,
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

    if "jsonl" in dataset_name:
        raw_datasets = load_dataset("json", data_files=dataset_name)["train"]
    else:
        raw_datasets = load_dataset(dataset_name, split="train", trust_remote_code=True)
    if dataset_filter_condition is not None:
        # e.g. "example['source'] == 'riddle_sense'"
        before_filter = len(raw_datasets)
        raw_datasets = raw_datasets.filter(lambda example: eval(dataset_filter_condition))
        after_filter = len(raw_datasets)
        logger.info(f"Filtered {before_filter - after_filter} samples from {before_filter} samples")

    # debugging
    if debugging:
        raw_datasets = raw_datasets.select(range(1000))
        checkpointing_steps = 2

    if "olmoe" in base_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125-Instruct", trust_remote_code=True)
    elif "deepseek-v2" in base_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
    elif "moonlight" in base_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)
    elif "llama-3.2" in base_model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"Tokenizer for {base_model_name} not implemented.")
    tokenizer.model_max_length = max_length
    if "moonlight" in base_model_name.lower():
        model = DeepseekV3ForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

    # lora
    if enable_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules="all-linear"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    with accelerator.main_process_first():
        columns_names = raw_datasets.column_names
        if "olmoe" in base_model_name.lower():
            proc_name = "sft-olmoe-train"
        elif "deepseek-v2" in base_model_name.lower():
            proc_name = "sft-deepseek-v2-train"
        elif "moonlight" in base_model_name.lower():
            proc_name = "sft-moonlight-train"
        elif "llama-3.2" in base_model_name.lower():
            proc_name = "math-reasoning-llama-3.2-train"
        else:
            raise NotImplementedError(f"Preprocess for {base_model_name} not implemented.")
        sft_dataset = raw_datasets.map(
            lambda x: batch_preprocess_fn(x, task=proc_name, tokenizer=tokenizer),
            batched=True,
            remove_columns=columns_names,
            num_proc=num_workers,
        )
        logger.info(sft_dataset[0])
        logger.info(sft_dataset[1])
    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, max_length=max_length)
    dataloader = DataLoader(
        sft_dataset,
        collate_fn=data_collator,
        batch_size=batch_size_per_device,
        num_workers=num_workers,
        shuffle=True
    )
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * num_train_epochs
    if checkpointing_steps == -1:
        checkpointing_steps = num_update_steps_per_epoch
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

    accelerator.init_trackers(project_name="doge")

    # Train!
    total_batch_size = batch_size_per_device * accelerator.num_processes * gradient_accumulation_steps

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
    logger.info(f"logging_steps = {logging_steps}")

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    model.train()
    completed_steps = 0

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dataloader):
            model.train()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
                unwrapped_model = accelerator.unwrap_model(model)
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
            }, step=completed_steps)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    checkpointing_dir = os.path.join(output_dir, f"checkpoint-{completed_steps}")
    unwrapped_model.save_pretrained(
        checkpointing_dir,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model),
    )
    logger.info(f"Saving model checkpoint to {checkpointing_dir}")
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training completed.")


if __name__ == "__main__":
    Fire(train_sft)
