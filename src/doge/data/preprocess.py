from functools import partial
from typing import Any, Dict, List, Optional

from loguru import logger
from transformers import PreTrainedTokenizerBase

__all__ = ["batch_preprocess_fn"]


def batch_preprocess_fn(
        examples: Dict[str, List[Any]], task: str, tokenizer: PreTrainedTokenizerBase = None, task_type: str=None,
) -> Dict[str, List[Any]]:
    task_to_fn = {
        "chat-gen": partial(chat_eval_batch_preprocess_fn, tokenizer=tokenizer),
        "chat-gen-gsm8k": partial(gsm8k_chat_eval_batch_preprocess_fn, tokenizer=tokenizer),
        "chat-profile": partial(chat_profile_batch_preprocess_fn, tokenizer=tokenizer),
        "sft-olmoe-train": partial(sft_train_batch_preprocess_fn, tokenizer=tokenizer, boa_token="<|assistant|>"),
        "sft-deepseek-v2-train": partial(sft_train_batch_preprocess_fn, tokenizer=tokenizer, boa_token="Assistant:"),
        "sft-moonlight-train": partial(sft_train_batch_preprocess_fn, tokenizer=tokenizer,
                                       boa_token="<|im_assistant|>assistant<|im_middle|>"),
        "math-reasoning-llama-3.2-train": partial(reasoning_batch_preprocess_fn, tokenizer=tokenizer, task_type=task_type),
        "math-reasoning-llama-3.2-eval": partial(reasoning_batch_preprocess_fn, tokenizer=tokenizer, is_eval=True, task_type=task_type),
    }
    return task_to_fn[task](examples)


def chat_eval_batch_preprocess_fn(
        examples: Dict[str, List[Any]], tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> Dict[str, List[Any]]:
    """
    Parameters
    ----------
    examples: Dict[str, List[Any]]
        examples to preprocess
    tokenizer: PreTrainedTokenizerBase, optional
        tokenizer to use

    Returns
    -------
    Dict[str, List[Any]]
        preprocessed examples

    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)
    >>> raw_examples = {"messages": [[{"content": "Hello, how are you?", "role": "user"}, {"content": "I am good, how can I help you?", "role": "system"}]]}
    >>> preprocessed_examples = chat_eval_batch_preprocess_fn(raw_examples, tokenizer)
    >>> preprocessed_examples.keys()
    dict_keys(['input_ids'])
    """
    messages_list = [messeges[0]["content"] for messeges in examples["messages"]]
    chat_list = [
        [{"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
         {"role": "user", "content": messages}] for messages in messages_list
    ]
    if tokenizer is None:
        return {"content": chat_list}
    else:
        input_ids_list = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True)
        return {"input_ids": input_ids_list, "content": messages_list}


def gsm8k_chat_eval_batch_preprocess_fn(
        examples: Dict[str, List[Any]], tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> Dict[str, List[Any]]:
    chat_list = [
        [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
         {"role": "user", "content": message}] for message in examples["question"]
    ]
    if tokenizer is None:
        return {"content": chat_list}
    else:
        input_ids_list = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True)
        return {"input_ids": input_ids_list, "content": examples["question"]}


def chat_profile_batch_preprocess_fn(
        examples: Dict[str, List[Any]], tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> Dict[str, List[Any]]:
    question_list = []
    response_list = []
    source_list = []
    for q, r, s in zip(examples["question"], examples["response"], examples["source"]):
        if q is not None and r is not None and s is not None:
            question_list.append(q)
            response_list.append(r)
            source_list.append(s)
        else:
            logger.warning(f"Skipping example with None values: {q}, {r}, {s}")

    chat_list = [
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": question},
         {"role": "assistant", "content": response}]
        for question, response in zip(question_list, response_list)
    ]

    if tokenizer is None:
        return {"content": chat_list, "source": source_list}
    else:
        input_ids_list = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True)
        return {"input_ids": input_ids_list, "source": source_list}


def apply_general_chat_template(
        question: str,
        tokenizer: PreTrainedTokenizerBase,
        response: Optional[str] = None,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    if response is None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        messages.append({"role": "assistant", "content": response})
        return tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)


def sft_train_batch_preprocess_fn(
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizerBase,
        boa_token: str,
):
    if tokenizer is None:
        raise ValueError("Tokenizer is required for SFT training.")

    # 1. apply general chat template to each example
    all_chat_texts = []

    for question, response in zip(examples["question"], examples["response"]):
        chat_text = apply_general_chat_template(question, response=response, tokenizer=tokenizer)
        all_chat_texts.append(chat_text)

    # 2. Tokenize the chat
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for chat_text in all_chat_texts:
        encoded = tokenizer(chat_text, padding=False, truncation=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # 3. Only apply LM loss on the assistant's response & "<|endoftext|>"
        labels = [-100] * len(input_ids)

        assistant_token_id = tokenizer(boa_token, add_special_tokens=False)["input_ids"]

        pos_assistant = -1

        i = 0
        while i <= len(input_ids) - len(assistant_token_id):
            matched = True
            for j in range(len(assistant_token_id)):
                if input_ids[i + j] != assistant_token_id[j]:
                    matched = False
                    break

            if matched:
                pos_assistant = i + len(assistant_token_id) - 1
                break
            i += 1

        if pos_assistant != -1:
            for i in range(pos_assistant + 1, len(input_ids)):
                labels[i] = input_ids[i]
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
        else:
            print("Assistant token not found in the input_ids. SKIPPING...")

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    }


def reasoning_batch_preprocess_fn(
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizerBase,
        is_eval: bool = False,
        task_type: str = "math",
):
    """
    Examples
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
    >>> raw_examples = {"prompt": ["Hello, how are you?", "What is the capital of France?"], "response": ["I am good, how can I help you?", "The capital of France is Paris."]}
    >>> preprocessed_examples = reasoning_batch_preprocess_fn(raw_examples, tokenizer)
    >>> preprocessed_examples.keys()
    dict_keys(['input_ids', 'labels'])
    """
    if task_type == "table":
        examples["prompt"] = [f"###Table: {table}\n###Question: {question}" for table, question in zip(examples["table"], examples["question"])]
    elif task_type == "csqa":
        for choices in examples["choices"]:
            assert len(choices["text"]) == 5, f"CSQA choices must have 5 options, but got {len(choices['text'])}"
        examples["prompt"] = [
            f"""Question: {question} (A) {choices["text"][0]} (B) {choices["text"][1]} (C) {choices["text"][2]} (D) {choices["text"][3]} (E) {choices["text"][4]}""" 
            for question, choices in zip(examples["question"], examples["choices"])
        ]
    elif task_type == "arcc":
        # examples["prompt"] = [
        #     f"""Question: {question} (A) {choices["text"][0]} (B) {choices["text"][1]} (C) {choices["text"][2]} (D) {choices["text"][3]}""" 
        #     for question, choices in zip(examples["question"], examples["choices"])
        # ]
        examples["prompt"] = []
        for question, choices in zip(examples["question"], examples["choices"]):
            if len(choices["text"]) == 4:
                examples["prompt"].append(f"""Question: {question} (A) {choices["text"][0]} (B) {choices["text"][1]} (C) {choices["text"][2]} (D) {choices["text"][3]}""")
            elif len(choices["text"]) == 3:
                examples["prompt"].append(f"""Question: {question} (A) {choices["text"][0]} (B) {choices["text"][1]} (C) {choices["text"][2]}""")
            elif len(choices["text"]) == 5:
                examples["prompt"].append(f"""Question: {question} (A) {choices["text"][0]} (B) {choices["text"][1]} (C) {choices["text"][2]} (D) {choices["text"][3]} (E) {choices["text"][4]}""")
            else:
                raise ValueError(f"ARCC choices got {len(choices['text'])}")
    elif "question" in examples and "prompt" not in examples:
        examples["prompt"] = examples["question"]
    
    if task_type == "csqa":
        examples["response"] = [f"Answer: {answer}" for answer in examples["answerKey"]]
    elif task_type == "arcc":
        examples["response"] = [f"Answer: {answer}" for answer in examples["answerKey"]]
    elif "answer" in examples and "response" not in examples:
        examples["response"] = examples["answer"]
        
    TASK_TO_INSTRUCTION = {
        "math": "Please reason step by step, and put your final answer within \\boxed{{}}.",
        "table": "Please reason step by step, and put your final answer anfter 'Answer:'.",
        "arcc": "Please reason step by step, select your final answer from A, B, C, or D and put your final answer anfter 'Answer:'.",
        "csqa": "Please reason step by step, select your final answer from A, B, C, D, or E and put your final answer anfter 'Answer:'.",
    }

    response_list = [f"{response}<|eot_id|>" for response in examples["response"]]
    chat_list = [
        [{"role": "system", "content": TASK_TO_INSTRUCTION[task_type]},
         {"role": "user", "content": message}] for message in examples["prompt"]
    ]
    input_str_list = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True, tokenize=False)
    input_str_list = [f"{input_str}<think>\n" for input_str in input_str_list]
    if is_eval:
        input_ids = tokenizer(input_str_list)["input_ids"]
        return {"input_ids": input_ids, "response": examples["response"], "prompt": input_str_list}
    
    input_id_list = tokenizer(input_str_list, add_special_tokens=False)["input_ids"]
    response_id_list = tokenizer(response_list, add_special_tokens=False)["input_ids"]

    input_ids = [
        input_id + response_id
        for input_id, response_id in zip(input_id_list, response_id_list)
    ]
    labels = [
        [-100] * len(input_id) + response_id
        for input_id, response_id in zip(input_id_list, response_id_list)
    ]

    filtered_input_ids = []
    filtered_labels = []

    for i in range(len(input_ids)):
        if len(input_ids[i]) > tokenizer.model_max_length:
            logger.info(f"Skip sample {i} with length {len(input_ids[i])}")
            continue

        filtered_input_ids.append(input_ids[i])
        filtered_labels.append(labels[i])

    return {
        "input_ids": filtered_input_ids,
        "labels": filtered_labels
    }
