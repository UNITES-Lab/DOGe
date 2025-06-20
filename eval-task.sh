export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES=2

TASK_NAME=${1:-"gsm8k"}

MODEL_PATH="outputs/llama-3.2-1b-distill--qwen-antidistill-coef0.00003-temp2-epoch2-lr5e-5-checkpoint-60/checkpoint-177"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name=$TASK_NAME

MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name=$TASK_NAME


MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name=$TASK_NAME


MODEL_PATH="outputs/qwen7b-antidistill-coef0.00003-temp2-epoch2-lr5e-5/checkpoint-60"
python scripts/generate-eval.py \
    --model_path="$MODEL_PATH" --task_name=$TASK_NAME



