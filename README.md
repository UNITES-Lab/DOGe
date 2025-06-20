# 🐶 DOGe

**_Defensive Output Generation for LLM Protection Against Knowledge Distillation_**

TLDR: We make LLMs much more difficult to distill while maintaining their performance/quality.

* Checkpoints will be released soon.

---

### 1. Setup Environment

Simply run `bash setup.sh` under the root directory of this repository to set up the environment.

### 2. Replicate DOGe

#### 2.1 Generate training data

Launch the model through vllm, for example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 2333
```

Then check the `generate.sh` script, you might want to comment/modify some lines/parameters to fit your needs.

```bash
bash generate.sh
```

#### 2.2 Train DOGe on the teacher model

Check out the `train-doge.sh` script, you might want to comment/modify some lines/parameters to fit your needs.

```bash
bash train-doge.sh
```

#### 2.3 Distill DOGe/vanilla teacher model to a student model

Check out the `train-distill.sh` script, you might want to comment/modify some lines/parameters to fit your needs.

```bash
bash train-distill.sh
```

#### 2.4 Evaluate the distilled student model (or any model)

Check out the `eval-task.sh` script, you might want to comment/modify some lines/parameters to fit your needs.

```bash
bash eval-task.sh
```
