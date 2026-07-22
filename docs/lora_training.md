# LoRA Training

This repo now includes a local LoRA SFT path that trains against the exported chat-format SFT dataset.

## What It Trains

The intended `v1` objective is:

- Science Bowl formatting
- category fit
- clue-flow conventions
- answer-line conventions

It is not meant to directly optimize overall question quality. That is a later RLHF / preference-tuning step.

## Prerequisites

Install the repo plus training dependencies:

```powershell
pip install -e .[dev,train]
```

Install a CUDA-enabled PyTorch build separately if you do not already have one.

## Model-Specific Notes

### Qwen 3.5 9B

If you use the `Qwen3.5-9B` example config, also follow the current Qwen model-card guidance and install:

```powershell
.venv\Scripts\python.exe -m pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@main"
.venv\Scripts\python.exe -m pip install pillow torchvision
```

The Qwen example config is set up for `4-bit QLoRA`, which is the recommended first run for `Qwen3.5-9B`.

Install `bitsandbytes` as well:

```powershell
.venv\Scripts\python.exe -m pip install bitsandbytes
```

### GPT-OSS 20B

If you use the GPT-OSS example config, use the newer dependency floor recommended in OpenAI's cookbook:

```powershell
.venv\Scripts\python.exe -m pip install -U "transformers>=4.55.0" "peft>=0.17.0" "trl>=0.20.0" kernels
```

The GPT-OSS trainer path uses the model tokenizer's chat template, so the Harmony response format is applied automatically during tokenization.

## Files

- training module: `src/scibowl/training/lora.py`
- Qwen example config: `configs/lora_sft.example.yaml`
- GPT-OSS example config: `configs/lora_sft.gptoss20b.example.yaml`
- entrypoint script: `scripts/train_lora_sft.py`

## Run It

From the repo root:

```powershell
.venv\Scripts\python.exe scripts\train_lora_sft.py configs\lora_sft.example.yaml
```

For GPT-OSS:

```powershell
.venv\Scripts\python.exe scripts\train_lora_sft.py configs\lora_sft.gptoss20b.example.yaml
```

The Qwen example config uses:

- train: `../data/processed/training/sft_v1/sft_examples_train_materialized.jsonl`
- val: `../data/processed/training/sft_v1/sft_examples_val.jsonl`
- output: `../models/qwen35_9b_scibowl_sft_v1`
- model load: `4-bit QLoRA`
- max length: `1024`

The GPT-OSS example config uses:

- train: `../data/processed/training/sft_v1/sft_examples_train_materialized.jsonl`
- val: `../data/processed/training/sft_v1/sft_examples_val.jsonl`
- output: `../models/gpt_oss_20b_scibowl_sft_v1`
- model load: `MXFP4 + LoRA`
- max length: `2048`

## Notes

- The script trains on assistant tokens only. Prompt tokens are masked out of the loss.
- It expects the SFT dataset to already be exported in the chat format produced by `export-sft-dataset`.
- The same exported chat dataset can be used for both Qwen and GPT-OSS; GPT-OSS compatibility comes from the tokenizer's chat template, not from a separate dataset export format.
- It saves the adapter/model, tokenizer, resolved config, and trainer log history into the output directory.
- It prints progress while tokenizing, training, evaluating, and checkpointing.
- It will save normal `checkpoint-*` folders during training according to `save_steps`.
- If you stop it with `Ctrl+C`, it saves a recovery checkpoint before exiting. When the Transformers trainer checkpoint path is available, that recovery save is a normal `checkpoint-*` folder that can be resumed directly.
- To resume later, set `training.resume_from_checkpoint` to a checkpoint path or to `last`.
- It now fails fast before model download or tokenization if the active `.venv` is using CPU-only PyTorch.

## Recommended First Run

Start with the config that matches your base model and change only:

- `model.model_name_or_path`
- `training.output_dir`
- optionally `training.resume_from_checkpoint`
- optionally `model.load_in_4bit`
- optionally `data.max_length`

If memory is tight, first lower:

- `data.max_length`
- `gradient_accumulation_steps`

If throughput is too low, reduce:

- model size
- max length

Do not start by scaling epochs or LoRA rank aggressively.

## Recommended First Run Settings

For a first overnight run on a consumer GPU, the Qwen example config is intentionally conservative:

- `Qwen/Qwen3.5-9B`
- `load_in_4bit: true`
- `use_bf16: true`
- `max_length: 1024`
- `batch_size: 1`
- `gradient_accumulation_steps: 16`

That is a safer starting point than trying full bf16 LoRA on the 9B model.

For GPT-OSS, the official cookbook example uses:

- `openai/gpt-oss-20b`
- `attn_implementation: eager`
- `Mxfp4Config(dequantize=True)`
- `r: 8`
- `lora_alpha: 16`
- `target_modules: all-linear`
- additional expert projections via `target_parameters`
- `max_length: 2048`

This repo's `lora_sft.gptoss20b.example.yaml` mirrors that setup while keeping the original Qwen path available.
