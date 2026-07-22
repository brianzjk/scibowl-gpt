from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class LoraModelConfig(BaseModel):
    model_name_or_path: str
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    use_bf16: bool = True
    use_fp16: bool = False
    load_in_4bit: bool = False
    quantization: Literal["none", "bnb_4bit", "mxfp4"] | None = None
    mxfp4_dequantize: bool = True
    gradient_checkpointing: bool = True


class LoraAdapterConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: str | list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    target_parameters: list[str] = Field(default_factory=list)


class SFTDataConfig(BaseModel):
    train_path: str
    val_path: str | None = None
    max_length: int = 1536


class SFTTrainingConfig(BaseModel):
    output_dir: str
    resume_from_checkpoint: str | None = None
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float | None = 0.03
    warmup_steps: int | None = None
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 200
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_total_limit: int = 2
    seed: int = 42


class LoraSFTConfig(BaseModel):
    model: LoraModelConfig
    adapter: LoraAdapterConfig = Field(default_factory=LoraAdapterConfig)
    data: SFTDataConfig
    training: SFTTrainingConfig


@dataclass
class TokenizedSFTExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def load_lora_sft_config(path: Path) -> LoraSFTConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return LoraSFTConfig.model_validate(payload)


def train_lora_from_config(config_path: Path) -> None:
    config = load_lora_sft_config(config_path)
    _train_lora(config=config, config_path=config_path)


def _train_lora(config: LoraSFTConfig, config_path: Path) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Mxfp4Config,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_lora_config.json"
    resolved_config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    _preflight_torch_environment(config, torch)
    _print_status(f"Model: {config.model.model_name_or_path}")
    _print_status(f"Train data: {config.data.train_path}")
    _print_status(f"Val data: {config.data.val_path or '<none>'}")
    _print_status(f"Output dir: {output_dir}")
    _print_status(f"Max length: {config.data.max_length}")
    _print_status(f"Torch: {torch.__version__}")
    _print_status(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if "Qwen3.5" in config.model.model_name_or_path:
        _print_status(
            "Qwen3.5 note: the model card recommends the latest Hugging Face Transformers build "
            "and having torchvision plus pillow installed."
        )
    if "gpt-oss" in config.model.model_name_or_path.casefold():
        _print_status(
            "GPT-OSS note: the model relies on its chat template / harmony format. "
            "This trainer uses tokenizer.apply_chat_template, so the exported chat dataset remains compatible."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation
    quantization_mode = _resolve_quantization_mode(config.model)
    if quantization_mode == "bnb_4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.model.use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif quantization_mode == "mxfp4":
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=config.model.mxfp4_dequantize)
        model_kwargs["torch_dtype"] = torch.bfloat16 if config.model.use_bf16 else torch.float16
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        **model_kwargs,
    )
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if quantization_mode in {"bnb_4bit", "mxfp4"} and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=config.adapter.r,
        lora_alpha=config.adapter.lora_alpha,
        lora_dropout=config.adapter.lora_dropout,
        bias=config.adapter.bias,
        task_type="CAUSAL_LM",
        target_modules=config.adapter.target_modules,
        target_parameters=config.adapter.target_parameters or None,
    )
    model = get_peft_model(model, peft_config)

    train_examples = _load_tokenized_examples(
        Path(config.data.train_path),
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        label="train",
    )
    val_examples = (
        _load_tokenized_examples(
            Path(config.data.val_path),
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            label="val",
        )
        if config.data.val_path
        else None
    )
    _print_status(f"Tokenized train examples: {len(train_examples)}")
    _print_status(f"Tokenized val examples: {len(val_examples) if val_examples is not None else 0}")
    warmup_steps = _resolve_warmup_steps(
        train_examples_count=len(train_examples),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        warmup_steps=config.training.warmup_steps,
        warmup_ratio=config.training.warmup_ratio,
    )
    _print_status(f"Warmup steps: {warmup_steps}")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        eval_strategy=config.training.eval_strategy if val_examples is not None else "no",
        eval_steps=config.training.eval_steps if val_examples is not None else None,
        save_total_limit=config.training.save_total_limit,
        seed=config.training.seed,
        bf16=config.model.use_bf16,
        fp16=config.model.use_fp16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_AssistantOnlyDataset(train_examples),
        eval_dataset=_AssistantOnlyDataset(val_examples) if val_examples is not None else None,
        data_collator=_AssistantOnlyCollator(tokenizer),
        callbacks=[_build_progress_callback(TrainerCallback)],
    )
    resume_checkpoint = _resolve_resume_checkpoint(config.training.resume_from_checkpoint, output_dir)
    if resume_checkpoint is not None:
        _print_status(f"Resuming from checkpoint: {resume_checkpoint}")

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        _print_status("Interrupted. Saving recovery checkpoint.")
        _save_interrupt_checkpoint(trainer, tokenizer, output_dir)
        raise

    _print_status(f"Training complete. Saving final adapter to {output_dir}")
    _save_checkpoint(trainer, tokenizer, output_dir)
    metrics = trainer.state.log_history
    (output_dir / "train_log_history.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "source_config_path.txt").write_text(str(config_path.resolve()), encoding="utf-8")


def _load_tokenized_examples(path: Path, tokenizer: Any, max_length: int, label: str) -> list[TokenizedSFTExample]:
    rows: list[TokenizedSFTExample] = []
    label_token_counts: list[int] = []
    _print_status(f"Tokenizing {label} dataset from {path}")
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = _tokenize_messages(payload["messages"], tokenizer=tokenizer, max_length=max_length)
            label_count = sum(1 for token in record.labels if token != -100)
            if label_count == 0:
                example_id = payload.get("example_id", f"{label}:{index}")
                raise RuntimeError(
                    f"Example {example_id} has zero assistant label tokens after tokenization. "
                    "This usually means max_length is too small for the prompt+answer pair or the SFT prompt is too long."
                )
            rows.append(record)
            label_token_counts.append(label_count)
            if index % 500 == 0:
                _print_status(f"Tokenized {label} examples: {index}")
    if label_token_counts:
        average = sum(label_token_counts) / len(label_token_counts)
        _print_status(
            f"{label.capitalize()} assistant label tokens: min={min(label_token_counts)} "
            f"avg={average:.1f} max={max(label_token_counts)}"
        )
    return rows


def _tokenize_messages(messages: list[dict[str, str]], tokenizer: Any, max_length: int) -> TokenizedSFTExample:
    if len(messages) < 3 or messages[-1]["role"] != "assistant":
        raise ValueError("Expected chat messages ending in an assistant turn")

    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_tokens = tokenizer(full_text, truncation=False, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt_text, truncation=False, add_special_tokens=False)

    full_input_ids = list(full_tokens["input_ids"])
    prompt_input_ids = list(prompt_tokens["input_ids"])
    if len(full_input_ids) < len(prompt_input_ids) or full_input_ids[: len(prompt_input_ids)] != prompt_input_ids:
        raise RuntimeError("Prompt tokenization is not a prefix of the full conversation tokenization.")

    assistant_input_ids = full_input_ids[len(prompt_input_ids) :]
    if not assistant_input_ids:
        raise RuntimeError("Assistant response tokenization is empty.")
    if len(assistant_input_ids) >= max_length:
        raise RuntimeError(
            "Assistant response alone exceeds max_length. Increase data.max_length or shorten assistant outputs."
        )

    prompt_budget = max_length - len(assistant_input_ids)
    kept_prompt_ids = prompt_input_ids[-prompt_budget:] if len(prompt_input_ids) > prompt_budget else prompt_input_ids
    input_ids = kept_prompt_ids + assistant_input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(kept_prompt_ids) + assistant_input_ids
    return TokenizedSFTExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


class _AssistantOnlyDataset:
    def __init__(self, records: list[TokenizedSFTExample]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        return {
            "input_ids": record.input_ids,
            "attention_mask": record.attention_mask,
            "labels": record.labels,
        }


class _AssistantOnlyCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            batch_labels.append(feature["labels"] + [-100] * pad_length)

        import torch

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _save_checkpoint(trainer: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(path))
    tokenizer.save_pretrained(str(path))
    trainer.save_state()


def _save_interrupt_checkpoint(trainer: Any, tokenizer: Any, output_dir: Path) -> None:
    if hasattr(trainer, "_save_checkpoint"):
        trainer._save_checkpoint(trainer.model, trial=None)
        latest = _resolve_resume_checkpoint("last", output_dir)
        if latest is not None:
            _print_status(f"Recovery checkpoint saved at {latest}")
        return

    interrupt_dir = output_dir / "interrupt_checkpoint"
    _save_checkpoint(trainer, tokenizer, interrupt_dir)
    _print_status(f"Recovery checkpoint saved at {interrupt_dir}")


def _resolve_resume_checkpoint(value: str | None, output_dir: Path) -> str | None:
    if not value:
        return None
    if value != "last":
        return value

    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda item: _checkpoint_step(item.name),
    )
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def _preflight_torch_environment(config: LoraSFTConfig, torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this venv. Current torch build is CPU-only or cannot see the GPU. "
            "Install a CUDA-enabled PyTorch build in .venv, then retry."
        )
    if config.model.use_bf16 and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Config requests bf16, but the active CUDA setup does not report bf16 support.")
    if config.model.use_bf16 and config.model.use_fp16:
        raise RuntimeError("Set only one of use_bf16 or use_fp16 to true.")


def _resolve_warmup_steps(
    train_examples_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_steps: int | None,
    warmup_ratio: float | None,
) -> int:
    if warmup_steps is not None:
        return warmup_steps

    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_examples_count / effective_batch))
    total_train_steps = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    if warmup_ratio is None:
        return 0
    return max(0, int(total_train_steps * warmup_ratio))


def _resolve_quantization_mode(config: LoraModelConfig) -> str:
    if config.quantization is not None:
        return config.quantization
    if config.load_in_4bit:
        return "bnb_4bit"
    return "none"


def _checkpoint_step(name: str) -> int:
    suffix = name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else -1


def _print_status(message: str) -> None:
    print(f"[lora-sft] {message}", flush=True)


def _build_progress_callback(base_class: type[Any]) -> Any:
    class ProgressPrinterCallback(base_class):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            _print_status("Training started")

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not logs:
                return
            filtered = {key: value for key, value in logs.items() if key != "total_flos"}
            _print_status(f"Log step={state.global_step}: {json.dumps(filtered, ensure_ascii=False)}")

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            _print_status(f"Checkpoint saved at step {state.global_step}")

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            _print_status(f"Evaluation step={state.global_step}: {json.dumps(metrics or {}, ensure_ascii=False)}")

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            _print_status("Training finished")

    return ProgressPrinterCallback()
