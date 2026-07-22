from pathlib import Path

from scibowl.training.lora import _resolve_warmup_steps, _tokenize_messages, load_lora_sft_config
from scibowl.utils.ids import make_id


def test_load_lora_sft_config() -> None:
    tmp_path = Path("tests_runtime") / make_id("lora_cfg")
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  model_name_or_path: Qwen/Qwen2.5-7B-Instruct",
                "  attn_implementation: sdpa",
                "  use_bf16: true",
                "adapter:",
                "  r: 8",
                "data:",
                "  train_path: ../data/train.jsonl",
                "  val_path: ../data/val.jsonl",
                "training:",
                "  output_dir: ../data/models/test",
                "  num_train_epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_lora_sft_config(config_path)

    assert config.model.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert config.model.attn_implementation == "sdpa"
    assert config.adapter.r == 8
    assert config.data.train_path == "../data/train.jsonl"
    assert config.training.output_dir == "../data/models/test"


def test_load_gpt_oss_example_config() -> None:
    config = load_lora_sft_config(Path("configs/lora_sft.gptoss20b.example.yaml"))

    assert config.model.model_name_or_path == "openai/gpt-oss-20b"
    assert config.model.quantization == "mxfp4"
    assert config.adapter.target_modules == "all-linear"
    assert "7.mlp.experts.gate_up_proj" in config.adapter.target_parameters


def test_resolve_warmup_steps_from_ratio() -> None:
    warmup_steps = _resolve_warmup_steps(
        train_examples_count=1000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=10,
        num_train_epochs=2,
        warmup_steps=None,
        warmup_ratio=0.05,
    )

    assert warmup_steps == 10


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = "".join(parts)
        if tokenize:
            return [ord(character) for character in rendered]
        return rendered

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=False):
        input_ids = [ord(character) for character in text]
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


def test_tokenize_messages_preserves_assistant_tokens_when_prompt_is_long() -> None:
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "S" * 120},
        {"role": "user", "content": "U" * 120},
        {"role": "assistant", "content": "{\"question_text\":\"Q\",\"answer_text\":\"ANSWER: A\",\"choices\":[]}"},
    ]

    tokenized = _tokenize_messages(messages, tokenizer=tokenizer, max_length=80)
    non_masked = [token for token in tokenized.labels if token != -100]

    assert non_masked
    assert len(tokenized.input_ids) <= 80
    assert non_masked == tokenized.input_ids[-len(non_masked) :]
