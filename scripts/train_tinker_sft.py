from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scibowl.training import validate_tinker_sft_dataset


@dataclass
class FixedSplitBuilder:
    train_builder: Any
    validation_builder: Any

    def __call__(self):
        train_dataset, _ = self.train_builder()
        validation_dataset, _ = self.validation_builder()
        return train_dataset, validation_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run clean Science Bowl SFT on Tinker.')
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('log_path')
    parser.add_argument('--model', default='thinkingmachines/Inkling')
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lora-rank', type=int, default=32)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--max-steps', type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_tinker_sft_dataset(args.data_dir)

    try:
        from tinker_cookbook import cli_utils, model_info
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised import train
        from tinker_cookbook.supervised.data import FromConversationFileBuilder
        from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    except ImportError as exc:
        raise SystemExit('Install the Tinker extra with: pip install -e .[tinker]') from exc

    renderer_name = model_info.get_recommended_renderer_name(args.model)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    train_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(args.data_dir / 'clean_sft_train.jsonl'),
    )
    validation_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(args.data_dir / 'clean_sft_val.jsonl'),
    )

    config = train.Config(
        log_path=args.log_path,
        model_name=args.model,
        recipe_name='scibowl_clean_sft',
        renderer_name=renderer_name,
        dataset_builder=FixedSplitBuilder(train_builder, validation_builder),
        learning_rate=args.learning_rate,
        lr_schedule='linear',
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
    )
    cli_utils.check_log_dir(args.log_path, behavior_if_exists='ask')
    asyncio.run(train.main(config))


if __name__ == '__main__':
    main()
