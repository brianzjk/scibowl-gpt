from __future__ import annotations

import sys
from pathlib import Path

from scibowl.training.lora import train_lora_from_config


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/train_lora_sft.py <config.yaml>")
    train_lora_from_config(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
