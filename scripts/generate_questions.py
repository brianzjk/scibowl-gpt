from __future__ import annotations

import sys
from pathlib import Path

from scibowl.generate.batch import load_generation_batch_config, run_generation_batch


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/generate_questions.py <config.yaml>")
    config_path = Path(sys.argv[1])
    config = load_generation_batch_config(config_path)
    records = run_generation_batch(config_path)
    print(f"Wrote {len(records)} generated question runs to {config.output_path}")


if __name__ == "__main__":
    main()
