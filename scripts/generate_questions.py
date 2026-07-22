from __future__ import annotations

import argparse
from pathlib import Path

from scibowl.generate.batch import load_generation_batch_config, run_generation_batch
from scibowl.llm.runtime import apply_runtime_model_overrides


def main() -> None:
    parser = argparse.ArgumentParser(prog="generate_questions.py")
    parser.add_argument("config_path")
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--writer-model")
    parser.add_argument("--writer-base-url")
    parser.add_argument("--verifier-model")
    parser.add_argument("--verifier-base-url")
    parser.add_argument("--writer-timeout-seconds", type=int)
    parser.add_argument("--verifier-timeout-seconds", type=int)
    parser.add_argument("--disable-writer-fallback", action="store_true")
    args = parser.parse_args()

    apply_runtime_model_overrides(
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        writer_model=args.writer_model,
        writer_base_url=args.writer_base_url,
        verifier_model=args.verifier_model,
        verifier_base_url=args.verifier_base_url,
        writer_timeout_seconds=args.writer_timeout_seconds,
        verifier_timeout_seconds=args.verifier_timeout_seconds,
        disable_writer_fallback=args.disable_writer_fallback,
    )

    config_path = Path(args.config_path)
    config = load_generation_batch_config(config_path)
    records = run_generation_batch(config_path)
    print(f"Wrote {len(records)} generated question runs to {config.output_path}")


if __name__ == "__main__":
    main()
