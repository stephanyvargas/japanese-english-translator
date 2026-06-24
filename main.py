#!/usr/bin/env python3
"""
Japanese-to-English translator.

Modes:
  python3 main.py                        # conversation mode (default): mic always on,
                                         #   translates every 8s with full context
  python3 main.py --interval 12          # adjust how often a chunk is processed (seconds)
  python3 main.py --once                 # single utterance, full quality pipeline, then exit
  python3 main.py --text "日本語テキスト"  # translate typed text, full quality pipeline, exit
  python3 main.py [--once|--text] --notes  # include translator's notes

Model selection (default: sonnet):
  python3 main.py --model sonnet         # claude-sonnet-4-6  — balanced cost/quality
  python3 main.py --model opus           # claude-opus-4-8    — highest quality
  python3 main.py --model haiku          # claude-haiku-4-5   — fastest, cheapest
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from translator.pipeline import run, run_conversation, run_from_mic  # noqa: E402

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-8",
    "haiku":  "claude-haiku-4-5",
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Japanese -> English translator (quality-focused)")
    parser.add_argument("--text", "-t", metavar="TEXT", help="Translate typed Japanese text and exit")
    parser.add_argument("--once", "-1", action="store_true",
                        help="Single mic utterance with full quality pipeline, then exit")
    parser.add_argument("--notes", "-n", action="store_true",
                        help="Print translator's notes (--once and --text only)")
    parser.add_argument("--model", "-m", default="sonnet", metavar="MODEL",
                        help="Model to use: sonnet (default), opus, haiku, or a full model ID")
    parser.add_argument("--interval", type=int, default=8, metavar="SECS",
                        help="How often to process accumulated audio in conversation mode (default: 8)")
    args = parser.parse_args()

    model = resolve_model(args.model)
    print(f"Model: {model}", flush=True)

    try:
        if args.text:
            result = run(args.text.strip(), model=model)
            _print_result(result, args.notes)

        elif args.once:
            result = run_from_mic(model=model)
            _print_result(result, args.notes)

        else:
            run_conversation(interval_seconds=args.interval, model=model)

    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(0)


def _print_result(result, show_notes: bool) -> None:
    print("\n" + "-" * 60)
    print(result.english_text)
    if show_notes and result.translator_notes:
        print("\n-- Translator's Notes --")
        for note in result.translator_notes:
            print(f"  * {note}")
    print("-" * 60)


if __name__ == "__main__":
    main()
