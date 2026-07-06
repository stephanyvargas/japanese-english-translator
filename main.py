#!/usr/bin/env python3
"""
Spoken-language-to-English translator (CLI).

NOTE: the web app (server.py + frontend/) is the meeting surface with the full
quality stack (VAD segmentation, diarization, glossary, cached rich context,
self-repair). CLI conversation mode below is a simpler fixed-interval mode.

Modes:
  python3 main.py                           # basic conversation mode (default): mic always on,
                                            #   translates every 8s with rolling context
  python3 main.py --interval 12             # adjust processing interval (seconds)
  python3 main.py --once                    # single utterance, full quality pipeline, then exit
  python3 main.py --text "テキスト"           # translate typed text, full quality pipeline, exit
  python3 main.py [--once|--text] --notes   # include translator's notes

Source language (default: ja):
  python3 main.py --source-lang ko          # Korean
  python3 main.py --source-lang zh          # Chinese
  python3 main.py --source-lang es          # Spanish

Model (default: sonnet):
  python3 main.py --model sonnet            # claude-sonnet-4-6  — balanced cost/quality
  python3 main.py --model opus              # claude-opus-4-8    — highest quality
  python3 main.py --model haiku             # claude-haiku-4-5   — fastest, cheapest
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

LANGUAGE_NAMES = {
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def resolve_lang_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code.upper())


def main() -> None:
    parser = argparse.ArgumentParser(description="Spoken-language-to-English translator")
    parser.add_argument("--text", "-t", metavar="TEXT", help="Translate typed text and exit")
    parser.add_argument("--once", "-1", action="store_true",
                        help="Single mic utterance with full quality pipeline, then exit")
    parser.add_argument("--notes", "-n", action="store_true",
                        help="Print translator's notes (--once and --text only)")
    parser.add_argument("--model", "-m", default="sonnet", metavar="MODEL",
                        help="Model: sonnet (default), opus, haiku, or a full model ID")
    parser.add_argument("--source-lang", "-l", default="ja", metavar="LANG",
                        help="Source language ISO code, e.g. ja, ko, zh, es, fr (default: ja)")
    parser.add_argument("--context", "-c", default="", metavar="TEXT",
                        help="Optional setting description to guide translation "
                             "(e.g. 'business meeting', 'medical consultation', 'bank appointment')")
    parser.add_argument("--interval", type=int, default=8, metavar="SECS",
                        help="Processing interval in conversation mode (default: 8)")
    args = parser.parse_args()

    model = resolve_model(args.model)
    source_lang = args.source_lang.lower()
    lang_name = resolve_lang_name(source_lang)

    context = args.context.strip()
    context_label = f" | Context: {context}" if context else ""
    print(f"Model: {model} | Language: {lang_name} -> English{context_label}", flush=True)

    try:
        if args.text:
            result = run(args.text.strip(), model=model, source_lang=source_lang,
                         lang_name=lang_name, context=context)
            _print_result(result, args.notes)

        elif args.once:
            result = run_from_mic(model=model, source_lang=source_lang, lang_name=lang_name,
                                  context=context)
            _print_result(result, args.notes)

        else:
            run_conversation(
                interval_seconds=args.interval,
                model=model,
                source_lang=source_lang,
                lang_name=lang_name,
                context=context,
            )

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
