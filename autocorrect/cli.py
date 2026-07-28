"""Interactive command-line interface for the autocorrect tool."""

from __future__ import annotations

import argparse
import sys

from .corrector import Autocorrect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autocorrect",
        description="Suggest spelling corrections trained on a Shakespearean corpus.",
    )
    parser.add_argument(
        "word", nargs="?", help="Word to correct. Omit to start an interactive session."
    )
    parser.add_argument(
        "-n",
        "--num-suggestions",
        type=int,
        default=5,
        help="Number of ranked suggestions to show (default: 5)",
    )
    return parser


def print_suggestions(model: Autocorrect, word: str, limit: int) -> None:
    suggestions = model.suggest(word, limit=limit)
    if not suggestions:
        print(f"No suggestions found for {word!r}.")
        return

    print(f"\nSuggestions for {word!r}:")
    for rank, s in enumerate(suggestions, start=1):
        marker = " (already correct)" if s.edit_distance == 0 else ""
        print(
            f"  {rank}. {s.word:<15} edit distance={s.edit_distance}  "
            f"P(word)={s.probability:.6f}{marker}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("Loading Shakespearean corpus...", file=sys.stderr)
    model = Autocorrect.from_corpus()
    print(f"Loaded {len(model.word_frequency)} unique words.", file=sys.stderr)

    if args.word:
        print_suggestions(model, args.word, args.num_suggestions)
        return 0

    print("Type a word to get corrections. Ctrl+C or 'quit' to exit.\n")
    try:
        while True:
            word = input("> ").strip()
            if not word:
                continue
            if word.lower() in {"quit", "exit"}:
                break
            print_suggestions(model, word, args.num_suggestions)
    except (KeyboardInterrupt, EOFError):
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
