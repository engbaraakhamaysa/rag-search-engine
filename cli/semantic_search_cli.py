import argparse

from lib.semantic_search import verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    parser.add_argument(
        "command",
        choices=["verify"],
        help="Available commands",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()