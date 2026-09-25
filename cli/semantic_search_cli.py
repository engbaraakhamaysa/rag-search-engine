import argparse

from lib.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    load_movies,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Search CLI"
    )

    parser.add_argument(
        "command",
        choices=[
            "verify",
            "embed_text",
            "verify_embeddings",
            "embed_query",
            "search",
        ],
        help="Available commands",
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to embed or search for",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of search results",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.text)

        case "search":
            search = SemanticSearch()

            movies = load_movies()
            search.load_or_create_embeddings(movies)

            results = search.search(
                args.text,
                args.limit,
            )

            for i, result in enumerate(results, start=1):
                print(
                    f"{i}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(f"  {result['description']}")

                if i < len(results):
                    print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()