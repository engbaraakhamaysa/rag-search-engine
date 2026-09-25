import argparse

from lib.hybrid_search import HybridSearch
from lib.semantic_search import load_movies


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize scores to the range 0-1",
    )
    normalize_parser.add_argument(
        "scores",
        nargs="*",
        type=float,
        help="Scores to normalize",
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Run weighted hybrid search",
    )
    weighted_parser.add_argument(
        "query",
        help="Search query",
    )
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight given to keyword search",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Run Reciprocal Rank Fusion search",
    )
    rrf_parser.add_argument(
        "query",
        help="Search query",
    )
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF constant",
    )
    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)

            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            movies = load_movies()
            search = HybridSearch(movies)

            results = search.weighted_search(
                args.query,
                args.alpha,
                args.limit,
            )

            for i, result in enumerate(
                results[:args.limit],
                start=1,
            ):
                print(f"{i}. {result['title']}")
                print(
                    f"  Hybrid Score: {result['hybrid_score']:.3f}"
                )
                print(
                    f"  BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(
                    f"  {result['description'][:200]}"
                )

                if i < min(len(results), args.limit):
                    print()

        case "rrf-search":
            movies = load_movies()
            search = HybridSearch(movies)

            results = search.rrf_search(
                args.query,
                args.k,
                args.limit,
            )

            for i, result in enumerate(
                results[:args.limit],
                start=1,
            ):
                print(f"{i}. {result['title']}")
                print(
                    f"  RRF Score: {result['rrf_score']:.3f}"
                )
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(
                    f"  {result['description'][:200]}"
                )

                if i < min(len(results), args.limit):
                    print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()