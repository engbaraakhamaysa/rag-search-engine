import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from lib.hybrid_search import HybridSearch
from lib.semantic_search import load_movies


load_dotenv()


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


def enhance_query(query: str, method: str | None) -> str:
    if method is None:
        return query

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    if method == "spell":
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    elif method == "rewrite":
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    elif method == "expand":
        prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms, related concepts, and specific concepts that might appear in movie descriptions.
For movie searches, include relevant people, genres, themes, occupations, and concepts when useful.
Keep expansions relevant and focused.
Do not add unrelated genres or concepts.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "horror grizzly bear terrifying survival"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"
- "math movie" -> "mathematics mathematician genius equations numbers"

User query: "{query}"
"""

    else:
        return query

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    enhanced_query = response.choices[0].message.content

    if not enhanced_query:
        return query

    enhanced_query = enhanced_query.strip()

    if method == "expand":
        return f"{query} {enhanced_query}"

    return enhanced_query


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
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
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
            query = args.query

            enhanced_query = enhance_query(
                query,
                args.enhance,
            )

            if args.enhance and enhanced_query != query:
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )

            movies = load_movies()
            search = HybridSearch(movies)

            results = search.rrf_search(
                enhanced_query,
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