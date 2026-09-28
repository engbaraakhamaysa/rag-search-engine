import argparse
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

from lib.hybrid_search import HybridSearch


load_dotenv()


def enhance_query(
    query: str,
    method: str | None,
) -> str:
    if not method:
        return query

    if method == "spell":
        prompt = f"""Correct any spelling mistakes in the movie search query below.

User query: "{query}"

Return only the corrected query.

Do not add explanations.
"""

    elif method == "rewrite":
        prompt = f"""Rewrite the movie search query below to make it clearer
and more effective for semantic and keyword search.

Preserve the user's original intent.

Return only the rewritten query.

Do not add explanations.

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

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty response from LLM")

            content = content.strip()

            if method == "expand":
                return f"{query} {content}"

            return content

        except Exception as error:
            print(
                f"Query enhancement attempt "
                f"{attempt + 1}/3 failed: {error}"
            )

            if attempt < 2:
                time.sleep(3)

    print("Using original query.")
    return query


def rerank_individual(
    query: str,
    results: list[dict],
) -> list[dict]:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=30.0,
    )

    for index, result in enumerate(results, start=1):
        print(
            f"Scoring result {index}/{len(results)}: "
            f"{result['title']}"
        )

        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {result.get("title", "")} - {result.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

        score = 0.0

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                content = response.choices[0].message.content

                if not content:
                    raise ValueError("Empty response from LLM")

                score = float(content.strip())

                if not 0 <= score <= 10:
                    raise ValueError(
                        "Score must be between 0 and 10"
                    )

                break

            except Exception as error:
                print(
                    f"  Attempt {attempt + 1}/3 failed: "
                    f"{error}"
                )

                if attempt < 2:
                    time.sleep(3)

        result["rerank_score"] = score

        time.sleep(3)

    results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return results


def rerank_batch(
    query: str,
    results: list[dict],
) -> list[dict]:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )

    doc_list = []

    for result in results:
        doc_list.append(
            f"ID: {result['id']}\n"
            f"Title: {result['title']}\n"
            f"Description: {result['description']}"
        )

    doc_list_str = "\n\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown.
Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty response from LLM")

            content = content.strip()

            ranked_ids = json.loads(content)

            if not isinstance(ranked_ids, list):
                raise ValueError(
                    "LLM response is not a JSON array"
                )

            if not all(
                isinstance(movie_id, int)
                for movie_id in ranked_ids
            ):
                raise ValueError(
                    "All ranked movie IDs must be integers"
                )

            result_ids = {
                result["id"]
                for result in results
            }

            if set(ranked_ids) != result_ids:
                raise ValueError(
                    "LLM response does not contain "
                    "exactly the provided movie IDs"
                )

            if len(ranked_ids) != len(results):
                raise ValueError(
                    "LLM response contains duplicate movie IDs"
                )

            rank_by_id = {
                movie_id: rank
                for rank, movie_id in enumerate(
                    ranked_ids,
                    start=1,
                )
            }

            for result in results:
                result["rerank_rank"] = rank_by_id[
                    result["id"]
                ]

            results.sort(
                key=lambda result: result["rerank_rank"]
            )

            return results

        except Exception as error:
            print(
                f"Batch re-ranking attempt "
                f"{attempt + 1}/3 failed: {error}"
            )

            if attempt < 2:
                time.sleep(3)

    raise RuntimeError(
        "Batch re-ranking failed after 3 attempts."
    )


def rerank_cross_encoder(
    query: str,
    results: list[dict],
) -> list[dict]:
    pairs = []

    for result in results:
        pairs.append(
            [
                query,
                f"{result.get('title', '')} - "
                f"{result.get('description', '')}",
            ]
        )

    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L2-v2"
    )

    scores = cross_encoder.predict(pairs)

    for result, score in zip(results, scores):
        result["cross_encoder_score"] = float(score)

    results.sort(
        key=lambda result: result["cross_encoder_score"],
        reverse=True,
    )

    return results


def evaluate_results(
    query: str,
    results: list[dict],
) -> list[dict]:
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )

    formatted_results = []

    for result in results:
        formatted_results.append(
            f"Title: {result['title']}\n"
            f"Description: {result['description']}"
        )

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents.
Return a valid JSON list, nothing else.

For example:
[2, 0, 3, 2, 0, 1]
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty response from LLM")

            scores = json.loads(content.strip())

            if not isinstance(scores, list):
                raise ValueError(
                    "LLM response is not a JSON list"
                )

            if len(scores) != len(results):
                raise ValueError(
                    "Number of scores does not match "
                    "number of results"
                )

            if not all(
                isinstance(score, int) and 0 <= score <= 3
                for score in scores
            ):
                raise ValueError(
                    "All scores must be integers from 0 to 3"
                )

            for result, score in zip(results, scores):
                result["evaluation_score"] = score

            return results

        except Exception as error:
            print(
                f"Evaluation attempt "
                f"{attempt + 1}/3 failed: {error}"
            )

            if attempt < 2:
                time.sleep(3)

    raise RuntimeError(
        "LLM evaluation failed after 3 attempts."
    )


def print_results(
    query: str,
    k: int,
    results: list[dict],
    rerank_method: str | None,
    limit: int,
) -> None:
    print(
        f"\nReciprocal Rank Fusion Results "
        f"for '{query}' (k={k}):\n"
    )

    for index, result in enumerate(
        results[:limit],
        start=1,
    ):
        print(
            f"{index}. {result['title']}"
        )

        if rerank_method == "individual":
            print(
                f"   Re-rank Score: "
                f"{result['rerank_score']:.3f}/10"
            )

        elif rerank_method == "batch":
            print(
                f"   Re-rank Rank: "
                f"{result['rerank_rank']}"
            )

        elif rerank_method == "cross_encoder":
            print(
                f"   Cross Encoder Score: "
                f"{result['cross_encoder_score']:.3f}"
            )

        print(
            f"   RRF Score: "
            f"{result['rrf_score']:.3f}"
        )

        print(
            f"   BM25 Rank: "
            f"{result['bm25_rank']}, "
            f"Semantic Rank: "
            f"{result['semantic_rank']}"
        )

        print(
            f"   {result['description'][:200]}..."
        )

        print()


def print_evaluation(
    results: list[dict],
) -> None:
    print()

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{index}. {result['title']}: "
            f"{result['evaluation_score']}/3"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid movie search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Run Reciprocal Rank Fusion search",
    )

    rrf_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    rrf_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="RRF k parameter",
    )

    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    rrf_parser.add_argument(
        "--enhance-method",
        choices=[
            "spell",
            "rewrite",
            "expand",
        ],
        default=None,
        help="LLM query enhancement method",
    )

    rrf_parser.add_argument(
        "--rerank-method",
        choices=[
            "individual",
            "batch",
            "cross_encoder",
        ],
        default=None,
        help="Re-ranking method",
    )

    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate search results using an LLM",
    )

    args = parser.parse_args()

    if args.command == "rrf-search":
        movies = load_movies()

        search = HybridSearch(movies)

        enhanced_query = enhance_query(
            args.query,
            args.enhance_method,
        )

        if enhanced_query != args.query:
            print(
                f"\nOriginal query: {args.query}"
            )

            print(
                f"Enhanced query: {enhanced_query}\n"
            )

        search_limit = args.limit

        if args.rerank_method:
            search_limit = args.limit * 5

        results = search.rrf_search(
            enhanced_query,
            args.k,
            search_limit,
        )

        if args.rerank_method:
            print(
                f"Re-ranking top {args.limit} results "
                f"using {args.rerank_method} method...\n"
            )

        if args.rerank_method == "individual":
            results = rerank_individual(
                enhanced_query,
                results,
            )

        elif args.rerank_method == "batch":
            results = rerank_batch(
                enhanced_query,
                results,
            )

        elif args.rerank_method == "cross_encoder":
            results = rerank_cross_encoder(
                enhanced_query,
                results,
            )

        print_results(
            enhanced_query,
            args.k,
            results,
            args.rerank_method,
            args.limit,
        )

        if args.evaluate:
            evaluated_results = evaluate_results(
                enhanced_query,
                results[:args.limit],
            )

            print_evaluation(
                evaluated_results,
            )


def load_movies() -> list[dict]:
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value

    raise ValueError(
        "Could not find a list of movies in data/movies.json"
    )


if __name__ == "__main__":
    main()
