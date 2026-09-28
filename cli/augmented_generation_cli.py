import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from lib.hybrid_search import HybridSearch


load_dotenv()


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


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )


def generate_rag_response(
    query: str,
    results: list[dict],
) -> str:
    client = create_client()

    documents = "\n\n".join(
        f"Title: {result['title']}\n"
        f"Description: {result['description']}"
        for result in results
    )

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{documents}

Answer:"""

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

    return content.strip()


def generate_summary(
    query: str,
    results: list[dict],
) -> str:
    client = create_client()

    documents = "\n\n".join(
        f"Title: {result['title']}\n"
        f"Description: {result['description']}"
        for result in results
    )

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{documents}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

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

    return content.strip()


def generate_citation_answer(
    query: str,
    results: list[dict],
) -> str:
    client = create_client()

    documents = "\n\n".join(
        f"[{index}] Title: {result['title']}\n"
        f"Description: {result['description']}"
        for index, result in enumerate(results, start=1)
    )

    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

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

    return content.strip()


def generate_question_answer(
    question: str,
    results: list[dict],
) -> str:
    client = create_client()

    context = "\n\n".join(
        f"Title: {result['title']}\n"
        f"Description: {result['description']}"
        for result in results
    )

    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

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

    return content.strip()


def print_search_results(
    results: list[dict],
) -> None:
    print("Search Results:")

    for result in results:
        print(f"  - {result['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # RAG command
    rag_parser = subparsers.add_parser(
        "rag",
        help="Perform RAG (search + generate answer)",
    )

    rag_parser.add_argument(
        "query",
        type=str,
        help="Search query for RAG",
    )

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize search results",
    )

    summarize_parser.add_argument(
        "query",
        type=str,
        help="Search query for summarization",
    )

    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize",
    )

    # Citations command
    citations_parser = subparsers.add_parser(
        "citations",
        help="Answer a query with source citations",
    )

    citations_parser.add_argument(
        "query",
        type=str,
        help="Search query for citation-aware answer",
    )

    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use",
    )

    # Question command
    question_parser = subparsers.add_parser(
        "question",
        help="Answer a question using search results",
    )

    question_parser.add_argument(
        "question",
        type=str,
        help="Question to answer",
    )

    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            movies = load_movies()
            search = HybridSearch(movies)

            results = search.rrf_search(
                query,
                60,
                5,
            )

            response = generate_rag_response(
                query,
                results,
            )

            print_search_results(results)

            print()
            print("RAG Response:")
            print(response)

        case "summarize":
            query = args.query

            movies = load_movies()
            search = HybridSearch(movies)

            results = search.rrf_search(
                query,
                60,
                args.limit,
            )

            summary = generate_summary(
                query,
                results,
            )

            print_search_results(results)

            print()
            print("LLM Summary:")
            print(summary)

        case "citations":
            query = args.query

            movies = load_movies()
            search = HybridSearch(movies)

            results = search.rrf_search(
                query,
                60,
                args.limit,
            )

            answer = generate_citation_answer(
                query,
                results,
            )

            print_search_results(results)

            print()
            print("LLM Answer:")
            print(answer)

        case "question":
            question = args.question

            movies = load_movies()
            search = HybridSearch(movies)

            results = search.rrf_search(
                question,
                60,
                args.limit,
            )

            answer = generate_question_answer(
                question,
                results,
            )

            print_search_results(results)

            print()
            print("Answer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()