import argparse

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    load_movies,
    semantic_chunk_text,
    verify_embeddings,
    verify_model,
)


def chunk_text(text, chunk_size, overlap=0):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])

        if chunk:
            chunks.append(chunk)

    return chunks


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
            "search_chunked",
            "chunk",
            "semantic_chunk",
            "embed_chunks",
        ],
        help="Available commands",
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to embed, search, or chunk",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of search results",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words or sentences shared between chunks",
    )

    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per semantic chunk",
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

        case "search_chunked":
            movies = load_movies()

            search = ChunkedSemanticSearch()

            search.load_or_create_chunk_embeddings(
                movies
            )

            results = search.search_chunks(
                args.text,
                args.limit,
            )

            for i, result in enumerate(results, start=1):
                print(
                    f"\n{i}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(
                    f"   {result['document']}..."
                )

        case "chunk":
            chunks = chunk_text(
                args.text,
                args.chunk_size,
                args.overlap,
            )

            print(f"Chunking {len(args.text)} characters")

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            chunks = semantic_chunk_text(
                args.text,
                args.max_chunk_size,
                args.overlap,
            )

            print(
                f"Semantically chunking {len(args.text)} characters"
            )

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            movies = load_movies()

            search = ChunkedSemanticSearch()

            embeddings = search.load_or_create_chunk_embeddings(
                movies
            )

            print(
                f"Generated {len(embeddings)} chunked embeddings"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()