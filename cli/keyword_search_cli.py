import argparse
import math
import string

from inverted_index import BM25_B, BM25_K1, InvertedIndex
from text_utils import tokenize_term, tokenize_text


def build_command():
    # Create and build the inverted index
    index = InvertedIndex()
    index.build()

    # Save the index and document map to disk
    index.save()


def bm25_idf_command(term: str) -> float:
    # Create an inverted index
    index = InvertedIndex()

    # Load the index and document map from disk
    index.load()

    # Tokenize the term
    token = tokenize_term(term, index.stopwords)

    # Calculate and return the BM25 IDF score
    return index.get_bm25_idf(token)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    # Create an inverted index
    index = InvertedIndex()

    # Load the index and document map from disk
    index.load()

    # Tokenize the term
    token = tokenize_term(term, index.stopwords)

    # Calculate and return the BM25 TF score
    return index.get_bm25_tf(doc_id, token, k1, b)


def main() -> None:
    # Create the main argument parser for the CLI
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    # Create subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # Create the "search" command
    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    # Create the "build" command
    subparsers.add_parser(
        "build",
        help="Build the inverted index",
    )

    # Create the "tf" command
    tf_parser = subparsers.add_parser(
        "tf",
        help="Get the term frequency for a document",
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    # Create the "idf" command
    idf_parser = subparsers.add_parser(
        "idf",
        help="Get the inverse document frequency of a term",
    )
    idf_parser.add_argument("term", type=str, help="Search term")

    # Create the "tfidf" command
    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Get the TF-IDF score for a document and term",
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    # Create the "bm25idf" command
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf",
        help="Get BM25 IDF score for a given term",
    )
    bm25_idf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 IDF score for",
    )

    # Create the "bm25tf" command
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 TF score for a given document ID and term",
    )
    bm25_tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID",
    )
    bm25_tf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 TF score for",
    )
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help="Tunable BM25 K1 parameter",
    )
    bm25_tf_parser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=BM25_B,
        help="Tunable BM25 b parameter",
    )

    # Create the "bm25search" command
    bm25_search_parser = subparsers.add_parser(
        "bm25search",
        help="Search movies using full BM25 scoring",
    )
    bm25_search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    bm25_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    # Parse args
    args = parser.parse_args()

    match args.command:
        case "search":
            index = InvertedIndex()
            index.load()

            with open("data/stopwords.txt") as f:
                stopwords = f.read().splitlines()

            translator = str.maketrans("", "", string.punctuation)

            stopwords = [
                word.lower().translate(translator)
                for word in stopwords
            ]

            query_tokens = tokenize_text(args.query, stopwords)

            results = []

            for token in query_tokens:
                doc_ids = index.get_documents(token)

                for doc_id in doc_ids:
                    if doc_id not in results:
                        results.append(doc_id)

                    if len(results) == 5:
                        break

                if len(results) == 5:
                    break

            print(f"Searching for: {args.query}")

            for position, doc_id in enumerate(results, start=1):
                movie = index.docmap[doc_id]
                print(f"{position}. {movie['title']} (ID: {doc_id})")

        case "build":
            build_command()

        case "tf":
            index = InvertedIndex()
            index.load()

            term = tokenize_term(args.term, index.stopwords)

            print(index.get_tf(args.doc_id, term))

        case "idf":
            index = InvertedIndex()
            index.load()

            term = tokenize_term(args.term, index.stopwords)

            term_match_doc_count = len(index.get_documents(term))
            total_doc_count = len(index.docmap)

            idf = math.log(
                (total_doc_count + 1)
                / (term_match_doc_count + 1)
            )

            print(
                f"Inverse document frequency of '{args.term}': {idf:.2f}"
            )

        case "tfidf":
            index = InvertedIndex()
            index.load()

            term = tokenize_term(args.term, index.stopwords)

            tf = index.get_tf(args.doc_id, term)

            term_match_doc_count = len(index.get_documents(term))
            total_doc_count = len(index.docmap)

            idf = math.log(
                (total_doc_count + 1)
                / (term_match_doc_count + 1)
            )

            tf_idf = tf * idf

            print(
                f"TF-IDF score of '{args.term}' "
                f"in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "bm25idf":
            # Calculate the BM25 IDF score
            bm25idf = bm25_idf_command(args.term)

            # Print the BM25 IDF score rounded to 2 decimal places
            print(
                f"BM25 IDF score of '{args.term}': {bm25idf:.2f}"
            )

        case "bm25tf":
            # Calculate the BM25 TF score
            bm25tf = bm25_tf_command(
                args.doc_id,
                args.term,
                args.k1,
                args.b,
            )

            # Print the BM25 TF score rounded to 2 decimal places
            print(
                f"BM25 TF score of '{args.term}' "
                f"in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            # Load the inverted index
            index = InvertedIndex()
            index.load()

            # Search using BM25
            results = index.bm25_search(
                args.query,
                args.limit,
            )

            # Print the search results
            for position, (doc_id, score) in enumerate(
                results,
                start=1,
            ):
                movie = index.docmap[doc_id]

                print(
                    f"{position}. ({doc_id}) "
                    f"{movie['title']} - Score: {score:.2f}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()