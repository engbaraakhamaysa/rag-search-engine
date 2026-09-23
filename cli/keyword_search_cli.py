import argparse
import math
import string

from inverted_index import InvertedIndex
from text_utils import tokenize_term, tokenize_text


def build_command():
    # Create and build the inverted index
    index = InvertedIndex()
    index.build()

    # Save the index and document map to disk
    index.save()


def main() -> None:
    # Create the main argument parser for the CLI
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    # Create subcommands such as "search", "build", "tf", "idf", and "tfidf"
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # Create the "search" command
    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords",
    )

    # Add the search query as a required argument
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

    # Add the document ID argument
    tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID",
    )

    # Add the term argument
    tf_parser.add_argument(
        "term",
        type=str,
        help="Search term",
    )

    # Create the "idf" command
    idf_parser = subparsers.add_parser(
        "idf",
        help="Get the inverse document frequency of a term",
    )

    # Add the term argument
    idf_parser.add_argument(
        "term",
        type=str,
        help="Search term",
    )

    # Create the "tfidf" command
    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Get the TF-IDF score for a document and term",
    )

    # Add the document ID argument
    tfidf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID",
    )

    # Add the term argument
    tfidf_parser.add_argument(
        "term",
        type=str,
        help="Search term",
    )

    # Parse the arguments provided by the user
    args = parser.parse_args()

    match args.command:
        case "search":
            # Create an inverted index
            index = InvertedIndex()

            # Load the index and document map from disk
            index.load()

            # Load the stop words
            with open("data/stopwords.txt") as f:
                stopwords = f.read().splitlines()

            # Normalize the stop words
            translator = str.maketrans("", "", string.punctuation)
            stopwords = [
                word.lower().translate(translator)
                for word in stopwords
            ]

            # Tokenize the search query
            query_tokens = tokenize_text(args.query, stopwords)

            # Store matching document IDs
            results = []

            # Search for each query token
            for token in query_tokens:
                doc_ids = index.get_documents(token)

                for doc_id in doc_ids:
                    if doc_id not in results:
                        results.append(doc_id)

                    # Stop once we have 5 results
                    if len(results) == 5:
                        break

                if len(results) == 5:
                    break

            # Print the search query
            print(f"Searching for: {args.query}")

            # Print each matching movie with its ID and title
            for position, doc_id in enumerate(results, start=1):
                movie = index.docmap[doc_id]
                print(f"{position}. {movie['title']} (ID: {doc_id})")

        case "build":
            build_command()

        case "tf":
            # Create an inverted index
            index = InvertedIndex()

            # Load the index, document map, and term frequencies
            index.load()

            # Tokenize the requested term
            term = tokenize_term(args.term, index.stopwords)

            # Get and print the term frequency
            print(index.get_tf(args.doc_id, term))

        case "idf":
            # Create an inverted index
            index = InvertedIndex()

            # Load the index and document map from disk
            index.load()

            # Tokenize the requested term
            term = tokenize_term(args.term, index.stopwords)

            # Get the number of documents containing the term
            term_match_doc_count = len(index.get_documents(term))

            # Get the total number of documents
            total_doc_count = len(index.docmap)

            # Calculate inverse document frequency
            idf = math.log(
                (total_doc_count + 1)
                / (term_match_doc_count + 1)
            )

            # Print the IDF value rounded to 2 decimal places
            print(
                f"Inverse document frequency of '{args.term}': {idf:.2f}"
            )

        case "tfidf":
            # Create an inverted index
            index = InvertedIndex()

            # Load the index, document map, and term frequencies
            index.load()

            # Tokenize the requested term
            term = tokenize_term(args.term, index.stopwords)

            # Calculate term frequency
            tf = index.get_tf(args.doc_id, term)

            # Get the number of documents containing the term
            term_match_doc_count = len(index.get_documents(term))

            # Get the total number of documents
            total_doc_count = len(index.docmap)

            # Calculate inverse document frequency
            idf = math.log(
                (total_doc_count + 1)
                / (term_match_doc_count + 1)
            )

            # Calculate TF-IDF
            tf_idf = tf * idf

            # Print the TF-IDF score rounded to 2 decimal places
            print(
                f"TF-IDF score of '{args.term}' "
                f"in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case _:
            # Show the help message if no valid command was provided
            parser.print_help()


if __name__ == "__main__":
    main()