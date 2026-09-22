import argparse
import json
import string

from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


def matches_query(query: str, title: str, stopwords: list[str]) -> bool:
    # Create a translation table that removes punctuation
    translator = str.maketrans("", "", string.punctuation)

    # Normalize the query and remove punctuation
    query = query.lower().translate(translator)

    # Normalize the title and remove punctuation
    title = title.lower().translate(translator)

    # Split the query into tokens and remove stop words
    query_tokens = [
        token for token in query.split()
        if token not in stopwords
    ]

    # Split the title into tokens and remove stop words
    title_tokens = [
        token for token in title.split()
        if token not in stopwords
    ]

    # Reduce each token to its stem
    query_tokens = [stemmer.stem(token) for token in query_tokens]
    title_tokens = [stemmer.stem(token) for token in title_tokens]

    # Check if any query token partially matches any title token
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True

    # No query token matched any title token
    return False


def main() -> None:
    # Create the main argument parser for the CLI
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    # Create subcommands such as "search"
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

    # Parse the arguments provided by the user
    args = parser.parse_args()

    match args.command:
        case "search":
            # Open the movie dataset
            with open("data/movies.json") as f:
                # Load the JSON data into a Python dictionary
                data = json.load(f)

            # Create a translation table that removes punctuation
            translator = str.maketrans("", "", string.punctuation)

            # Load the stop words from the file
            with open("data/stopwords.txt") as f:
                stopwords = f.read().splitlines()

            # Normalize the stop words using the same preprocessing
            stopwords = [
                word.lower().translate(translator)
                for word in stopwords
            ]

            # Create an empty list to store matching movies
            results = []

            # Search through all movies in the dataset
            for movie in data["movies"]:
                # Check if the movie title matches the search query
                if matches_query(args.query, movie["title"], stopwords):
                    # Add the matching movie to the results
                    results.append(movie)

            # Limit the results to a maximum of 5 movies
            results = results[:5]

            # Print the search query
            print(f"Searching for: {args.query}")

            # Print each matching movie with its position
            for index, movie in enumerate(results, start=1):
                print(f"{index}. {movie['title']}")

        case _:
            # Show the help message if no valid command was provided
            parser.print_help()


if __name__ == "__main__":
    main()