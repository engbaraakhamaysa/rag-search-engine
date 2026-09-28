import argparse
import json

from hybrid_search_cli import load_movies
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r") as file:
        golden_dataset = json.load(file)

    movies = load_movies()
    search = HybridSearch(movies)

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        results = search.rrf_search(
            query,
            60,
            limit,
        )

        retrieved = [result["title"] for result in results]

        relevant_retrieved = [
            title for title in retrieved
            if title in relevant_docs
        ]

        precision = len(relevant_retrieved) / len(retrieved)
        recall = len(relevant_retrieved) / len(relevant_docs)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        print(f"Query: {query}")
        print(f"Precision@{limit}: {precision:.4f}")
        print(f"Recall@{limit}: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Retrieved: {', '.join(retrieved)}")
        print(f"Relevant: {', '.join(relevant_docs)}")
        print()


if __name__ == "__main__":
    main()
