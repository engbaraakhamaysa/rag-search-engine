import argparse

from lib.multimodal_search import (
    image_search_command,
    verify_image_embedding,
)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate and verify an image embedding",
    )

    verify_parser.add_argument(
        "image_path",
        help="Path to the image",
    )

    image_search_parser = subparsers.add_parser(
        "image_search",
        help="Search movies using an image",
    )

    image_search_parser.add_argument(
        "image_path",
        help="Path to the image",
    )

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)

    elif args.command == "image_search":
        results = image_search_command(args.image_path)

        for index, result in enumerate(results, start=1):
            print(
                f"{index}. {result['title']} "
                f"(similarity: {result['similarity']:.3f})"
            )
            print(f"   {result['description']}")
            print()


if __name__ == "__main__":
    main()