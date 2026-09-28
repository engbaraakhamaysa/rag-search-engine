import argparse
import base64
import mimetypes
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a movie search query using an image"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Text query to rewrite",
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as file:
        img = file.read()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    system_prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""

    data_url = (
        f"data:{mime};base64,"
        f"{base64.b64encode(img).decode()}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt.strip(),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    },
                },
                {
                    "type": "text",
                    "text": args.query.strip(),
                },
            ],
        }
    ]

    response = None

    for attempt in range(3):
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
        )

        if response.choices:
            break

        if attempt < 2:
            time.sleep(2)

    if response is None or not response.choices:
        raise RuntimeError(
            f"LLM returned no choices. Response: {response}"
        )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            f"LLM returned empty content. Response: {response}"
        )

    print(f"Rewritten query: {content.strip()}")

    if response.usage is not None:
        print(
            f"Total tokens:    {response.usage.total_tokens}"
        )


if __name__ == "__main__":
    main()