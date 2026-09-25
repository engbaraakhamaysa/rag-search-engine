import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text cannot be empty")

        embeddings = self.model.encode([text])

        return embeddings[0]

    def build_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]

        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
        )

        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = "cache/movie_embeddings.npy"

        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        results = []

        for i, document_embedding in enumerate(self.embeddings):
            score = cosine_similarity(
                query_embedding,
                document_embedding,
            )

            document = self.documents[i]

            results.append(
                (
                    score,
                    document,
                )
            )

        results.sort(
            key=lambda result: result[0],
            reverse=True,
        )

        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in results[:limit]
        ]


def load_movies():
    with open("data/movies.json") as f:
        data = json.load(f)

    return data["movies"]


def verify_model():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    result = []

    for i in range(len(vec1)):
        result.append(vec1[i] + vec2[i])

    return result


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    result = []

    for i in range(len(vec1)):
        result.append(vec1[i] - vec2[i])

    return result


def dot(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    total = 0.0

    for i in range(len(vec1)):
        total += vec1[i] * vec2[i]

    return total


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    movies = load_movies()

    search = SemanticSearch()
    embeddings = search.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: "
        f"{embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")