import json

from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)

        self.texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]

        self.text_embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
        )

    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)

        similarities = cosine_similarity(
            [image_embedding],
            self.text_embeddings,
        )[0]

        results = []

        for index, similarity in enumerate(similarities):
            document = self.documents[index]

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "description": document["description"],
                    "similarity": float(similarity),
                }
            )

        results.sort(
            key=lambda result: result["similarity"],
            reverse=True,
        )

        return results[:5]


def load_movies():
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


def verify_image_embedding(image_path):
    search = MultimodalSearch([])

    image = Image.open(image_path)
    embedding = search.model.encode([image])[0]

    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    movies = load_movies()
    search = MultimodalSearch(movies)

    return search.search_with_image(image_path)