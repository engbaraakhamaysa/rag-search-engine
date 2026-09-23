import json
import pickle
import string
from collections import Counter
from pathlib import Path

from text_utils import tokenize_text


class InvertedIndex:
    def __init__(self):
        # Map each token to a set of document IDs
        self.index = {}

        # Map each document ID to the full document object
        self.docmap = {}

        # Map each document ID to its term frequency counter
        self.term_frequencies = {}

        # Load and normalize stop words
        with open("data/stopwords.txt") as f:
            stopwords = f.read().splitlines()

        translator = str.maketrans("", "", string.punctuation)

        self.stopwords = [
            word.lower().translate(translator)
            for word in stopwords
        ]

    def __add_document(self, doc_id, text):
        # Tokenize the document text
        tokens = tokenize_text(text, self.stopwords)

        # Create a counter for this document
        self.term_frequencies[doc_id] = Counter()

        # Add the document ID to the index for each token
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

            # Increment the term frequency for this token
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        # Get document IDs for a token
        doc_ids = self.index.get(term, set())

        # Return document IDs sorted in ascending order
        return sorted(doc_ids)

    def get_tf(self, doc_id, term):
        # Get the term frequency counter for the document
        frequencies = self.term_frequencies.get(doc_id, Counter())

        # Return the frequency of the term, or 0 if it does not exist
        return frequencies.get(term, 0)

    def build(self):
        # Load all movies from the dataset
        movies = load_movies()

        # Add every movie to the document map and inverted index
        for movie in movies:
            doc_id = movie["id"]

            # Store the full movie object
            self.docmap[doc_id] = movie

            # Combine the title and description for indexing
            text = f"{movie['title']} {movie['description']}"

            # Add the movie to the inverted index
            self.__add_document(doc_id, text)

    def save(self):
        # Create the cache directory if it does not exist
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        # Save the inverted index to disk
        with open(cache_dir / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        # Save the document map to disk
        with open(cache_dir / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        # Save the term frequencies to disk
        with open(cache_dir / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        # Load the inverted index from disk
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)

        # Load the document map from disk
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        # Load the term frequencies from disk
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)


def load_movies():
    # Open the movie dataset
    with open("data/movies.json") as f:
        # Load the JSON data
        data = json.load(f)

    # Return the list of movies
    return data["movies"]