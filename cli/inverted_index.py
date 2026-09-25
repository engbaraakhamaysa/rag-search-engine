import json
import math
import pickle
import string
from collections import Counter
from pathlib import Path

from text_utils import tokenize_text


BM25_K1 = 1.5
BM25_B = 0.75


class InvertedIndex:
    def __init__(self):
        # Map each token to a set of document IDs
        self.index = {}

        # Map each document ID to the full document object
        self.docmap = {}

        # Map each document ID to its term frequency counter
        self.term_frequencies = {}

        # Map each document ID to the number of tokens in the document
        self.doc_lengths = {}

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

        # Store the total number of tokens in the document
        self.doc_lengths[doc_id] = len(tokens)

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

    def get_bm25_idf(self, term: str) -> float:
        # Get the total number of documents
        total_doc_count = len(self.docmap)

        # Get the number of documents containing the term
        term_match_doc_count = len(self.get_documents(term))

        # Calculate BM25 IDF
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def __get_avg_doc_length(self) -> float:
        # Return 0.0 when there are no documents
        if not self.doc_lengths:
            return 0.0

        # Calculate the average document length
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        # Get the raw term frequency
        tf = self.get_tf(doc_id, term)

        # Get the document length
        doc_length = self.doc_lengths.get(doc_id, 0)

        # Get the average document length
        avg_doc_length = self.__get_avg_doc_length()

        # Calculate the document length normalization
        if avg_doc_length == 0:
            length_norm = 1
        else:
            length_norm = 1 - b + b * (
                doc_length / avg_doc_length
            )

        # Apply BM25 term frequency saturation and length normalization
        return (tf * (k1 + 1)) / (
            tf + k1 * length_norm
        )

    def bm25(self, doc_id, term):
        # Calculate the BM25 term frequency component
        bm25_tf = self.get_bm25_tf(doc_id, term)

        # Calculate the BM25 inverse document frequency component
        bm25_idf = self.get_bm25_idf(term)

        # Multiply TF and IDF to get the full BM25 score
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        # Tokenize the query
        query_tokens = tokenize_text(query, self.stopwords)

        # Map document IDs to their total BM25 scores
        scores = {}

        # Calculate BM25 scores for every document
        for doc_id in self.docmap:
            total_score = 0.0

            # Add the BM25 score for each query token
            for token in query_tokens:
                total_score += self.bm25(doc_id, token)

            scores[doc_id] = total_score

        # Sort documents by score from highest to lowest
        sorted_scores = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # Return the top documents with their scores
        return sorted_scores[:limit]

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

        # Save the document lengths to disk
        with open(cache_dir / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

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

        # Load the document lengths from disk
        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)


def load_movies():
    # Open the movie dataset
    with open("data/movies.json") as f:
        # Load the JSON data
        data = json.load(f)

    # Return the list of movies
    return data["movies"]