from inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: float = 0.5,
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, movies: list[dict]):
        self.movies = movies

        self.idx = InvertedIndex()

        self.semantic_search = ChunkedSemanticSearch()

        self.semantic_search.load_or_create_chunk_embeddings(
            self.movies
        )

    def _bm25_search(
        self,
        query: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self,
        query: str,
        alpha: float = 0.5,
        limit: int = 5,
    ) -> list[dict]:
        search_limit = limit * 500

        keyword_results = self._bm25_search(
            query,
            search_limit,
        )

        semantic_results = self.semantic_search.search_chunks(
            query,
            search_limit,
        )

        keyword_scores = normalize_scores(
            [score for _, score in keyword_results]
        )

        semantic_scores = normalize_scores(
            [result["score"] for result in semantic_results]
        )

        documents = {}

        for (document_id, _), score in zip(
            keyword_results,
            keyword_scores,
        ):
            documents[document_id] = {
                "bm25_score": score,
                "semantic_score": 0.0,
            }

        for result, score in zip(
            semantic_results,
            semantic_scores,
        ):
            document_id = result["id"]

            if document_id not in documents:
                documents[document_id] = {
                    "bm25_score": 0.0,
                    "semantic_score": score,
                }
            else:
                documents[document_id]["semantic_score"] = score

        movies_by_id = {
            movie["id"]: movie
            for movie in self.movies
        }

        results = []

        for document_id, scores in documents.items():
            movie = movies_by_id[document_id]

            bm25 = scores["bm25_score"]
            semantic = scores["semantic_score"]

            score = hybrid_score(
                bm25,
                semantic,
                alpha,
            )

            results.append(
                {
                    "id": document_id,
                    "title": movie["title"],
                    "description": movie["description"],
                    "bm25_score": bm25,
                    "semantic_score": semantic,
                    "hybrid_score": score,
                }
            )

        results.sort(
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )

        return results

    def rrf_search(
        self,
        query: str,
        k: int = 60,
        limit: int = 5,
    ) -> list[dict]:
        search_limit = limit * 500

        keyword_results = self._bm25_search(
            query,
            search_limit,
        )

        semantic_results = self.semantic_search.search_chunks(
            query,
            search_limit,
        )

        documents = {}

        for rank, (document_id, _) in enumerate(
            keyword_results,
            start=1,
        ):
            documents[document_id] = {
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):
            document_id = result["id"]

            semantic_rrf = rrf_score(rank, k)

            if document_id not in documents:
                documents[document_id] = {
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": semantic_rrf,
                }
            else:
                documents[document_id]["semantic_rank"] = rank
                documents[document_id]["rrf_score"] += semantic_rrf

        movies_by_id = {
            movie["id"]: movie
            for movie in self.movies
        }

        results = []

        for document_id, data in documents.items():
            movie = movies_by_id[document_id]

            results.append(
                {
                    "id": document_id,
                    "title": movie["title"],
                    "description": movie["description"],
                    "bm25_rank": data["bm25_rank"],
                    "semantic_rank": data["semantic_rank"],
                    "rrf_score": data["rrf_score"],
                }
            )

        results.sort(
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        return results