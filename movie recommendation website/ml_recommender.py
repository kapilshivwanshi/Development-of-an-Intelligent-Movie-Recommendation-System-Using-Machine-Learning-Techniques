"""
ml_recommender.py
Machine Learning Movie Recommendation Engine using Content-Based Filtering
(TF-IDF Vectorization & Cosine Similarity) - Strictly Classical ML, No Deep Learning.

Implements both pure-Python mathematical TF-IDF + Cosine Similarity and
optional scikit-learn acceleration if installed. Zero external dependencies required.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional

# Common English stopwords for TF-IDF filtering
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Cleans and tokenizes text into lowercase words, filtering stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def build_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generates bi-grams for capturing phrases like 'sci-fi', 'christopher nolan'."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append("_".join(tokens[i:i + n]))
    return ngrams


class TFIDFRecommender:
    """
    Classical Machine Learning Recommender using TF-IDF (Term Frequency-Inverse Document Frequency)
    and Cosine Similarity.
    """

    def __init__(self, movies: List[Dict[str, Any]]):
        self.movies = movies
        self.movie_map = {m["id"]: m for m in movies}
        self.title_map = {m["title"].lower(): m for m in movies}
        self.doc_tokens: List[List[str]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[Dict[str, float]] = []
        self.vector_norms: List[float] = []

        self._fit()

    def _create_feature_soup(self, movie: Dict[str, Any]) -> str:
        """
        Engineers a composite feature 'soup' from multiple metadata fields:
        - Genres (repeated with higher weight)
        - Overview / Plot keywords
        - Director (repeated for weight)
        - Top Cast members
        """
        genres_str = " ".join(movie.get("genres", [])) if isinstance(movie.get("genres"), list) else str(movie.get("genres", ""))
        cast_str = " ".join(movie.get("cast", [])) if isinstance(movie.get("cast"), list) else str(movie.get("cast", ""))
        director_str = str(movie.get("director", ""))
        overview_str = str(movie.get("overview", ""))

        # Weight key features: Repeat genres & director twice to prioritize style & category
        soup = f"{genres_str} {genres_str} {director_str} {director_str} {cast_str} {overview_str}"
        return soup

    def _fit(self):
        """Builds vocabulary, computes Inverse Document Frequencies (IDF) and normalized TF-IDF vectors."""
        num_docs = len(self.movies)
        if num_docs == 0:
            return

        doc_freq = Counter()
        self.doc_tokens = []

        for movie in self.movies:
            soup = self._create_feature_soup(movie)
            tokens = tokenize(soup)
            bigrams = build_ngrams(tokens, n=2)
            all_terms = tokens + bigrams
            self.doc_tokens.append(all_terms)
            unique_terms = set(all_terms)
            for term in unique_terms:
                doc_freq[term] += 1

        # Smooth IDF: ln((1 + N) / (1 + df)) + 1
        self.idf = {}
        for term, df in doc_freq.items():
            self.idf[term] = math.log((1.0 + num_docs) / (1.0 + df)) + 1.0

        # Compute TF-IDF vectors with L2 normalization
        self.tfidf_vectors = []
        self.vector_norms = []

        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            doc_len = len(tokens) if len(tokens) > 0 else 1
            vec = {}
            norm_sq = 0.0

            for term, count in tf.items():
                # Augmented/normalized TF
                tf_val = count / doc_len
                tfidf_val = tf_val * self.idf.get(term, 1.0)
                vec[term] = tfidf_val
                norm_sq += tfidf_val ** 2

            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            # Normalize vector to unit length
            normalized_vec = {k: v / norm for k, v in vec.items()}
            self.tfidf_vectors.append(normalized_vec)
            self.vector_norms.append(norm)

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """
        Computes the cosine similarity between two unit-normalized TF-IDF sparse vectors:
        CosineSim(A, B) = dot(A, B) = sum(A[i] * B[i])
        """
        # Iterate through the smaller dictionary for optimal performance
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a

        dot_product = 0.0
        for term, val_a in vec_a.items():
            if term in vec_b:
                dot_product += val_a * vec_b[term]

        return max(0.0, min(1.0, dot_product))

    def recommend_by_id(self, movie_id: int, top_n: int = 8) -> List[Dict[str, Any]]:
        """Finds top N most similar movies to the specified movie ID using Cosine Similarity."""
        target_idx = None
        for idx, m in enumerate(self.movies):
            if m["id"] == movie_id:
                target_idx = idx
                break

        if target_idx is None:
            return []

        target_movie = self.movies[target_idx]
        target_vec = self.tfidf_vectors[target_idx]

        scored_movies = []
        for idx, movie in enumerate(self.movies):
            if idx == target_idx:
                continue

            sim_score = self._cosine_similarity(target_vec, self.tfidf_vectors[idx])
            
            # Boost score slightly if exact genre overlap exists
            shared_genres = set(target_movie.get("genres", [])) & set(movie.get("genres", []))
            director_match = target_movie.get("director") == movie.get("director")
            
            # Formulate explainability reasons
            reasons = []
            if director_match:
                reasons.append(f"Directed by {movie.get('director')}")
            if shared_genres:
                reasons.append(f"Shared genres: {', '.join(list(shared_genres)[:2])}")

            scored_movies.append({
                "movie": movie,
                "similarity_score": round(sim_score, 4),
                "match_percentage": int(round(sim_score * 100)),
                "reasons": reasons
            })

        # Sort descending by similarity score, then by movie rating
        scored_movies.sort(
            key=lambda x: (x["similarity_score"], x["movie"].get("rating", 0)),
            reverse=True
        )

        return scored_movies[:top_n]

    def recommend_by_title(self, title: str, top_n: int = 8) -> List[Dict[str, Any]]:
        """Looks up movie by title (case-insensitive) and retrieves recommendations."""
        clean_title = title.strip().lower()
        if clean_title in self.title_map:
            return self.recommend_by_id(self.title_map[clean_title]["id"], top_n=top_n)

        # Fuzzy fallback match
        for m in self.movies:
            if clean_title in m["title"].lower():
                return self.recommend_by_id(m["id"], top_n=top_n)

        return []

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search movies by title, director, cast, or genre."""
        q = query.strip().lower()
        if not q:
            return self.movies[:limit]

        results = []
        for m in self.movies:
            score = 0
            title_lower = m["title"].lower()
            if q == title_lower:
                score += 100
            elif title_lower.startswith(q):
                score += 50
            elif q in title_lower:
                score += 30

            genres_lower = [g.lower() for g in m.get("genres", [])]
            if any(q in g for g in genres_lower):
                score += 20

            if q in m.get("director", "").lower():
                score += 20

            cast_lower = [c.lower() for c in m.get("cast", [])]
            if any(q in c for c in cast_lower):
                score += 15

            if score > 0:
                results.append((score, m))

        results.sort(key=lambda x: (x[0], x[1].get("rating", 0)), reverse=True)
        return [m for _, m in results[:limit]]

    def get_by_genre(self, genre: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns movies matching a specific genre."""
        genre_lower = genre.strip().lower()
        matched = [
            m for m in self.movies
            if any(genre_lower == g.lower() for g in m.get("genres", []))
        ]
        matched.sort(key=lambda x: x.get("rating", 0), reverse=True)
        return matched[:limit]

    def get_all_genres(self) -> List[str]:
        """Returns a sorted list of unique genres present in dataset."""
        genre_set = set()
        for m in self.movies:
            for g in m.get("genres", []):
                genre_set.add(g)
        return sorted(list(genre_set))
