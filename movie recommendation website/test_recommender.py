"""
test_recommender.py
Automated verification tests for the Machine Learning Content-Based Recommender.
Runs with pure Python standard library (unittest).
"""

import unittest
from dataset_loader import load_movies_from_json
from ml_recommender import TFIDFRecommender, tokenize, build_ngrams


class TestMovieRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.movies = load_movies_from_json()
        cls.recommender = TFIDFRecommender(cls.movies)

    def test_dataset_loaded(self):
        """Verify the dataset contains movies with all required schema fields."""
        self.assertGreaterEqual(len(self.movies), 50)
        sample = self.movies[0]
        required_fields = ["id", "title", "genres", "overview", "director", "cast", "rating", "poster_path"]
        for field in required_fields:
            self.assertIn(field, sample)

    def test_tokenizer_and_stopwords(self):
        """Verify stopwords are filtered and tokens are cleaned properly."""
        tokens = tokenize("This is an Inception movie about a dream within a dream.")
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("an", tokens)
        self.assertIn("inception", tokens)
        self.assertIn("dream", tokens)

    def test_ngrams(self):
        """Verify bigrams generation."""
        tokens = ["christopher", "nolan", "director"]
        bigrams = build_ngrams(tokens, n=2)
        self.assertIn("christopher_nolan", bigrams)
        self.assertIn("nolan_director", bigrams)

    def test_tfidf_vectors_initialized(self):
        """Ensure TF-IDF vectors are computed for all movies."""
        self.assertEqual(len(self.recommender.tfidf_vectors), len(self.movies))
        self.assertGreater(len(self.recommender.idf), 100)

    def test_recommendation_relevance(self):
        """Verify Inception returns Christopher Nolan / Sci-Fi films among top recommendations."""
        recs = self.recommender.recommend_by_title("Inception", top_n=6)
        self.assertGreaterEqual(len(recs), 3)

        rec_titles = [r["movie"]["title"] for r in recs]
        print("\nRecommendations for Inception:")
        for idx, r in enumerate(recs, 1):
            print(f"  {idx}. {r['movie']['title']} ({r['match_percentage']}% Match) - Reasons: {r['reasons']}")

        # Ensure similarity score is normalized in [0, 1]
        for r in recs:
            self.assertTrue(0.0 <= r["similarity_score"] <= 1.0)
            self.assertTrue(0 <= r["match_percentage"] <= 100)

        # Expected top recommendations should include sci-fi or Nolan films
        nolan_or_scifi = any(
            t in rec_titles for t in ["Tenet", "Interstellar", "The Prestige", "The Matrix", "Blade Runner 2049"]
        )
        self.assertTrue(nolan_or_scifi, "Should recommend relevant Sci-Fi/Nolan films for Inception")

    def test_search_functionality(self):
        """Verify search finds exact and partial matches."""
        results = self.recommender.search("Godfather")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "The Godfather")

    def test_genre_filter(self):
        """Verify filtering by genre returns only movies of that genre."""
        sci_fi_movies = self.recommender.get_by_genre("Sci-Fi")
        self.assertGreater(len(sci_fi_movies), 0)
        for m in sci_fi_movies:
            genres_lower = [g.lower() for g in m.get("genres", [])]
            self.assertIn("sci-fi", genres_lower)


if __name__ == "__main__":
    unittest.main()
