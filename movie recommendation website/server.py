"""
server.py
Zero-dependency HTTP Server & REST API for the Movie Recommendation System.
Powered entirely by Python's built-in standard library (http.server, json, urllib).
"""

import http.server
import socketserver
import os
import json
import urllib.parse
from dataset_loader import load_movies_from_json
from ml_recommender import TFIDFRecommender

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Initialize and train ML Recommender once at startup
print("[ML ENGINE] Loading movie dataset...")
MOVIES = load_movies_from_json()
print(f"[ML ENGINE] Initializing TF-IDF Vectorizer & Cosine Similarity on {len(MOVIES)} movies...")
RECOMMENDER = TFIDFRecommender(MOVIES)
print("[ML ENGINE] Model trained and ready for instant recommendations!")


class MovieAppRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handles static web assets and REST API requests for movie recommendations."""

    def end_headers(self):
        # Enable CORS for easy API testing and browser fetch
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Route: REST API
        if path.startswith("/api/"):
            self.handle_api(path, query_params)
            return

        # Route: Frontend Homepage
        if path == "/" or path == "/index.html":
            index_path = os.path.join(BASE_DIR, "index.html")
            if not os.path.exists(index_path):
                index_path = os.path.join(TEMPLATES_DIR, "index.html")
            if os.path.exists(index_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # Route: Static Assets (CSS, JS, Images)
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):].lstrip("/")
            file_path = os.path.join(STATIC_DIR, rel_path)
            if os.path.isfile(file_path):
                self.send_static_file(file_path)
                return

        # Fallback to standard handler
        super().do_GET()

    def send_static_file(self, file_path: str):
        """Sends static files with correct MIME types."""
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon"
        }
        content_type = mime_types.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data: Any, status_code: int = 200):
        """Helper to serialize and respond with JSON."""
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_api(self, path: str, params: dict):
        """Dispatches API endpoints."""
        try:
            if path == "/api/recommend":
                movie_id = params.get("id", [None])[0]
                title = params.get("title", [None])[0]
                top_n = int(params.get("top_n", [8])[0])

                if movie_id is not None:
                    try:
                        m_id = int(movie_id)
                    except ValueError:
                        self.send_json({"error": "Invalid movie id"}, 400)
                        return
                    selected_movie = RECOMMENDER.movie_map.get(m_id)
                    if not selected_movie:
                        self.send_json({"error": "Movie not found"}, 404)
                        return
                    recommendations = RECOMMENDER.recommend_by_id(m_id, top_n=top_n)
                    self.send_json({
                        "selected_movie": selected_movie,
                        "recommendations": recommendations,
                        "algorithm": "Content-Based Filtering (TF-IDF + Cosine Similarity)"
                    })
                    return

                elif title is not None:
                    recommendations = RECOMMENDER.recommend_by_title(title, top_n=top_n)
                    selected_movie = None
                    clean = title.strip().lower()
                    if clean in RECOMMENDER.title_map:
                        selected_movie = RECOMMENDER.title_map[clean]
                    elif recommendations:
                        # Found through fuzzy match
                        match_title = recommendations[0]["movie"]["title"]
                        selected_movie = RECOMMENDER.title_map.get(match_title.lower())

                    self.send_json({
                        "query_title": title,
                        "selected_movie": selected_movie,
                        "recommendations": recommendations,
                        "algorithm": "Content-Based Filtering (TF-IDF + Cosine Similarity)"
                    })
                    return
                else:
                    self.send_json({"error": "Please provide 'id' or 'title' parameter"}, 400)
                    return

            elif path == "/api/movies":
                query = params.get("search", [""])[0]
                genre = params.get("genre", [""])[0]
                sort_by = params.get("sort_by", ["rating"])[0]
                limit = int(params.get("limit", [50])[0])

                results = MOVIES
                if query:
                    results = RECOMMENDER.search(query, limit=100)
                elif genre and genre.lower() != "all":
                    results = RECOMMENDER.get_by_genre(genre, limit=100)

                # Sorting
                if sort_by == "rating":
                    results = sorted(results, key=lambda x: x.get("rating", 0), reverse=True)
                elif sort_by == "year":
                    results = sorted(results, key=lambda x: x.get("release_year", 0), reverse=True)
                elif sort_by == "title":
                    results = sorted(results, key=lambda x: x.get("title", ""))

                self.send_json({
                    "total": len(results),
                    "movies": results[:limit]
                })
                return

            elif path == "/api/genres":
                genres = RECOMMENDER.get_all_genres()
                self.send_json({"genres": genres})
                return

            elif path == "/api/movie":
                movie_id = params.get("id", [None])[0]
                if not movie_id:
                    self.send_json({"error": "Missing 'id' parameter"}, 400)
                    return
                m = RECOMMENDER.movie_map.get(int(movie_id))
                if not m:
                    self.send_json({"error": "Movie not found"}, 404)
                    return
                self.send_json(m)
                return

            else:
                self.send_json({"error": f"Endpoint {path} not found"}, 404)

        except Exception as e:
            self.send_json({"error": str(e)}, 500)


def run_server(port=PORT):
    server_address = ("", port)
    handler_class = MovieAppRequestHandler
    # Allow address reuse to prevent port busy on restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, handler_class) as httpd:
        print(f"\n=======================================================")
        print(f"  🎬 Movie Recommendation Web Server is LIVE!")
        print(f"  👉 Open URL in Browser: http://localhost:{port}")
        print(f"  👉 API: http://localhost:{port}/api/movies")
        print(f"  👉 ML Engine: TF-IDF + Cosine Similarity (Non-Deep-Learning)")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")


if __name__ == "__main__":
    run_server()
