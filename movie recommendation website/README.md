# CineMatch - Movie Recommendation System Website 🎬

A full-stack, responsive Movie Recommendation website powered by **classical Machine Learning (TF-IDF Vectorization & Cosine Similarity)** with **zero deep learning**.

Built using **Python, HTML5, CSS3, JavaScript, and Java**.

---

## 🚀 Key Features

1. **Content-Based Machine Learning (No Deep Learning)**:
   - Evaluates movie metadata (plot synopsis, genres, director, top cast, keywords).
   - Computes weighted **TF-IDF (Term Frequency-Inverse Document Frequency)** vectors with n-grams.
   - Calculates pairwise **Cosine Similarity** to instantly recommend mathematically closest films.
   - Transparent recommendations with similarity percentages (e.g. *94% Match*) and explainability reasons.

2. **Full-Featured Modern Frontend**:
   - **Cinematic Dark Theme**: Netflix/Apple TV+ glassmorphic UI with smooth animations.
   - **Live Search & Autocomplete**: Real-time suggestion dropdown as you type.
   - **Recommendation Spotlight Stage**: Dedicated interactive banner & grid for the selected movie.
   - **Genre Filtering**: Filter catalog by Sci-Fi, Action, Drama, Thriller, Animation, Comedy, etc.
   - **Sorting**: Sort by IMDb rating, release year, or title.
   - **Movie Details Modal**: High-res poster, full synopsis, cast, director, and runtime.
   - **Watchlist Drawer**: Save your favorite movies locally in your browser (`localStorage`).
   - **"Surprise Me" Feature**: One-click random recommendation generator.

3. **Multi-Language Support**:
   - **Python Web Server & REST API** (`server.py`): Zero external dependencies; runs on Python standard library!
   - **Java Server Alternative** (`MovieRecommenderServer.java`): Built using standard Java SE `com.sun.net.httpserver`.

---

## 📁 Project Structure

```text
movie recommendation website/
├── data/
│   ├── movies.json            # Curated movie dataset with rich metadata
│   └── movies.csv             # Exported CSV dataset
├── static/
│   ├── css/
│   │   └── style.css          # Cinematic dark theme styling
│   └── js/
│       └── app.js             # Interactive frontend application logic
├── templates/
│   └── index.html             # Semantic HTML5 layout
├── dataset_loader.py          # Data ingestion and CSV exporter
├── ml_recommender.py          # TF-IDF & Cosine Similarity ML engine
├── server.py                  # Python HTTP web server & REST API
├── MovieRecommenderServer.java# Standalone Java server implementation
├── test_recommender.py        # Automated ML verification test suite
└── README.md                  # Project documentation
```

---

## 🏃 How to Run the Website

### Option 1: Run with Python (Recommended)

1. Open your terminal in this directory:
   ```bash
   cd "c:\Users\HP\Documents\movie recommendation website"
   ```
2. Start the Python server:
   ```bash
   python server.py
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

*(Note: No pip packages are required! The ML engine and server run on pure Python 3 standard library).*

---

### Option 2: Run with Java

1. Open your terminal in this directory:
   ```bash
   javac MovieRecommenderServer.java
   java MovieRecommenderServer
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

---

## 🧪 Run Automated Tests

To test the Machine Learning algorithms and recommendation accuracy:
```bash
python test_recommender.py
```

All tests verify dataset schema integrity, tokenization, IDF computations, and that films like *Inception* accurately produce high-similarity sci-fi/Nolan matches (*Tenet*, *Interstellar*, *The Matrix*).
