"""AI-powered semantic book recommendation engine."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/processed/cleaned_books.csv")
CACHE_DIR = Path("data/processed/embedding_cache")
MODEL_NAME = "all-MiniLM-L6-v2"


def load_recommender():
    """Load book data, AI model, and cached embeddings."""

    books = pd.read_csv(DATA_PATH)
    books = books.dropna(subset=["description"]).reset_index(drop=True)

    descriptions = books["description"].astype(str).tolist()

    if not descriptions:
        raise ValueError("No book descriptions available.")

    model = SentenceTransformer(MODEL_NAME)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.sha256(
        ("\n".join(descriptions) + MODEL_NAME).encode("utf-8")
    ).hexdigest()[:12]

    cache_path = CACHE_DIR / f"embeddings_{cache_key}.npy"

    if cache_path.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(cache_path, allow_pickle=False)

        if embeddings.ndim != 2 or len(embeddings) != len(books):
            raise ValueError("Invalid embedding cache shape.")
    else:
        print("Generating book embeddings...")
        embeddings = model.encode(
            descriptions,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        np.save(cache_path, embeddings)
        print("Embeddings saved successfully!")

    return books, model, embeddings


def recommend_books(query, books, model, embeddings, top_n=5):
    """Recommend books based on semantic similarity."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a book description.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if len(books) != len(embeddings):
        raise ValueError("Books and embeddings do not match.")

    query_embedding = model.encode([query.strip()])

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_n]

    recommendations = books.iloc[top_indices].copy()

    recommendations["similarity_score"] = scores[top_indices]

    return recommendations.reset_index(drop=True)


def main():
    """Run a simple recommendation demo."""

    books, model, embeddings = load_recommender()

    print("Books loaded:", len(books))
    print("Embeddings shape:", embeddings.shape)

    user_query = (
        "I want a mysterious novel with a dark "
        "atmosphere and complex characters."
    )

    results = recommend_books(
        user_query,
        books,
        model,
        embeddings,
        top_n=5
    )

    print("\nTop 5 Book Recommendations:\n")

    for rank, (_, book) in enumerate(results.iterrows(), start=1):
        print(f"{rank}. {book['title']}")
        print(f"   Similarity: {book['similarity_score']:.3f}")
        print()


if __name__ == "__main__":
    main()
