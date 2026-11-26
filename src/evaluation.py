"""Compare semantic book recommendations with a TF-IDF baseline."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender import load_recommender, recommend_books


def tfidf_recommend(query, books, vectorizer, tfidf_matrix, top_n=5):
    """Return the most similar books using TF-IDF."""

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = books.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    return results.reset_index(drop=True)


def main():
    # Load the semantic recommendation system
    books, model, embeddings = load_recommender()

    # Build the traditional TF-IDF baseline
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(
        books["description"].fillna("")
    )

    query = (
        "I want a mysterious novel with a dark "
        "atmosphere and complex characters."
    )

    semantic_results = recommend_books(
        query, books, model, embeddings, top_n=5
    )

    tfidf_results = tfidf_recommend(
        query, books, vectorizer, tfidf_matrix, top_n=5
    )

    print("\n=== Semantic Recommendations ===")
    print(semantic_results[["title", "similarity_score"]].to_string(index=False))

    print("\n=== TF-IDF Recommendations ===")
    print(tfidf_results[["title", "similarity_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
