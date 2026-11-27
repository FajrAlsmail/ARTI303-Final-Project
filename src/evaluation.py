"""Compare semantic book recommendations with a TF-IDF baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender import load_recommender, recommend_books


EVALUATION_QUERIES = {
    "mystery": (
        "I want a mysterious novel with a dark atmosphere "
        "and complex characters."
    ),
    "fantasy": (
        "I want an epic fantasy adventure with magic, "
        "kingdoms, and dangerous quests."
    ),
    "science_fiction": (
        "I want a science fiction story about space "
        "exploration and alien civilizations."
    ),
}


def tfidf_recommend(query, books, vectorizer, tfidf_matrix, top_n=5):
    """Return the most similar books using TF-IDF."""

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_n]

    results = books.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    return results.reset_index(drop=True)


def calculate_precision_at_5(candidates_path, labels_path):
    """Evaluate top-5 recommendations using reviewed relevance labels."""

    keys = ["query_id", "method", "rank", "book_id"]

    candidates = pd.read_csv(
        candidates_path,
        dtype={"book_id": str}
    )

    labels = pd.read_csv(
        labels_path,
        dtype={"book_id": str}
    )

    if len(candidates) != len(labels):
        raise ValueError("Candidates and labels have different row counts.")

    if not labels["relevant"].isin([0, 1]).all():
        raise ValueError("Relevance labels must be 0 or 1.")

    labels = labels[keys + ["relevant"]].rename(
        columns={"relevant": "human_label"}
    )

    evaluated = candidates.merge(
        labels,
        on=keys,
        how="left",
        validate="one_to_one"
    )

    if evaluated["human_label"].isna().any():
        raise ValueError("Some recommendations have no matching relevance label.")

    summary = (
        evaluated.groupby(["query_id", "method"])["human_label"]
        .agg(relevant_books="sum", evaluated_books="count")
        .reset_index()
    )

    if not (summary["evaluated_books"] == 5).all():
        raise ValueError("Each query and method must have 5 evaluated books.")

    summary["precision_at_5"] = summary["relevant_books"] / 5

    print("\n=== Precision@5 Evaluation ===")
    print(summary.to_string(index=False))

    print("\n=== Average Precision@5 by Method ===")
    print(
        summary.groupby("method")["precision_at_5"]
        .mean()
        .round(3)
    )

    summary.to_csv(
        "assets/evaluation_summary.csv",
        index=False
    )

    print("\nEvaluation summary saved successfully.")


def main():
    books, model, embeddings = load_recommender()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(
        books["description"].fillna("")
    )

    evaluation_rows = []

    for query_id, query in EVALUATION_QUERIES.items():
        print(f"\n=== Query: {query_id} ===")
        print(query)

        semantic_results = recommend_books(
            query, books, model, embeddings, top_n=5
        )

        tfidf_results = tfidf_recommend(
            query, books, vectorizer, tfidf_matrix, top_n=5
        )

        for method, results in [
            ("Semantic", semantic_results),
            ("TF-IDF", tfidf_results),
        ]:
            print(f"\n--- {method} Recommendations ---")
            print(
                results[["title", "similarity_score"]]
                .to_string(index=False)
            )

            for rank, (_, book) in enumerate(results.iterrows(), start=1):
                evaluation_rows.append({
                    "query_id": query_id,
                    "query": query,
                    "method": method,
                    "rank": rank,
                    "book_id": book["book_id"],
                    "title": book["title"],
                    "description": book["description"],
                    "relevant": "",
                })

    output_path = Path("assets/evaluation_candidates.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(evaluation_rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nEvaluation candidates saved to: {output_path}")
    print(f"Total recommendations: {len(evaluation_rows)}")

    calculate_precision_at_5(
        output_path,
        Path("assets/evaluation_labels.csv")
    )


if __name__ == "__main__":
    main()

