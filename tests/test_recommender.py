
"""Unit tests for the semantic book recommendation engine."""

import src.recommender as recommender
import numpy as np
import pandas as pd
import pytest

from src.recommender import recommend_books


class FakeModel:
    """Simulate the AI model without downloading model weights."""

    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


@pytest.fixture
def sample_data():
    """Create a small dataset for testing."""

    books = pd.DataFrame({
        "book_id": [1, 2, 3],
        "title": [
            "Matching Book",
            "Related Book",
            "Unrelated Book"
        ],
        "description": [
            "A mysterious adventure",
            "A similar story",
            "An unrelated topic"
        ]
    })

    embeddings = np.array([
        [1.0, 0.0],
        [0.6, 0.8],
        [-1.0, 0.0]
    ])

    return books, FakeModel(), embeddings


def test_recommendations_are_ranked(sample_data):
    """Books should be ranked by semantic similarity."""

    books, model, embeddings = sample_data

    results = recommend_books(
        "A mysterious adventure",
        books,
        model,
        embeddings,
        top_n=3
    )

    assert results["title"].tolist() == [
        "Matching Book",
        "Related Book",
        "Unrelated Book"
    ]


def test_top_n_recommendations(sample_data):
    """The function should return the requested number of books."""

    books, model, embeddings = sample_data

    results = recommend_books(
        "A fantasy story",
        books,
        model,
        embeddings,
        top_n=2
    )

    assert len(results) == 2
    assert "similarity_score" in results.columns


@pytest.mark.parametrize("query", ["", "   ", None])
def test_invalid_query(sample_data, query):
    """Empty or invalid queries should raise an error."""

    books, model, embeddings = sample_data

    with pytest.raises(ValueError):
        recommend_books(query, books, model, embeddings)


@pytest.mark.parametrize("top_n", [0, -1])
def test_invalid_top_n(sample_data, top_n):
    """The number of recommendations must be positive."""

    books, model, embeddings = sample_data

    with pytest.raises(ValueError):
        recommend_books(
            "Adventure",
            books,
            model,
            embeddings,
            top_n=top_n
        )


def test_embedding_mismatch(sample_data):
    """Books and embeddings must have matching lengths."""

    books, model, embeddings = sample_data

    with pytest.raises(ValueError):
        recommend_books(
            "Adventure",
            books,
            model,
            embeddings[:2]
        )


def test_original_data_not_modified(sample_data):
    """Recommendations must not modify the original dataset."""

    books, model, embeddings = sample_data

    recommend_books(
        "Adventure",
        books,
        model,
        embeddings
    )

    assert "similarity_score" not in books.columns


def test_top_n_larger_than_dataset(sample_data):
    """Requesting extra books should return only available books."""

    books, model, embeddings = sample_data

    results = recommend_books(
        "Adventure",
        books,
        model,
        embeddings,
        top_n=10
    )

    assert len(results) == 3


def test_embedding_cache_is_reused(tmp_path, monkeypatch):
    """Generate embeddings once and reuse the saved cache."""

    data_path = tmp_path / "books.csv"
    cache_dir = tmp_path / "embedding_cache"

    sample_books = pd.DataFrame({
        "book_id": [1, 2],
        "title": ["Book One", "Book Two"],
        "description": [
            "A magical adventure",
            "A mysterious story"
        ]
    })

    sample_books.to_csv(data_path, index=False)

    class CountingModel:
        def __init__(self):
            self.encode_calls = 0

        def encode(self, texts, **kwargs):
            self.encode_calls += 1
            return np.array([
                [1.0, 0.0],
                [0.0, 1.0]
            ])

    fake_model = CountingModel()

    monkeypatch.setattr(recommender, "DATA_PATH", data_path)
    monkeypatch.setattr(recommender, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        recommender,
        "SentenceTransformer",
        lambda model_name: fake_model
    )

    # First run: generate embeddings
    books, model, embeddings = recommender.load_recommender()

    assert len(books) == 2
    assert embeddings.shape == (2, 2)
    assert fake_model.encode_calls == 1
    assert len(list(cache_dir.glob("*.npy"))) == 1

    # Second run: load existing embeddings
    books, model, cached_embeddings = recommender.load_recommender()

    assert fake_model.encode_calls == 1
    np.testing.assert_array_equal(
        embeddings,
        cached_embeddings
    )
