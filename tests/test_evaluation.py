
"""Unit tests for recommendation evaluation."""

import pandas as pd
import pytest

from src.evaluation import calculate_precision_at_5


def test_precision_at_5_calculation(tmp_path, monkeypatch):
    """Check that Precision@5 is calculated correctly."""

    monkeypatch.chdir(tmp_path)

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    candidates = []
    labels = []

    for method in ["Semantic", "TF-IDF"]:
        for rank in range(1, 6):
            book_id = f"{method}-{rank}"

            candidates.append({
                "query_id": "mystery",
                "method": method,
                "rank": rank,
                "book_id": book_id,
                "title": f"Book {rank}",
            })

            labels.append({
                "query_id": "mystery",
                "method": method,
                "rank": rank,
                "book_id": book_id,
                "relevant": int(
                    rank <= (3 if method == "Semantic" else 1)
                ),
            })

    candidates_path = assets_dir / "candidates.csv"
    labels_path = assets_dir / "labels.csv"

    pd.DataFrame(candidates).to_csv(candidates_path, index=False)
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    calculate_precision_at_5(candidates_path, labels_path)

    summary = pd.read_csv(
        assets_dir / "evaluation_summary.csv"
    )

    scores = dict(zip(
        summary["method"],
        summary["precision_at_5"]
    ))

    assert scores["Semantic"] == pytest.approx(0.6)
    assert scores["TF-IDF"] == pytest.approx(0.2)


def test_invalid_relevance_label(tmp_path, monkeypatch):
    """Reject relevance values other than 0 or 1."""

    monkeypatch.chdir(tmp_path)

    (tmp_path / "assets").mkdir()

    candidates = pd.DataFrame([{
        "query_id": "mystery",
        "method": "Semantic",
        "rank": 1,
        "book_id": "101",
    }])

    labels = candidates.copy()
    labels["relevant"] = 2

    candidates_path = tmp_path / "candidates.csv"
    labels_path = tmp_path / "labels.csv"

    candidates.to_csv(candidates_path, index=False)
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ValueError):
        calculate_precision_at_5(candidates_path, labels_path)
