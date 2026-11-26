"""Unit tests for Student 1 data loading and preprocessing modules."""

import pandas as pd
import pytest

from src.data_loader import DataLoadError, load_csv_dataset
from src.preprocessing import OUTPUT_COLUMNS, PreprocessingError, clean_books_dataframe, export_cleaned_dataset


def test_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(DataLoadError):
        load_csv_dataset(tmp_path / "nonexistent.csv")


def test_load_csv_missing_required_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["Book A"]}).to_csv(bad_csv, index=False)

    with pytest.raises(DataLoadError):
        load_csv_dataset(bad_csv)


def test_clean_books_dataframe_standardizes_and_deduplicates():
    raw = pd.DataFrame(
        {
            "isbn13": [9780007458424.0, 9780007458424.0, 9780140449136.0, 9780000000001.0],
            "title": ["  The Hobbit  ", "The Hobbit", "Crime and Punishment", "No Desc Book"],
            "authors": ["J.R.R. Tolkien", "J.R.R. Tolkien", "Fyodor Dostoevsky", "Author X"],
            "categories": ["Fiction; Fantasy", "Fiction; Fantasy", "Fiction", "Mystery"],
            "description": ["A <b>great</b> journey.", "Duplicate row", "A psychological novel.", None],
            "average_rating": [4.28, 4.28, 4.23, 3.9],
        }
    )

    cleaned, stats = clean_books_dataframe(raw)

    assert list(cleaned.columns) == OUTPUT_COLUMNS
    assert len(cleaned) == 2
    assert cleaned.loc[0, "book_id"] == "9780007458424"
    assert cleaned.loc[0, "description"] == "A great journey."
    assert stats["dropped_missing_title_or_description"] == 1
    assert stats["dropped_duplicates"] == 1


def test_export_cleaned_dataset_writes_csv(tmp_path):
    df = pd.DataFrame(
        [
            {
                "book_id": "9780007458424",
                "title": "The Hobbit",
                "authors": "J.R.R. Tolkien",
                "description": "A great journey.",
                "genres": "Fiction; Fantasy",
                "average_rating": 4.28,
            }
        ]
    )
    out_path = export_cleaned_dataset(df, tmp_path / "processed" / "cleaned_books.csv")
    assert out_path.is_file()
    loaded_back = pd.read_csv(out_path)
    assert len(loaded_back) == 1