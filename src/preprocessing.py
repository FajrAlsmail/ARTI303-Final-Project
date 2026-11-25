"""Student 1: deduplicate, handle missing fields, clean text, export data."""
# TODO: implement dataset-specific cleaning without fabricating book attributes.


from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS: list[str] = [
    "book_id",
    "title",
    "authors",
    "description",
    "genres",
    "average_rating",
]


class PreprocessingError(Exception):
    """Raised when dataset cleaning or transformation fails."""


def _clean_text(value: Any) -> str | None:
    """Normalize whitespace and decode HTML entities without inventing content."""
    if value is None or pd.isna(value):
        return None
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def _format_id(value: Any) -> str | None:
    """Convert numeric or string IDs cleanly without scientific notation."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        # Avoid '.0' on float representations of ISBNs/IDs
        if float(value).is_integer():
            return str(int(value))
    text = re.sub(r"\s+", "", str(value)).strip()
    return text if text else None


def _normalize_delimiters(value: Any) -> str | None:
    """Clean semicolon/pipe/comma-separated author or genre strings."""
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    parts = [p.strip() for p in re.split(r"[;|]+", cleaned) if p.strip()]
    return "; ".join(parts) if parts else None


def clean_books_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and standardize raw book data into the team's target schema.

    Args:
        df: Raw DataFrame loaded from the source CSV.

    Returns:
        A tuple of `(cleaned_df, summary_stats)`.

    Raises:
        PreprocessingError: If input DataFrame is empty or lacks essential columns.
    """
    if df is None or df.empty:
        raise PreprocessingError("Cannot preprocess an empty DataFrame.")

    working = df.copy()
    raw_rows = len(working)

    # Map source column names to standardized schema names
    rename_map: dict[str, str] = {}
    if "isbn13" in working.columns and "book_id" not in working.columns:
        rename_map["isbn13"] = "book_id"
    if "categories" in working.columns and "genres" not in working.columns:
        rename_map["categories"] = "genres"
    if rename_map:
        working = working.rename(columns=rename_map)

    for required in ("title", "description"):
        if required not in working.columns:
            raise PreprocessingError(f"Missing required column '{required}' for cleaning.")

    # Clean text fields without fabricating missing book attributes
    if "book_id" in working.columns:
        working["book_id"] = working["book_id"].apply(_format_id)
    else:
        working["book_id"] = [f"row_{i}" for i in range(len(working))]

    working["title"] = working["title"].apply(_clean_text)
    working["description"] = working["description"].apply(_clean_text)

    if "authors" in working.columns:
        working["authors"] = working["authors"].apply(_normalize_delimiters)
    else:
        working["authors"] = None

    if "genres" in working.columns:
        working["genres"] = working["genres"].apply(_normalize_delimiters)
    else:
        working["genres"] = None

    if "average_rating" in working.columns:
        working["average_rating"] = pd.to_numeric(working["average_rating"], errors="coerce")
        # Keep only plausible ratings in [0.0, 5.0]; set out-of-range values to NaN
        invalid_ratings = working["average_rating"].notna() & (
            (working["average_rating"] < 0.0) | (working["average_rating"] > 5.0)
        )
        if invalid_ratings.any():
            logger.warning("Found %d out-of-range ratings; setting to NaN.", invalid_ratings.sum())
            working.loc[invalid_ratings, "average_rating"] = float("nan")
    else:
        working["average_rating"] = float("nan")

    # Drop rows missing title or description (needed for semantic book discovery)
    before_required_drop = len(working)
    working = working.dropna(subset=["title", "description"]).copy()
    dropped_missing_core = before_required_drop - len(working)

    # Deduplicate by book_id when available, or normalized (title, authors)
    before_dedup = len(working)
    dedup_subset = ["book_id"] if working["book_id"].notna().all() else ["title", "authors"]
    working = working.drop_duplicates(subset=dedup_subset, keep="first").copy()
    dropped_duplicates = before_dedup - len(working)

    # If book_id still had any missing values, generate deterministic fallback IDs
    missing_id_mask = working["book_id"].isna()
    if missing_id_mask.any():
        working.loc[missing_id_mask, "book_id"] = [
            f"book_{idx}" for idx in working.index[missing_id_mask]
        ]

    working = working[OUTPUT_COLUMNS].reset_index(drop=True)

    summary: dict[str, Any] = {
        "raw_rows": raw_rows,
        "dropped_missing_title_or_description": dropped_missing_core,
        "dropped_duplicates": dropped_duplicates,
        "cleaned_rows": len(working),
        "missing_authors_after_clean": int(working["authors"].isna().sum()),
        "missing_genres_after_clean": int(working["genres"].isna().sum()),
        "missing_rating_after_clean": int(working["average_rating"].isna().sum()),
    }
    logger.info("Cleaning summary: %s", summary)
    return working, summary


def export_cleaned_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Export the cleaned DataFrame to CSV."""
    if df is None or df.empty:
        raise PreprocessingError("Refusing to export an empty cleaned DataFrame.")

    out_file = Path(output_path)
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False, encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to write cleaned dataset to '{out_file}': {exc}"
        logger.error(msg)
        raise PreprocessingError(msg) from exc

    logger.info("Exported %d cleaned rows to '%s'.", len(df), out_file)
    return out_file


def run_pipeline(
    input_csv: str | Path = "data/raw/books.csv",
    output_csv: str | Path = "data/processed/cleaned_books.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Load -> Validate -> Clean -> Transform -> Export -> Summary for Student 1."""
    from src.data_loader import load_csv_dataset

    raw_df = load_csv_dataset(input_csv)
    cleaned_df, summary = clean_books_dataframe(raw_df)
    export_cleaned_dataset(cleaned_df, output_csv)
    return cleaned_df, summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cleaned, stats = run_pipeline()
    print("=== Data Cleaning Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")