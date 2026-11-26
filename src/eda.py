"""Student 2: Exploratory Data Analysis and data summary."""

import logging
from pathlib import Path

import pandas as pd

from src.data_loader import load_csv_dataset
import re
import html

logger = logging.getLogger(__name__)

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "cleaned_books.csv"
)

REQUIRED_COLUMNS = [
    "book_id",
    "title",
    "authors",
    "description",
    "genres",
    "average_rating"
]


def load_cleaned_data():
    """Load and validate the cleaned book dataset."""
    return load_csv_dataset(
        DATA_PATH,
        required_columns=REQUIRED_COLUMNS
    )


def explore_data(df: pd.DataFrame):
    """Display basic information about the dataset."""

    print("\n=== Dataset Shape ===")
    print(f"Number of books: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n=== Column Names ===")
    print(df.columns.tolist())

    print("\n=== Dataset Preview ===")
    print(df[["title", "genres", "average_rating"]].head())

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Rating Statistics ===")
    print(df["average_rating"].describe())

    logger.info("Initial data exploration completed.")

def get_genre_distribution(df: pd.DataFrame) -> pd.Series:
    """Count books in each genre."""

    genres = (
        df["genres"]
        .dropna()
        .str.split(";")
        .explode()
        .str.strip()
    )

    genres = genres[genres != ""]

    return genres.value_counts()

def add_description_length(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the number of words in each description."""

    result = df.copy()

    result["description_length"] = (
        result["description"]
        .fillna("")
        .str.split()
        .str.len()
    )

    return result

def get_rating_distribution(df: pd.DataFrame) -> pd.Series:
    """Return valid book ratings for analysis."""

    ratings = df["average_rating"].dropna()

    return ratings[ratings.between(0, 5)]

def transform_text(text: str) -> str:
    """Prepare book descriptions for semantic analysis."""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# Student 3: Use processed_description as input for text embeddings.

def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare book descriptions for the recommendation engine."""

    result = df.copy()

    result["processed_description"] = (
        result["description"]
        .fillna("")
        .apply(transform_text)
    )

    logger.info("Text transformation completed.")

    return result

def generate_summary(df: pd.DataFrame) -> dict:
    """Generate a statistical summary of the dataset."""

    lengths = add_description_length(df)["description_length"]
    ratings = get_rating_distribution(df)
    genres = get_genre_distribution(df)

    return {
        "total_books": len(df),
        "unique_genres": len(genres),
        "missing_authors": int(df["authors"].isna().sum()),
        "missing_genres": int(df["genres"].isna().sum()),
        "missing_ratings": int(df["average_rating"].isna().sum()),
        "average_rating": round(ratings.mean(), 2),
        "average_description_length": round(lengths.mean(), 2),
        "median_description_length": lengths.median(),
        "min_description_length": lengths.min(),
        "max_description_length": lengths.max(),
        "most_common_genre": genres.idxmax()
    }

def export_results(df: pd.DataFrame):
    """Export analytical results and transformed data."""

    project_dir = Path(__file__).resolve().parents[1]

    summary_path = project_dir / "assets" / "data_summary.csv"

    transformed_path = (
        project_dir / "data" / "processed" / "transformed_books.csv"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = generate_summary(df)

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # Student 3: Use transformed_books.csv for text embeddings.
    transformed_df = transform_dataset(df)
    transformed_df.to_csv(transformed_path, index=False)

    logger.info("Summary exported to %s", summary_path)
    logger.info("Transformed data exported to %s", transformed_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_cleaned_data()
    explore_data(df)

    print("\n=== Top 10 Genres ===")
    print(get_genre_distribution(df).head(10))

    print("\n=== Description Length Statistics ===")
    analyzed_df = add_description_length(df)
    print(analyzed_df["description_length"].describe())

    print("\n=== Rating Distribution ===")
    print(get_rating_distribution(df).describe())

    transformed_df = transform_dataset(df)

    print("\n=== Text Transformation Preview ===")
    print(
        transformed_df[
            ["description", "processed_description"]
        ].head()
    )
    print("\n=== Data Summary ===")

    for key, value in generate_summary(df).items():
        print(f"{key}: {value}")

    export_results(df)