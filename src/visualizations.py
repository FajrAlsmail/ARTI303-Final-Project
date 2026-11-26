"""Student 2: Book dataset visualizations."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eda import (
    load_cleaned_data,
    get_genre_distribution,
    add_description_length,
    get_rating_distribution
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets"


# Student 4: All plotting functions return fig for direct use with st.pyplot().
def plot_genre_distribution(df: pd.DataFrame):
    """Plot the ten most common book genres."""

    genres = get_genre_distribution(df).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    genres.sort_values().plot(
        kind="barh",
        ax=ax,
        color="#52796F"
    )

    ax.set_title("Top 10 Book Genres")
    ax.set_xlabel("Number of Books")
    ax.set_ylabel("Genre")

    fig.tight_layout()

    return fig

def plot_rating_distribution(df: pd.DataFrame):
    """Plot the distribution of book ratings."""

    ratings = get_rating_distribution(df)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        ratings,
        bins=25,
        range=(0, 5),
        color="#84A98C",
        edgecolor="white"
    )

    ax.axvline(
        ratings.mean(),
        color="#354F52",
        linestyle="--",
        label=f"Mean: {ratings.mean():.2f}"
    )

    ax.set_title("Book Rating Distribution")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Number of Books")
    ax.legend()

    fig.tight_layout()

    return fig

def plot_description_length(df: pd.DataFrame):
    """Plot the distribution of description lengths."""

    lengths = add_description_length(df)["description_length"]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        lengths,
        bins=40,
        color="#52796F",
        edgecolor="white"
    )

    ax.axvline(
        lengths.mean(),
        color="#354F52",
        linestyle="--",
        label=f"Mean: {lengths.mean():.2f} words"
    )

    ax.set_title("Book Description Length Distribution")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Number of Books")
    ax.legend()

    fig.tight_layout()

    return fig

def plot_average_rating_by_genre(df: pd.DataFrame):
    """Compare average ratings of the ten most common genres."""

    top_genres = get_genre_distribution(df).head(10).index

    genre_data = df[["genres", "average_rating"]].copy()

    genre_data["genres"] = genre_data["genres"].str.split(";")

    genre_data = genre_data.explode("genres")
    genre_data["genres"] = genre_data["genres"].str.strip()

    genre_data = genre_data[
        genre_data["genres"].isin(top_genres)
        & genre_data["average_rating"].between(0, 5)
    ]

    averages = (
        genre_data.groupby("genres")["average_rating"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    averages.plot(
        kind="barh",
        ax=ax,
        color="#84A98C"
    )

    ax.set_title("Average Rating by Popular Genre")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Genre")
    ax.set_xlim(0, 5)

    fig.tight_layout()

    return fig

def save_visualizations(df: pd.DataFrame):
    """Generate and export all dataset visualizations."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plots = {
        "genre_distribution.png": plot_genre_distribution,
        "rating_distribution.png": plot_rating_distribution,
        "description_length.png": plot_description_length,
        "average_rating_by_genre.png": plot_average_rating_by_genre
    }

    for filename, plot_function in plots.items():

        fig = plot_function(df)

        output_path = OUTPUT_DIR / filename

        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

        logger.info("Saved visualization: %s", output_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = load_cleaned_data()

    save_visualizations(df)

    print("\nAll visualizations generated successfully.")
