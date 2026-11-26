"""Student 1: load source data, validate required columns, handle file errors."""
# TODO: add documented loader and validation after dataset is selected.


from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RAW_COLUMNS: tuple[str, ...] = (
    "isbn13",
    "title",
    "authors",
    "categories",
    "description",
    "average_rating",
)


class DataLoadError(Exception):
    """Raised when a dataset file cannot be read or fails schema validation."""


def load_csv_dataset(
    file_path: str | Path,
    required_columns: Iterable[str] = DEFAULT_RAW_COLUMNS,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a CSV dataset safely and verify required columns exist.

    Args:
        file_path: Path to the CSV file.
        required_columns: Column names that must be present in the file.
        encoding: File character encoding (defaults to 'utf-8').

    Returns:
        Loaded pandas DataFrame.

    Raises:
        DataLoadError: If the file is missing, is not a file, cannot be parsed,
            is empty, or lacks required columns.
    """
    path = Path(file_path)

    if not path.exists():
        msg = f"Dataset file not found: {path}"
        logger.error(msg)
        raise DataLoadError(msg)

    if not path.is_file():
        msg = f"Expected a file path, got a directory or special path: {path}"
        logger.error(msg)
        raise DataLoadError(msg)

    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s; retrying with 'latin-1'.", path)
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception as exc:
            msg = f"Failed to parse CSV '{path}' with fallback encoding: {exc}"
            logger.error(msg)
            raise DataLoadError(msg) from exc
    except pd.errors.EmptyDataError as exc:
        msg = f"Dataset file is empty: {path}"
        logger.error(msg)
        raise DataLoadError(msg) from exc
    except pd.errors.ParserError as exc:
        msg = f"Malformed CSV in '{path}': {exc}"
        logger.error(msg)
        raise DataLoadError(msg) from exc
    except OSError as exc:
        msg = f"OS error while reading '{path}': {exc}"
        logger.error(msg)
        raise DataLoadError(msg) from exc

    if df.empty:
        msg = f"Loaded dataset '{path}' contains zero rows."
        logger.error(msg)
        raise DataLoadError(msg)

    required_set = list(dict.fromkeys(required_columns))
    missing_cols = [col for col in required_set if col not in df.columns]
    if missing_cols:
        msg = (
            f"Dataset '{path}' is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
        logger.error(msg)
        raise DataLoadError(msg)

    logger.info(
        "Successfully loaded '%s' with %d rows and %d columns.",
        path,
        len(df),
        len(df.columns),
    )
    return df