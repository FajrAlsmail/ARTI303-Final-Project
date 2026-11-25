# Dataset provenance and handling

**Status:** dataset choice is pending. Student 1 must fill in these details *before* data is used.

- Dataset name: 7k Books with Metadata (`books.csv`, compiled from Soumik's Goodreads-books dataset and Google Books API)
- Source URL: https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata
- License / usage conditions: CC0: Public Domain 
- Date downloaded: 2026-11-26
- Original row and column counts: 6,810 rows and 12 columns
- Any restrictions on redistributing raw text/ratings: Licensed under CC0: Public Domain on Kaggle; however, per team workflow and `.gitignore`, raw (`data/raw/`) and full processed (`data/processed/`) dataset files are not committed to Git by default. Only small permitted samples (if needed) should be placed in `data/sample/` and documented.
- Cleaning and transformation steps: Loaded `data/raw/books.csv` and validated required columns; mapped `isbn13` to `book_id` and `categories` to `genres`; decoded HTML entities and normalized whitespace/delimiters without inventing missing book attributes; validated `average_rating` values within `[0.0, 5.0]` (setting out-of-range values to `NaN`); dropped 262 rows missing `title` or `description`; removed 0 duplicate rows; exported 6,548 cleaned rows to `data/processed/cleaned_books.csv` (leaving 65 missing `authors`, 37 missing `genres`, and 37 missing `average_rating`).
- Final cleaned schema: `book_id`, `title`, `authors`, `description`, `genres`, `average_rating`
- Where teammates can legally download the source: Download from https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata
- Steps to recreate `data/processed/cleaned_books.csv`:   
  1. Download `books.csv` from the Kaggle dataset page and place it at `data/raw/books.csv`.
  2. Activate the project virtual environment (`.venv`) and install dependencies from `requirements.txt`.
  3. Run `python -m src.preprocessing` from the repository root to generate `data/processed/cleaned_books.csv`.


Do not commit confidential information, unlicensed data, or large data files by default. If small samples are permitted by the source terms, put a clearly identified sample in `data/sample/` and document it.

