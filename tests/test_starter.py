"""Starter smoke test only; feature owners must add substantive tests."""
from pathlib import Path


def test_expected_source_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").is_file()
    assert (root / "src" / "recommender.py").is_file()
