# AI-Powered Book Discovery System

> **Course:** ARTI303 — Programming for Artificial Intelligence  
> **Project type:** Final group project (5 students)  
> **Project name:**  
> **Status:** In development — this repository currently contains a starter structure, not a finished AI application.

## 1. Project Overview

Choosing a book can be difficult when conventional search tools mainly rely on book genres, authors, titles, or popularity. Readers often know the *reading experience* they want—such as a mysterious atmosphere, slow pacing, emotionally complex characters, or a specific theme—but may not know which book matches those preferences.

This project aims to develop an **AI-powered book discovery application** that helps readers find books based on the *meaning* of their preferences rather than only exact keywords, genres, or authors. Using **Natural Language Processing (NLP)** and **semantic similarity**, the planned system will compare a reader's description with book information from a documented, publicly available dataset and rank relevant books.

The application will be developed in **Python** and presented through an interactive **Streamlit** interface.

## 2. Problem Statement

Traditional book searches can produce overwhelming or poorly matched results when a reader's interests are more specific than a genre label. For example, someone may want a slow-paced mystery with wintery atmosphere and character development, while avoiding romance as the main plot. A genre-only search cannot express all these preferences.

**Target users:** General readers who want to discover their next book based on mood, atmosphere, themes, and personal reading preferences.

**Goal:** Reduce the time and effort needed to discover suitable books by offering explainable recommendations based on textual preferences.

## 3. Proposed Solution

A reader will describe the type of book they want. The system is planned to:

1. Accept a **free-text description** of the desired reading experience.
2. Optionally accept preferences such as **mood**, **themes**, and **elements to avoid**.
3. Process book metadata and descriptions using NLP.
4. Compute **semantic similarity** between the reader's request and available book descriptions.
5. Rank and display the **top five matching books**.
6. Provide a brief **“Why this book?”** explanation grounded in the available metadata, along with relevant book details.
7. Display at least one informative visualization about the dataset or results.

**Example user request:**

> I want a mysterious, atmospheric book with complex characters and a gradual pace. I would prefer romance not to be the main focus.

**Expected behavior:** The application returns a ranked list of candidate books and explains which available descriptions or themes support each suggestion. The example above is illustrative; no recommendation accuracy is claimed before evaluation.

### Planned MVP vs. optional features

| Core functionality (MVP) | Optional, only if data supports it |
| --- | --- |
| Free-text search | More detailed pacing preferences |
| Semantic book matching and ranking | Writing-style or character-development analysis |
| Mood/theme preferences based on available text | Match the experience of a previously enjoyed book |
| Top-five recommendations with evidence-based explanations | Richer exclusion of unwanted themes |
| Streamlit inputs, outputs, and visualization | Additional book cover or metadata lookup |

**Important limitation:** Descriptions and metadata may not reliably disclose every sensitive theme, writing style, or story element. An “avoid” filter can only flag or exclude content supported by the available data; it cannot guarantee the absence of a topic.

## 4. AI Approach (Planned)

- **Text preparation:** Validate, clean, and normalize the book fields needed for search.
- **Semantic representation:** Use a pretrained sentence-embedding model (for example, Sentence Transformers) to represent descriptions and search queries numerically.
- **Recommendation:** Use cosine similarity to rank relevant books; apply supported preference filters where applicable.
- **Evaluation:** Compare semantic retrieval against a simpler baseline (such as TF-IDF), examine representative user queries, and report relevant evaluation evidence and failures.
- **Explainability:** Ground each recommendation explanation in matching text or metadata, avoiding unsupported claims.

The exact model and evaluation protocol will be finalized after the dataset is inspected. **These are planned features, not a claim that the model is already implemented.**

## 5. Dataset and Processing Workflow

**Dataset status: Not finalized.** The team will choose a publicly available, documented book dataset (preferably from Kaggle) that includes useful textual descriptions. Book title, author, description, and genre/theme information are especially helpful. Ratings are optional.

Before use, the team will document the dataset's source URL, license or usage conditions, limitations, raw and cleaned sizes, and any restrictions on redistributing the files. See [`data/README.md`](data/README.md).

The required processing pipeline is:

**Load → Validate → Clean → Explore → Visualize → Transform → Export → Summary**

Planned explorations include genre distribution, availability/length of descriptions, and rating distribution when ratings are present. All charts will use real dataset fields.

### Proposed data and recommendation interface

The final cleaned schema will be agreed upon after inspecting the source. Proposed fields include `book_id`, `title`, `authors`, and `description`; `genres` and `average_rating` are optional.

The recommendation component is expected to accept a free-text `query` and optional `mood`, `themes`, and `avoid` preferences. Its proposed output contains `book_id`, `title`, `authors`, `description`, `score`, and `reason` for each suggested book. Scores indicate model similarity, **not guaranteed reader satisfaction**.

## 6. Technology Stack

- **Python** — application and data processing
- **pandas / NumPy** — loading, validation, cleaning, and exploration
- **Matplotlib** — visualizations
- **Sentence Transformers / scikit-learn** — NLP representations, similarity, and evaluation
- **Streamlit** — interactive user interface
- **pytest** — basic testing
- **GitHub** — version control and teamwork

The exact dependencies may be adjusted when implementation is completed.

## 7. Project Structure

```text
ARTI303-Final-Project/
├── app.py                    # Streamlit entry point (currently a placeholder)
├── src/
│   ├── data_loader.py        # Load and validate dataset
│   ├── preprocessing.py      # Data and text cleaning
│   ├── eda.py                # Exploratory data analysis
│   ├── visualizations.py     # Charts
│   ├── recommender.py        # NLP recommendation engine
│   ├── evaluation.py         # Model evaluation
│   ├── filters.py            # Preference filters
│   └── explanations.py       # Reasons for recommendations
├── data/                     # Dataset notes and permitted examples
├── assets/                   # Charts and other media
├── tests/                    # Automated tests
├── docs/                     # Report, presentation, AI usage log
├── config.example.json       # Example settings
├── requirements.txt          # Python dependencies
├── CONTRIBUTING.md           # Git workflow and team responsibilities
└── README.md
```

## 8. Running the Starter App

From the repository root, using Git Bash on Windows:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

**Current behavior:** The starter app accepts a description and explains that the recommendation engine is not connected yet. Book recommendations will appear only after the team implements and tests that feature. These setup instructions must be rechecked from a fresh clone before final submission.

## 9. Team Collaboration

The project is developed by **five students**. Members have separate primary responsibilities for data preparation, EDA/visualization, AI recommendation and evaluation, Streamlit, and integration/testing. Every member also contributes to the report, presentation, and AI usage documentation.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for task ownership, feature branches, Pull Requests, review rules, and internal deadlines; team coordination details are intentionally kept there instead of dominating the project description.

## 10. Deliverables and Limitations

The final submission is planned to include a working Streamlit application, reproducible source code, documented data workflow, AI evaluation, basic tests, a PDF project report, an AI Usage Report, and a presentation. The deadline is **December 3, 2026**.

Potential limitations include incomplete or outdated book descriptions, inconsistent genres and tags, language coverage, popularity bias, and uncertain inference of moods or themes. The final report will document observed limitations and actual results, not just plans.

---

**Note:** The final application name, dataset, authors, evaluation results, screenshots, and tested setup instructions will be added as the team completes those components.
