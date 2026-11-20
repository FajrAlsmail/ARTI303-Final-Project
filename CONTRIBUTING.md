# Team collaboration rules

## First-time setup

1. Accept the GitHub collaborator invitation.
2. Clone the group repository (do **not** create a fork for this group workflow).
3. Work in your local copy using your own feature branch.

## Every task

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/student2-eda    # change the branch name for your task
# edit only your assigned files where practical
git add <paths-you-changed>
git commit -m "Add EDA charts"
git push -u origin feature/student2-eda
```

Open a PR on GitHub: **base `main` ← compare your feature branch**. Link the task, describe changes, include evidence (screenshots/tests), and request review from leader. Do not merge until review and checks pass.

Once your PR is merged, start the next task from an updated `main`. Never run `git push --force` on `main`.

## Branch name suggestions

- Student 1: `feature/data-cleaning`
- Student 2: `feature/eda-visualizations`
- Student 3: `feature/book-recommender`
- Student 4: `feature/streamlit-ui`
- Student 5: `feature/integration-tests`

## Avoid conflicts

- Assign ONE primary editor per file.
- Students 3, 4, and 5 agree on function signatures early; see README data/model contract.
- If changing a shared file (`app.py`, `requirements.txt`, `README.md`), notify the owner first.
- Send a PR early enough to leave time to review; don't wait until the deadline.
- Treat `main` as the stable demo branch and feature branches as work-in-progress.

## Required PR checklist

- [ ] My code runs; I tested my changes.
- [ ] I added/updated meaningful tests where relevant.
- [ ] No secrets, private data, or large restricted dataset committed.
- [ ] I updated relevant documentation and AI usage log.
- [ ] My files do not conflict with another student's ownership.

## Team milestone checklist

- [ ] By Nov 23, initial dataset delivered and final column names agreed.
- [ ] By Nov 28, core AI and charts ready to connect.
- [ ] By Nov 30, application demo runs end-to-end.
- [ ] By Dec 2, report, AI log, presentation, and installation steps verified.
