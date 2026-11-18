# AURA — Student Performance Intelligence

A premium dark Streamlit dashboard for learning a supervised machine-learning workflow with a Scikit-Learn Decision Tree classifier.

## Features
- Interactive student profile prediction
- Adjustable Decision Tree depth and test split
- Accuracy, precision, recall and F1 score
- Confusion matrix and classification report
- Feature importance chart
- Dataset explorer and CSV download

## Run locally (macOS)

```bash
cd student_performance_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If the `streamlit` command is not found, use:

```bash
python3 -m streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Optional: run the terminal training script

```bash
python3 train.py
```

This creates `dataset.csv` in the current directory and prints evaluation metrics.

## Deploy free with Streamlit Community Cloud

1. Create or sign in to GitHub.
2. Create a new public repository, e.g. `student-performance-ml`.
3. Upload `app.py`, `train.py`, `requirements.txt`, and `README.md`.
   - `dataset.csv` is optional because the dashboard generates its own reproducible synthetic dataset.
   - Do not upload `.venv/`.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select your repository and branch.
6. Set **Main file path** to `app.py`.
7. Select **Deploy** and wait for the build to finish.

When you update files on GitHub, Streamlit Community Cloud will rebuild the app.

## Project structure

```text
student_performance_ml/
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## Important note

The dataset is synthetic and the labels are generated from a simplified formula. This is a learning demo, not a validated educational assessment tool. Do not use predictions to make decisions about real students.
