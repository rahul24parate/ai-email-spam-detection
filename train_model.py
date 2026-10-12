"""
Train and evaluate an email/SMS spam classifier.

Dataset: UCI SMS Spam Collection, downloaded from the public UCI repository.
The script saves a trained scikit-learn Pipeline to spam_model.joblib.
"""
from pathlib import Path
import io
import zipfile
import urllib.request

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_FILE = HERE / "spam_model.joblib"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def get_dataset():
    if DATA_FILE.exists():
        return
    print("Downloading the UCI SMS Spam Collection dataset...")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=30) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Dataset download failed. Check your internet connection and try again."
        ) from exc

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith("SMSSpamCollection")),
            None,
        )
        if member is None:
            raise RuntimeError("Could not find the dataset file inside the archive.")
        DATA_FILE.write_bytes(archive.read(member))


def main():
    get_dataset()
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )
    df = df.dropna().drop_duplicates(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["label"] = df["label"].map({"ham": "not spam", "spam": "spam"})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Rows used: {len(df)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=3))
    print("Confusion matrix (labels: not spam, spam):")
    print(confusion_matrix(y_test, predictions, labels=["not spam", "spam"]))

    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved model to: {MODEL_FILE}")
    print("Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
