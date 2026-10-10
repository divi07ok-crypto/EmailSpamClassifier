
import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

DATASET_PATH = os.path.join("dataset", "spam.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("Loading dataset...")

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH, encoding="latin-1")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]].copy()
    else:
        print("ERROR: Dataset needs v1/v2 or label/message columns.")
        return

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    if df.empty or df["target"].nunique() != 2:
        print("ERROR: Dataset must contain both ham and spam.")
        return

    X = df["message"].astype(str)
    y = df["target"].astype(int)

    print(f"Total messages: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "SVM": SVC(
            kernel="linear", probability=True, random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2)
            )),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1 Score": "{:.4f}".format
    }))

    results_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    best_name = results_df.sort_values(
        ["F1 Score", "Recall", "Precision"],
        ascending=False
    ).iloc[0]["Model"]

    best_model = trained_models[best_name]
    predictions = best_model.predict(X_test)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model: {best_name}")
    print(f"Model saved: {MODEL_PATH}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Ham (Not Spam)", "Spam"],
        zero_division=0
    ))

    # 1. Model comparison chart
    ax = results_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ].plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05), rot=0)

    ax.set_title("Email Spam Classifier - Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Model")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150
    )
    plt.close()

    # 2. Confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        labels=[0, 1],
        display_labels=["Ham", "Spam"],
        values_format="d"
    )
    display.ax_.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150
    )
    plt.close()

    # 3. Spam-indicating keyword chart
    # Use a separate logistic regression model for word explanations.
    print("\nGenerating spam keyword visualization...")

    keyword_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42
        ))
    ])

    keyword_model.fit(X_train, y_train)

    vectorizer = keyword_model.named_steps["tfidf"]
    classifier = keyword_model.named_steps["classifier"]

    words = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]

    keyword_df = pd.DataFrame({
        "Keyword": words,
        "Spam score": weights
    })

    spam_keywords = keyword_df.sort_values(
        "Spam score", ascending=False
    ).head(15)

    spam_keywords.to_csv(
        os.path.join(MODEL_DIR, "spam_keywords.csv"),
        index=False
    )

    ax = spam_keywords.sort_values("Spam score").plot(
        kind="barh",
        x="Keyword",
        y="Spam score",
        legend=False,
        figsize=(9, 7)
    )
    ax.set_title("Top 15 Spam-Indicating Keywords")
    ax.set_xlabel("Spam association score")
    ax.set_ylabel("Keyword")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODEL_DIR, "spam_keywords.png"), dpi=150
    )
    plt.close()

    print("\nGenerated files:")
    print("model/model_comparison.csv")
    print("model/model_comparison.png")
    print("model/confusion_matrix.png")
    print("model/spam_keywords.csv")
    print("model/spam_keywords.png")
    print("\nTraining and visualization completed successfully!")


if __name__ == "__main__":
    main()
