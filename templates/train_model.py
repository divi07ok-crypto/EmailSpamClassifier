
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

# File paths
DATASET_PATH = os.path.join("dataset", "spam.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("Loading dataset...")

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return

    # Load SMS Spam Collection dataset
    data = pd.read_csv(
        DATASET_PATH,
        encoding="latin-1"
    )

    # Support the standard Kaggle/UCI column names
    if "v1" in data.columns and "v2" in data.columns:
        data = data[["v1", "v2"]].copy()
        data.columns = ["label", "message"]
    elif "label" in data.columns and "message" in data.columns:
        data = data[["label", "message"]].copy()
    else:
        print("ERROR: Dataset must contain v1/v2 or label/message columns.")
        print("Available columns:", list(data.columns))
        return

    data.dropna(subset=["label", "message"], inplace=True)
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str)

    data = data[data["label"].isin(["ham", "spam"])].copy()
    data["target"] = data["label"].map({"ham": 0, "spam": 1})

    if data.empty or data["target"].nunique() < 2:
        print("ERROR: Dataset must contain both ham and spam messages.")
        return

    X = data["message"]
    y = data["target"]

    print(f"Total messages: {len(data)}")
    print("\nSplitting dataset into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Compare three machine-learning algorithms
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "SVM": SVC(
            kernel="linear",
            probability=True,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    print("\nTraining and comparing models...")

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=5000,
                    ngram_range=(1, 2)
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = pipeline

        print(f"Accuracy : {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall   : {recall * 100:.2f}%")
        print(f"F1 Score : {f1 * 100:.2f}%")

    # Save model comparison results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    ))

    # Select the model with the highest spam F1 score
    best_row = results_df.sort_values(
        by=["F1 Score", "Recall", "Precision"],
        ascending=False
    ).iloc[0]

    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    best_predictions = best_model.predict(X_test)

    # Save the selected model for the existing Flask website
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model: {best_name}")
    print(f"Model saved successfully: {MODEL_PATH}")

    print("\nClassification Report for Best Model:")
    print(classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["Ham (Not Spam)", "Spam"],
        zero_division=0
    ))

    # Create model comparison chart
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    chart_data = results_df.set_index("Model")[metrics]

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1.05),
        rot=0
    )
    ax.set_title("Email Spam Classifier - Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Model")
    ax.legend(loc="lower right")
    plt.tight_layout()

    chart_path = os.path.join(MODEL_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    # Create confusion matrix for the selected model
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        labels=[0, 1],
        display_labels=["Ham", "Spam"],
        values_format="d"
    )
    display.ax_.set_title(f"Confusion Matrix - {best_name}")

    matrix_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=150)
    plt.close()

    print("\nGenerated files:")
    print(f"1. {results_path}")
    print(f"2. {chart_path}")
    print(f"3. {matrix_path}")
    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
