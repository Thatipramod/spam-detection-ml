import os
import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# -------------------------------------------------------------
# Create folder for plots
# -------------------------------------------------------------
os.makedirs("plots", exist_ok=True)


# -------------------------------------------------------------
# Text Cleaning
# -------------------------------------------------------------
def clean_text(text):
    text = str(text)

    # Replace URLs with URL token
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)

    # Keep letters, numbers, spaces and spam symbols
    text = re.sub(r"[^a-zA-Z0-9\s!$%]", "", text)

    return text.lower()


# -------------------------------------------------------------
# Class Distribution Plot
# -------------------------------------------------------------
def plot_class_distribution(df):
    plt.figure(figsize=(6, 5))

    ax = sns.countplot(
        x="Category",
        hue="Category",
        data=df,
        palette=["green", "red"],
        legend=False
    )

    plt.title("Dataset Class Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"])

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                str(int(height)),
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom"
            )

    plt.tight_layout()
    plt.savefig("plots/class_distribution.png")
    plt.close()


# -------------------------------------------------------------
# Accuracy Comparison
# -------------------------------------------------------------
def plot_accuracy(results):
    plt.figure(figsize=(8, 5))

    models = list(results.keys())
    acc = [results[m]["accuracy"] * 100 for m in models]

    plt.bar(models, acc, color=["blue", "green", "orange"])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 105)

    for i, v in enumerate(acc):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center")

    plt.tight_layout()
    plt.savefig("plots/model_accuracy.png")
    plt.close()


# -------------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------------
def plot_cm(y_test, pred):
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Logistic Regression)")

    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()


# -------------------------------------------------------------
# ROC Curve
# -------------------------------------------------------------
def plot_roc(results, X_test, y_test):
    plt.figure(figsize=(8, 6))

    colors = {
        "Naive Bayes": "blue",
        "Logistic Regression": "green",
        "Random Forest": "orange"
    }

    for name, data in results.items():
        model = data["model"]
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} ({roc_auc:.3f})",
            color=colors[name]
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/roc_curve.png")
    plt.close()


# -------------------------------------------------------------
# Train Model
# -------------------------------------------------------------
def train():
    print("Loading dataset...")

    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["Category", "Message"]

    df["Category"] = (
        df["Category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"ham": 0, "spam": 1})
    )

    df = df.dropna()

    print("Dataset Loaded Successfully")
    print(df["Category"].value_counts())

    plot_class_distribution(df)

    print("Cleaning Text...")
    df["clean_message"] = df["Message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["Category"],
        test_size=0.3,
        random_state=42,
        stratify=df["Category"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    algorithms = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )
    }

    results = {}

    print("\nTraining Models...\n")

    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "prediction": pred
        }

        print("=" * 50)
        print(name)
        print(f"Accuracy : {acc * 100:.2f}%")
        print(classification_report(y_test, pred))

    # Best Model
    best_model = algorithms["Logistic Regression"]
    print("\nBest Model : Logistic Regression")

    plot_accuracy(results)
    plot_cm(y_test, results["Logistic Regression"]["prediction"])
    plot_roc(results, X_test, y_test)

    # Save Model
    model_data = {
        "vectorizer": vectorizer,
        "model": best_model
    }

    with open("model.pkl", "wb") as file:
        pickle.dump(model_data, file)

    print("\n✅ model.pkl saved successfully!")
    print("✅ All plots saved inside the 'plots/' folder successfully!")

    # Test Prediction
    test_msg = "Congratulations! You won $5000. Click the link to claim."
    cleaned = clean_text(test_msg)
    vec = vectorizer.transform([cleaned])
    pred = best_model.predict(vec)[0]

    print("\nTest Prediction :", "SPAM" if pred == 1 else "HAM")


if __name__ == "__main__":
    train()