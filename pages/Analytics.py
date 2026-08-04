import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    auc,
)

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Dataset & Model Analytics")

# ------------------------
# Load Dataset
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    # Keep first two columns
    df = df.iloc[:, :2]
    df.columns = ["Category", "Message"]
    df["Category"] = df["Category"].str.lower()
    return df

df = load_data()

# ------------------------
# Dataset Preview
# ------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.markdown("---")

# ------------------------
# Dataset Statistics
# ------------------------
spam = len(df[df["Category"] == "spam"])
ham = len(df[df["Category"] == "ham"])
total = len(df)

c1, c2, c3 = st.columns(3)
c1.metric("Total Messages", total)
c2.metric("Spam", spam)
c3.metric("Ham", ham)
st.markdown("---")

# ==========================================
# PAIR 1: Distribution & Pie Chart
# ==========================================
st.subheader("Class Distribution & Percentage")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="Category", data=df, palette=["green", "red"], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_title("Dataset Distribution")
    
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + 0.25, p.get_height() + 10))
        
    st.pyplot(fig, use_container_width=True)

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [ham, spam],
        labels=["Ham", "Spam"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["green", "red"]
    )
    ax.axis("equal")
    ax.set_title("Class Percentage")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ==========================================
# PAIR 2: Message Length
# ==========================================
df["Length"] = df["Message"].apply(len)
st.subheader("Message Length Analysis")

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="Length", hue="Category", bins=40, kde=True, palette=["green", "red"], ax=ax)
    ax.set_title("Message Length Distribution")
    st.pyplot(fig, use_container_width=True)

with col4:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Category", y="Length", data=df, palette=["green", "red"], ax=ax)
    ax.set_title("Average Message Length")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ==========================================
# Word Clouds 
# ==========================================
st.subheader("Word Cloud")
spam_text = " ".join(df[df.Category == "spam"]["Message"])
ham_text = " ".join(df[df.Category == "ham"]["Message"])

col5, col6 = st.columns(2)

with col5:
    st.write("### Spam")
    wc = WordCloud(width=600, height=400, background_color="white").generate(spam_text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

with col6:
    st.write("### Ham")
    wc = WordCloud(width=600, height=400, background_color="white").generate(ham_text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ==========================================
# PAIR 3: Top Words
# ==========================================
st.subheader("Top 20 Words Analysis")
col7, col8 = st.columns(2)

with col7:
    st.write("### Spam")
    spam_words = " ".join(df[df.Category == "spam"]["Message"]).lower().split()
    top_spam = Counter(spam_words).most_common(20)
    words_s = [x[0] for x in top_spam]
    counts_s = [x[1] for x in top_spam]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=counts_s, y=words_s, palette="Reds_r", ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words")
    st.pyplot(fig, use_container_width=True)

with col8:
    st.write("### Ham")
    ham_words = " ".join(df[df.Category == "ham"]["Message"]).lower().split()
    top_ham = Counter(ham_words).most_common(20)
    words_h = [x[0] for x in top_ham]
    counts_h = [x[1] for x in top_ham]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=counts_h, y=words_h, palette="Greens_r", ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Words")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")

# ==========================================
# PAIR 4: Model Evaluations & Scores
# ==========================================
st.title("🤖 Model Evaluation & Metrics")

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)
    text = re.sub(r"[^a-zA-Z0-9\s!$%]", "", text)
    return text.lower().strip()

@st.cache_data(show_spinner=False)
def run_model_evaluations(df):
    """Trains the models dynamically and returns strictly evaluation data (no large model objects)."""
    # 1. Clean & Prepare Data
    df_eval = df.copy()
    df_eval["Category"] = df_eval["Category"].map({"ham": 0, "spam": 1})
    df_eval = df_eval.dropna()
    df_eval["clean_message"] = df_eval["Message"].apply(clean_text)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df_eval["clean_message"], df_eval["Category"],
        test_size=0.3, random_state=42, stratify=df_eval["Category"]
    )

    # 3. Vectorization
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Models Definition
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    }

    metrics_list = []
    roc_data = {}
    cm_best = None

    # 5. Train & Evaluate
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        prob = model.predict_proba(X_test_vec)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        mcc = matthews_corrcoef(y_test, pred)

        metrics_list.append({
            "Model": name, 
            "Accuracy": acc, 
            "Precision": prec, 
            "Recall": rec, 
            "F1-Score": f1, 
            "MCC": mcc
        })

        # Calculate ROC details
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

        # Confusion Matrix for Logistic Regression (Selected as Best in train.py)
        if name == "Logistic Regression":
            cm_best = confusion_matrix(y_test, pred)

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    return metrics_df, roc_data, cm_best

with st.spinner("Generating Model Evaluation Metrics..."):
    metrics_df, roc_data, cm_best = run_model_evaluations(df)

st.subheader("Performance Metrics Overview")
# Highlight the max values in the table to show which model performs best
st.dataframe(
    metrics_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"),
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)
col9, col10 = st.columns(2)

# Plot ROC Curves
with col9:
    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"Naive Bayes": "blue", "Logistic Regression": "green", "Random Forest": "orange"}
    
    for name, data in roc_data.items():
        ax.plot(
            data["fpr"], data["tpr"], 
            label=f"{name} (AUC = {data['auc']:.3f})", 
            color=colors[name]
        )

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig, use_container_width=True)

# Plot Confusion Matrix
with col10:
    st.subheader("Confusion Matrix (Logistic Regression)")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_best, annot=True, fmt="d", cmap="Blues", 
        xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ by Pramod")