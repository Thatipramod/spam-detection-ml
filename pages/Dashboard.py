import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Spam Detection Dashboard")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["Category", "Message"]
    df["Category"] = df["Category"].str.lower()
    df["Message_Length"] = df["Message"].apply(len)
    df["Word_Count"] = df["Message"].apply(lambda x: len(str(x).split()))
    return df

df = load_data()

# -----------------------------
# Statistics
# -----------------------------
total = len(df)
spam = len(df[df["Category"] == "spam"])
ham = len(df[df["Category"] == "ham"])

spam_percent = (spam / total) * 100
ham_percent = (ham / total) * 100

avg_length = int(df["Message_Length"].mean())
avg_words = int(df["Word_Count"].mean())

# -----------------------------
# Dashboard Cards
# -----------------------------
st.subheader("Overview")

c1, c2, c3 = st.columns(3)

c1.metric("📩 Total Messages", total)
c2.metric("🚨 Spam Messages", spam)
c3.metric("✅ Ham Messages", ham)

c4, c5, c6 = st.columns(3)

c4.metric("📊 Spam %", f"{spam_percent:.2f}%")
c5.metric("📊 Ham %", f"{ham_percent:.2f}%")
c6.metric("🤖 Model", "Logistic Regression")

st.markdown("---")

c7, c8 = st.columns(2)

c7.metric("📝 Avg Message Length", avg_length)
c8.metric("📖 Avg Words / Message", avg_words)

st.markdown("---")

# -----------------------------
# Count Plot
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("Spam vs Ham")

    fig, ax = plt.subplots(figsize=(6,5))

    sns.countplot(
        x="Category",
        data=df,
        palette=["green","red"],
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("Messages")

    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x()+0.25,p.get_height()+8)
        )

    st.pyplot(fig)

# -----------------------------
# Pie Chart
# -----------------------------
with col2:

    st.subheader("Dataset Composition")

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        [ham, spam],
        labels=["Ham", "Spam"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["green","red"]
    )

    ax.axis("equal")

    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Message Length
# -----------------------------
st.subheader("Message Length Distribution")

fig, ax = plt.subplots(figsize=(10,5))

sns.histplot(
    data=df,
    x="Message_Length",
    hue="Category",
    bins=35,
    kde=True,
    ax=ax
)

ax.set_xlabel("Characters")
ax.set_ylabel("Frequency")

st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Average Length
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("Average Message Length")

    fig, ax = plt.subplots(figsize=(6,5))

    sns.barplot(
        x="Category",
        y="Message_Length",
        data=df,
        palette=["green","red"],
        ax=ax
    )

    st.pyplot(fig)

with col2:

    st.subheader("Average Word Count")

    fig, ax = plt.subplots(figsize=(6,5))

    sns.barplot(
        x="Category",
        y="Word_Count",
        data=df,
        palette=["green","red"],
        ax=ax
    )

    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Dataset Summary
# -----------------------------
st.subheader("Dataset Summary")

summary = pd.DataFrame({
    "Metric":[
        "Total Messages",
        "Spam Messages",
        "Ham Messages",
        "Spam Percentage",
        "Ham Percentage",
        "Average Length",
        "Average Words"
    ],
    "Value":[
        total,
        spam,
        ham,
        f"{spam_percent:.2f}%",
        f"{ham_percent:.2f}%",
        avg_length,
        avg_words
    ]
})

st.dataframe(summary, use_container_width=True)

st.markdown("---")

st.success("✔ Dashboard Loaded Successfully")

st.caption("Made with ❤️ by Pramod")