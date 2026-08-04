import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About This Project")

st.markdown("""
## 📩 Spam Detection System

The **Spam Detection System** is a Machine Learning application developed
using **Python** and **Streamlit**. It predicts whether an SMS or Email
message is **Spam** or **Ham (Not Spam)**.

The application provides an easy-to-use interface for both single message
prediction and batch prediction using CSV files.
""")

st.markdown("---")

# -------------------------------------------------
# Features
# -------------------------------------------------

st.header("🚀 Features")

st.markdown("""
- 📩 Spam Message Prediction
- 📂 Batch Prediction using CSV Upload
- 📊 Dataset Analytics
- 📈 Interactive Dashboard
- 📋 Download Prediction Results
- 🤖 Machine Learning Classification
- 🎯 Confidence Score
""")

st.markdown("---")

# -------------------------------------------------
# Dataset
# -------------------------------------------------

st.header("📂 Dataset")

st.write("""
Dataset Used:

- SMS Spam Collection Dataset

Dataset Size:

- **5,572 SMS Messages**

Classes:

- ✅ Ham
- 🚨 Spam
""")

st.markdown("---")

# -------------------------------------------------
# Machine Learning
# -------------------------------------------------

st.header("🤖 Machine Learning Model")

st.write("""
Algorithms Tested

• Naive Bayes

• Logistic Regression

• Random Forest

Final Selected Model

✅ Logistic Regression
""")

st.info("The Logistic Regression model was selected because it provided the best balance between accuracy and spam detection performance.")

st.markdown("---")

# -------------------------------------------------
# Libraries
# -------------------------------------------------

st.header("📚 Python Libraries")

libraries = [
    "Python",
    "Streamlit",
    "Pandas",
    "NumPy",
    "Scikit-Learn",
    "Matplotlib",
    "Seaborn",
    "WordCloud",
    "Pickle"
]

col1, col2 = st.columns(2)

for i, lib in enumerate(libraries):
    if i % 2 == 0:
        col1.write(f"✅ {lib}")
    else:
        col2.write(f"✅ {lib}")

st.markdown("---")

# -------------------------------------------------
# Workflow
# -------------------------------------------------

st.header("⚙️ Project Workflow")

st.markdown("""
1️⃣ Load Dataset

⬇️

2️⃣ Clean Text

⬇️

3️⃣ Convert Text using TF-IDF Vectorizer

⬇️

4️⃣ Predict using Logistic Regression

⬇️

5️⃣ Display Spam/Ham Result

⬇️

6️⃣ Analytics & Dashboard
""")

st.markdown("---")

# -------------------------------------------------
# Project Modules
# -------------------------------------------------

st.header("📁 Project Modules")

modules = [
    "🏠 Home",
    "📩 Spam Predictor",
    "📂 Batch Prediction",
    "📊 Analytics",
    "📈 Dashboard",
    "ℹ️ About"
]

for module in modules:
    st.write(module)

st.markdown("---")

# -------------------------------------------------
# Future Improvements
# -------------------------------------------------

st.header("🔮 Future Enhancements")

future = [
    "Support Email Spam Detection",
    "Real-time Prediction API",
    "Deep Learning Models (LSTM/BERT)",
    "Multi-language Spam Detection",
    "Model Comparison Dashboard",
    "Cloud Deployment"
]

for item in future:
    st.write("✔", item)

st.markdown("---")

# -------------------------------------------------
# Developer
# -------------------------------------------------

st.header("👨‍💻 Developer")

st.success("""
Name : Pramod

Project : Spam Detection System

Frontend : Streamlit

Machine Learning : Scikit-Learn

Language : Python
""")

st.markdown("---")

st.info("Thank you for using the Spam Detection System!")

st.markdown(
"""
<div style="text-align:center; font-size:18px;">
Made with ❤️ by <b>Pramod</b>
</div>
""",
unsafe_allow_html=True
)