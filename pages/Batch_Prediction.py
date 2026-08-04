import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import batch_predict

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Batch Spam Prediction",
    page_icon="📂",
    layout="wide"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Loads vectorizer and model once to optimize app performance."""
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            return data.get("vectorizer"), data.get("model")
        elif isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
    return None, None


def get_default_text_column(df: pd.DataFrame) -> str:
    """Auto-detects the column most likely to contain the message text."""
    candidate_keywords = ["message", "text", "sms", "email", "body", "content"]
    
    # 1. Match against known column names
    for col in df.columns:
        if str(col).lower().strip() in candidate_keywords:
            return col

    # 2. Exclude known target/label columns and pick the one with longest average string length
    excluded_names = ["category", "label", "target", "class", "id", "index"]
    candidate_cols = [c for c in df.columns if str(c).lower().strip() not in excluded_names]
    
    if candidate_cols:
        avg_lens = {c: df[c].astype(str).str.len().mean() for c in candidate_cols}
        return max(avg_lens, key=avg_lens.get)
        
    return df.columns[0]


vectorizer, model = load_model()

# -----------------------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------------------
st.title("📂 Batch Spam Prediction")
st.write(
    "Upload a `.csv` file containing SMS or Email messages. "
    "Select the text column to analyze whether each message is **Spam** or **Ham**."
)

st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success("✅ File Uploaded Successfully")
        
        # Dataset Preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("---")
        
        # Select Target Column (with smart default)
        st.subheader("🎯 Column Selection")
        default_column = get_default_text_column(df)
        column_options = list(df.columns)
        default_index = column_options.index(default_column) if default_column in column_options else 0
        
        column = st.selectbox(
            "Choose the column containing the text messages:",
            options=column_options,
            index=default_index,
            help="Select the column with raw message text, not the label or category column."
        )

        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            if not vectorizer or not model:
                st.error("🚨 Model files not found! Ensure 'model.pkl' is in the project root directory.")
            else:
                with st.spinner("Processing messages and running model..."):
                    # Execute prediction pipeline via utils module
                    result = batch_predict(df, column, vectorizer, model)
                    
                    st.success("✅ Batch Prediction Complete!")
                    st.markdown("---")
                    
                    # ---------------------------------------------------------
                    # RESULTS & SUMMARY METRICS
                    # ---------------------------------------------------------
                    st.subheader("📊 Prediction Results")
                    
                    # Standardize prediction labels to 'Spam' and 'Ham'
                    clean_preds = result["Prediction"].astype(str).str.strip().str.title()
                    
                    spam_count = (clean_preds == "Spam").sum()
                    ham_count = (clean_preds == "Ham").sum()
                    
                    # Fallback for numeric labels (1 / 0)
                    if spam_count == 0 and ham_count == 0:
                        spam_count = (result["Prediction"].astype(str) == "1").sum()
                        ham_count = (result["Prediction"].astype(str) == "0").sum()
                    
                    result["Prediction"] = clean_preds
                    total_messages = len(result)
                    
                    # Display Data Frame
                    st.dataframe(result, use_container_width=True)
                    
                    # Key Metrics Cards
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Messages Analyzed", total_messages)
                    m2.metric("Spam Detected 🚨", spam_count)
                    m3.metric("Ham (Safe) ✅", ham_count)

                    st.markdown("---")

                    # ---------------------------------------------------------
                    # VISUALIZATIONS
                    # ---------------------------------------------------------
                    chart_df = pd.DataFrame({
                        "Category": ["Ham", "Spam"],
                        "Count": [ham_count, spam_count]
                    })

                    col1, col2 = st.columns(2)

                    # 1. Pie Chart
                    with col1:
                        st.subheader("Proportion Breakdown")
                        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
                        ax_pie.pie(
                            chart_df["Count"],
                            labels=chart_df["Category"],
                            autopct="%1.1f%%",
                            startangle=90,
                            colors=["#22c55e", "#ef4444"],
                            wedgeprops=dict(width=0.4, edgecolor="white")
                        )
                        ax_pie.axis("equal")
                        st.pyplot(fig_pie)

                    # 2. Bar Chart
                    with col2:
                        st.subheader("Prediction Counts")
                        fig_bar, ax_bar = plt.subplots(figsize=(5, 4.2))
                        
                        sns.barplot(
                            data=chart_df,
                            x="Category",
                            y="Count",
                            hue="Category",
                            palette=["#22c55e", "#ef4444"],
                            ax=ax_bar,
                            legend=False
                        )
                        
                        ax_bar.set_xlabel("")
                        ax_bar.set_ylabel("Message Count")
                        ax_bar.set_title("Spam vs Ham Frequency")

                        # Annotate bars with exact numbers
                        for p in ax_bar.patches:
                            height = int(p.get_height())
                            ax_bar.annotate(
                                f"{height}",
                                (p.get_x() + p.get_width() / 2.0, height),
                                ha="center",
                                va="bottom",
                                fontsize=10,
                                xytext=(0, 3),
                                textcoords="offset points"
                            )

                        plt.tight_layout()
                        st.pyplot(fig_bar)

                    st.markdown("---")

                    # ---------------------------------------------------------
                    # EXPORT RESULTS
                    # ---------------------------------------------------------
                    csv_data = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Predicted CSV",
                        data=csv_data,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"❌ An error occurred while processing the file: {e}")

else:
    st.info("📂 Please upload a CSV file above to begin batch analysis.")

st.markdown("---")
st.caption("Made with ❤️ by Pramod")