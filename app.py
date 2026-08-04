import os
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS LOADER
# -----------------------------------------------------------------------------
def load_css(css_file="style.css"):
    """Injects custom CSS styling if style.css exists."""
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------------------------------------------------
# 3. HOME PAGE VIEW FUNCTION
# -----------------------------------------------------------------------------
def show_home():
    """Renders the Home page content only when Home is selected."""
    st.title("📩 Spam Detection System")
    st.write("Welcome! This application leverages machine learning to classify SMS and Email messages as **Spam** or **Ham** (Legitimate) in real time.")
    st.success("✅ Model Loaded Successfully")

    st.markdown("---")

    # Key Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="Model", value="Logistic Regression")
    c2.metric(label="Accuracy", value="98.2%")
    c3.metric(label="Dataset Size", value="5,572 SMS")
    c4.metric(label="Target Classes", value="Spam / Ham")

    st.markdown("---")

    # Features Overview
    st.subheader("💡 Key Features")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown("""
        * **📩 Spam Prediction:** Test individual text messages dynamically.
        * **📂 Batch Prediction:** Upload `.csv` files for mass analysis.
        * **📊 Analytics:** Explore class distributions and dataset insights.
        """)
        
    with f_col2:
        st.markdown("""
        * **📈 Dashboard:** View high-level metrics and model performance.
        * **ℹ️ About:** Technical breakdown of the machine learning pipeline.
        """)

    st.info("👈 Use the navigation menu on the left sidebar to get started.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888888; font-size: 0.9rem; margin-top: 20px;'>"
        "Made with ❤️ by <b>Pramod</b>"
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 4. NAVIGATION DEFINITION & ROUTING
# -----------------------------------------------------------------------------
home_page = st.Page(show_home, title="Home", icon="🏠", default=True)
spam_predictor = st.Page("pages/Spam_Predictor.py", title="Spam Predictor", icon="📩")
batch_prediction = st.Page("pages/Batch_Prediction.py", title="Batch Prediction", icon="📂")
analytics = st.Page("pages/Analytics.py", title="Analytics", icon="📊")
dashboard = st.Page("pages/Dashboard.py", title="Dashboard", icon="📈")
about = st.Page("pages/About.py", title="About", icon="ℹ️")

# Group pages into sidebar sections
pg = st.navigation({
    "Main App": [home_page, spam_predictor, batch_prediction],
    "Analytics & Info": [analytics, dashboard, about]
})

# --- SIDEBAR FOOTER ---
with st.sidebar:
    st.markdown("---")
    st.markdown("👨‍💻 **Developed by:**")
    st.markdown("### **Thati Pramod**")
    st.caption("© 2026 Spam Detection System")
pg.run()
