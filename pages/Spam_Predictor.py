import os
import pickle
import streamlit as st
from utils import predict_message

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished UI elements
st.markdown(
    """
    <style>
    /* Global Card & Container Styling */
    .stTextArea textarea {
        font-size: 15px !important;
        border-radius: 10px !important;
    }
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    /* Metric & Badge Customization */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. MODEL LOADING & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_classifier():
    """Loads and caches the trained model and TF-IDF vectorizer."""
    model_path = "model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            return data.get("vectorizer"), data.get("model")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    return None, None


vectorizer, model = load_classifier()


# -----------------------------------------------------------------------------
# 3. STATE MANAGEMENT & CALLBACKS
# -----------------------------------------------------------------------------
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""


def clear_text_callback():
    st.session_state["user_text"] = ""


def set_sample_text(text: str):
    st.session_state["user_text"] = text


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & APP INFO
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/isometric/100/security-configuration.png",
        width=80,
    )
    st.title("🛡️ Spam Shield AI")
    st.caption("Real-time SMS & Email Classification")

    st.markdown("---")

    st.subheader("📌 System Status")
    if vectorizer and model:
        st.success("Model loaded & operational", icon="✅")
    else:
        st.error("Model file (`model.pkl`) missing!", icon="🚨")

    st.markdown("---")

    st.subheader("💡 How it works")
    st.markdown(
        """
        1. **Paste text** into the input box or pick a sample.
        2. Click **Analyze Message**.
        3. The NLP model evaluates features via **TF-IDF Vectorization**.
        4. Receive instantaneous classification & confidence score.
        """
    )

    st.markdown("---")
    st.caption("Developed by **Pramod** • Built with Streamlit & Scikit-Learn")


# -----------------------------------------------------------------------------
# 5. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📩 Message Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Detect fraudulent scam messages, phishing attempts, and unwanted spam with high precision.</div>',
    unsafe_allow_html=True,
)

# Main Input Section inside a modern bordered card
with st.container(border=True):
    message_input = st.text_area(
        "Enter SMS or Email Content:",
        key="user_text",
        height=180,
        placeholder="e.g., Urgent! Your mobile number has won a $2,000 gift card. Claim now at http://bit.ly/fake-link",
    )

    col_btn1, col_btn2, _ = st.columns([1.5, 1, 3])

    with col_btn1:
        predict_btn = st.button("🚀 Analyze Message", use_container_width=True, type="primary")
    with col_btn2:
        st.button("🗑️ Clear Input", use_container_width=True, on_click=clear_text_callback)


# -----------------------------------------------------------------------------
# 6. PREDICTION & RESULTS DISPLAY
# -----------------------------------------------------------------------------
if predict_btn:
    if not vectorizer or not model:
        st.error("🚨 Model unavailable. Please make sure `model.pkl` exists in your root directory.")
    elif not message_input.strip():
        st.warning("⚠️ Please provide a message to analyze before submitting.")
    else:
        # Generate prediction via utils module
        prediction, confidence = predict_message(message_input, vectorizer, model)

        st.markdown("<br>", unsafe_allow_html=True)

        # Result Banner & Metrics Container
        with st.container(border=True):
            st.subheader("📊 Analysis Results")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if prediction == "Spam":
                    st.error("🚨 **SPAM DETECTED**", icon="🚨")
                    st.metric(label="Classification", value="SPAM", delta="High Risk", delta_color="inverse")
                else:
                    st.success("✅ **SAFE (HAM)**", icon="✅")
                    st.metric(label="Classification", value="HAM", delta="Legitimate", delta_color="normal")

            with res_col2:
                st.write("**Model Confidence Score:**")
                st.progress(int(confidence) / 100)
                st.markdown(f"**`{confidence:.2f}%`** certainty score based on feature extraction.")

                if prediction == "Spam":
                    st.caption("⚠️ **Advice:** Do not click on any embedded links or share sensitive details.")
                else:
                    st.caption("👍 **Advice:** This message appears to be safe and legitimate.")


samples = {
    "💬 Legitimate Message": "Hey, are we still meeting for lunch today around 1 PM?",
    "🎁 Prize Scam": "Congratulations! You've been selected to win a $1000 Walmart Gift Card. Claim now at http://bit.ly/claim-prize",
    "🏦 Banking Phishing": "ALERT: Your bank account is locked due to suspicious activity. Verify credentials immediately: www.bank-sec-verify.com",
    "🎰 Cash Lottery": "WIN CASH NOW! Reply YES to enter the daily $5000 sweepstakes. Terms apply.",
    "📌 Friendly Reminder": "Don't forget to submit the project documentation before 5 PM tomorrow."
}

tabs = st.tabs(list(samples.keys()))

for idx, (title, sample_text) in enumerate(samples.items()):
    with tabs[idx]:
        st.info(f"\"{sample_text}\"")
        st.button(
            f"Use this sample", 
            key=f"sample_btn_{idx}", 
            on_click=set_sample_text, 
            args=(sample_text,)
        )


# -----------------------------------------------------------------------------
# 8. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888888; font-size: 0.85rem;'>"
    "Made with ❤️ by <b>Pramod</b> | Spam Shield AI Dashboard"
    "</div>",
    unsafe_allow_html=True,
)