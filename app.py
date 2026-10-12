import streamlit as st
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "spam_model.joblib"

st.set_page_config(page_title="AI Email Spam Detector", page_icon="📩", layout="centered")
st.title("📩 AI-Powered Email Spam Detector")
st.write(
    "Enter an email or text message. The machine-learning model predicts whether it "
    "looks like spam or not spam. This is a demo, so do not use it as the only basis "
    "for security decisions."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

message = st.text_area(
    "Email or message text",
    placeholder="Example: Congratulations! You have won a prize. Click here to claim it.",
    height=160
)

if st.button("Check message", type="primary"):
    if not message.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        spam_index = classes.index("spam")
        spam_probability = float(probabilities[spam_index])

        if prediction == "spam":
            st.error("Prediction: Likely SPAM")
        else:
            st.success("Prediction: Likely NOT SPAM")

        st.metric("Model-estimated spam probability", f"{spam_probability:.1%}")
        st.caption(
            "The probability is the model's estimate, not a guarantee. "
            "Messages can be misclassified, especially if they differ from the training data."
        )

with st.expander("Try example messages"):
    st.code("Congratulations! You won a free prize. Call now to claim your reward.")
    st.code("Hi, are we still meeting at the library at 4 pm?")
