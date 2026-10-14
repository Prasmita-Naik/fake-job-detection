
import pickle
from pathlib import Path

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("fake_job_rnn.keras")
TOKENIZER_PATH = Path("tokenizer.pkl")
SEQ_LEN = 50

# ---------------- PURPLE THEME WITH CONTRASTING COLOURS ----------------

st.markdown("""
<style>

/* Main website background */
.stApp {
    background: linear-gradient(
        135deg,
        #24113F 0%,
        #352052 50%,
        #452766 100%
    );
    color: #F5EDFF;
}

/* Main text */
.stApp p,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stCaptionContainer"] {
    color: #F0E6FF;
}

/* Main headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #F6D98B !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #190B2C 0%,
        #2A1248 100%
    );
    border-right: 1px solid #7851A9;
}

[data-testid="stSidebar"] * {
    color: #F0E6FF;
}

[data-testid="stSidebar"] h2 {
    color: #F6D98B !important;
}

[data-testid="stSidebar"] hr {
    border-color: #7851A9;
}

/* Website header */
.hero {
    padding: 1.8rem 2rem;
    border-radius: 22px;
    color: #FFFFFF;
    background: linear-gradient(
        115deg,
        #6D28A8,
        #8747C7,
        #A56BDD
    );
    border: 1px solid #C5A1F0;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(12, 4, 25, 0.3);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.4rem;
    margin: 0;
}

.hero p {
    color: #F8F0FF !important;
    font-size: 1.05rem;
    margin-bottom: 0;
}

/* Form container */
div[data-testid="stForm"] {
    background: #DCCAF5;
    padding: 1.3rem;
    border: 2px solid #B18AE4;
    border-radius: 18px;
}

/* Text labels inside the form */
div[data-testid="stForm"] label,
div[data-testid="stForm"] p,
div[data-testid="stForm"] [data-testid="stMarkdownContainer"] {
    color: #32134F !important;
    font-weight: 600;
}

/* Input boxes */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
    background-color: #F4ECFF !important;
    color: #28113D !important;
    -webkit-text-fill-color: #28113D !important;
    border: 1px solid #9870C8 !important;
    border-radius: 8px !important;
    caret-color: #5B21B6;
}

/* Placeholder text */
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #735D89 !important;
    -webkit-text-fill-color: #735D89 !important;
    opacity: 1;
}

/* Focused input */
.stApp input:focus,
.stApp textarea:focus {
    border: 2px solid #6D28A8 !important;
    box-shadow: 0 0 0 2px rgba(109, 40, 168, 0.15);
}

/* Submit button */
div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(
        100deg,
        #F0C96D,
        #D9A441
    );
    color: #28113D !important;
    border: 2px solid #F8DFA0;
    border-radius: 12px;
    min-height: 2.8rem;
    font-weight: 750;
}

/* Button text */
div.stButton > button *,
div.stFormSubmitButton > button * {
    color: #28113D !important;
}

/* Button hover */
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #FFE3A0;
    color: #28113D !important;
    border: 2px solid #FFFFFF;
}

/* Analysis result metric */
[data-testid="stMetric"] {
    background: #DCCAF5;
    padding: 1rem;
    border: 2px solid #B18AE4;
    border-radius: 14px;
}

/* Metric label, value and delta */
[data-testid="stMetric"] label,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #45205F !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #35104F !important;
}

/* Safety reminder */
.result {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    background: #DCCAF5;
    color: #32134F !important;
    border: 2px solid #B18AE4;
    margin: .5rem 0;
}

.result b {
    color: #4C1D75;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #E9B949;
}

/* Divider */
.stApp hr {
    border-color: #7851A9;
}

/* Info and warning messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Text visibility in alerts */
[data-testid="stAlert"] p {
    color: #24113F !important;
}

/* Selectable text */
.stApp ::selection {
    background: #D9B8FF;
    color: #24113F;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🔍 Fake Job Detection")
    st.caption("RNN-based job-posting screening")
    st.divider()
    st.write("Enter job details to check the model's prediction.")
    st.caption("Model: SimpleRNN + Embedding")
    st.caption("Text length: 50 tokens")

# ---------------- WEBSITE HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🔍 Fake Job Detection</h1>
    <p>Check a job posting for signs of possible fraud before you apply.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL AND TOKENIZER ----------------

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        return None, None

    model = load_model(MODEL_PATH)

    with TOKENIZER_PATH.open("rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

if model is None or tokenizer is None:
    st.warning(
        "Model files are missing. Place fake_job_rnn.keras "
        "and tokenizer.pkl in the same folder as fakejob.py."
    )
    st.stop()

# ---------------- JOB DETAILS FORM ----------------

st.subheader("📋 Enter Job Posting Details")
st.write("Fill in the available information, then click the button to analyze the posting.")

with st.form("job_check_form"):
    title = st.text_input(
        "Job Title",
        placeholder="e.g., Data Analyst"
    )

    company = st.text_area(
        "Company Profile",
        placeholder="What does the company do?",
        height=90
    )

    location = st.text_input(
        "Job Location",
        placeholder="e.g., Pune, India"
    )

    salary = st.text_input(
        "Salary Range",
        placeholder="e.g., ₹4–6 LPA"
    )

    description = st.text_area(
        "Job Description",
        placeholder="Paste the job responsibilities...",
        height=150
    )

    requirements = st.text_area(
        "Requirements",
        placeholder="Skills, qualifications and experience...",
        height=120
    )

    benefits = st.text_area(
        "Benefits (Optional)",
        placeholder="Benefits and perks...",
        height=80
    )

    industry = st.text_input(
        "Industry (Optional)",
        placeholder="e.g., Information Technology"
    )

    submitted = st.form_submit_button(
        "🔍 Check This Job",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

if submitted:
    combined_text = " ".join([
        title,
        company,
        salary,
        requirements,
        benefits,
        location,
        description,
        industry
    ]).strip()

    if len(combined_text) < 15:
        st.error("Please enter more job-posting information before checking.")

    else:
        sequence = tokenizer.texts_to_sequences([combined_text])
        padded = pad_sequences(sequence, maxlen=SEQ_LEN)

        probabilities = model.predict(padded, verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class]) * 100

        st.divider()
        st.subheader("📊 Analysis Result")

        if predicted_class == 1:
            st.error(
                "⚠️ The model flags this posting as potentially fraudulent."
            )
        else:
            st.success(
                "✅ The model classifies this posting as likely legitimate."
            )

        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.progress(min(max(confidence / 100, 0.0), 1.0))

        st.caption(
            "Confidence is the model's probability for its predicted class. "
            "It does not guarantee that a job posting is safe."
        )

        st.markdown("""
        <div class="result">
            <b>🛡️ Safety Reminder</b><br>
            Never pay application or interview fees. Verify the company
            and recruiter through official channels before sharing personal
            information or accepting an offer.
        </div>
        """, unsafe_allow_html=True)