import re
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack
from pathlib import Path

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Fake Job Posting Detector")
st.write(
    "Enter a job posting below to check whether the "
    "machine learning model considers it potentially fraudulent."
)

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

MODEL_DIR = Path(__file__).parent / "fake_job_model"

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_DIR / "best_svm.pkl")
    tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    config = joblib.load(MODEL_DIR / "feature_config.pkl")
    return model, tfidf, encoder, config

try:
    model, tfidf, encoder, config = load_model_files()
except Exception as e:
    st.error("Could not load the model files.")
    st.error(str(e))
    st.stop()

categorical_columns = config["categorical_columns"]
numeric_columns = config["numeric_columns"]

with st.form("job_form"):
    st.subheader("Job posting details")

    title = st.text_input("Job title")
    company_profile = st.text_area("Company profile")
    description = st.text_area("Job description")
    requirements = st.text_area("Job requirements")
    benefits = st.text_area("Benefits")

    col1, col2 = st.columns(2)

    with col1:
        location = st.text_input("Location", value="Unknown")
        department = st.text_input("Department", value="Unknown")
        salary_range = st.text_input("Salary range", value="Unknown")
        employment_type = st.selectbox(
            "Employment type",
            ["Unknown", "Full-time", "Part-time",
             "Contract", "Temporary", "Other"]
        )
        required_experience = st.selectbox(
            "Required experience",
            ["Unknown", "Internship", "Entry level",
             "Associate", "Mid-Senior level",
             "Director", "Executive", "Other"]
        )

    with col2:
        required_education = st.selectbox(
            "Required education",
            ["Unknown", "High School or equivalent",
             "Associate Degree", "Bachelor's Degree",
             "Master's Degree", "Doctorate", "Other"]
        )
        industry = st.text_input("Industry", value="Unknown")
        function = st.text_input("Job function", value="Unknown")
        telecommuting = st.selectbox(
            "Telecommuting available?", [0, 1]
        )
        has_company_logo = st.selectbox(
            "Company logo present?", [0, 1]
        )
        has_questions = st.selectbox(
            "Screening questions present?", [0, 1]
        )

    submitted = st.form_submit_button("Analyze job posting")

if submitted:
    if not title.strip() or not description.strip():
        st.warning("Please enter at least a job title and description.")
        st.stop()

    job = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "location": location,
        "department": department,
        "salary_range": salary_range,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions
    }

    job_df = pd.DataFrame([job])

    text_columns = config["text_columns"]
    for col in text_columns:
       job_df[col] = job_df[col].fillna("").astype(str)
       job_df[col] = job_df[col].apply(clean_text)

    job_df["combined_text"] = (
       job_df["title"] + " " +
       job_df["company_profile"] + " " +
       job_df["description"] + " " +
       job_df["requirements"] + " " +
       job_df["benefits"]
    )

    text_features = tfidf.transform(job_df["combined_text"])

    for col in categorical_columns:
        job_df[col] = job_df[col].fillna("Unknown").astype(str)

    for col in numeric_columns:
        job_df[col] = pd.to_numeric(
            job_df[col], errors="coerce"
        ).fillna(0)

    categorical_features = encoder.transform(
        job_df[categorical_columns]
    )

    numeric_features = job_df[numeric_columns].values

    combined_features = hstack([
        text_features,
        categorical_features,
        numeric_features
    ])

    prediction = int(model.predict(combined_features)[0])

    st.subheader("Prediction result")

    if prediction == 1:
        st.error("⚠️ Potentially fraudulent job posting")
        st.write(
            "The model identified patterns associated with fraudulent "
            "postings. Verify the employer and job details independently."
        )
    else:
        st.success("✅ Likely genuine job posting")
        st.write(
            "The model classified this posting as genuine. "
            "This does not guarantee that the job is safe."
        )

    st.caption(
        "This is a machine learning prediction, not proof of fraud "
        "or legitimacy. Do not share sensitive personal information "
        "or pay recruitment fees without independently verifying "
        "the employer."
    )
