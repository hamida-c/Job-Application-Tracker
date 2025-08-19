import streamlit as st
import joblib
import pandas as pd
from datetime import date
from db import insert_job_application, get_all_applications, update_status, delete_application
import plotly.graph_objects as go
import spacy
import PyPDF2
from sentence_transformers import SentenceTransformer, util

# ---------------- LOAD MODELS ----------------
model = joblib.load("interview_predictor.pkl")   # <-- change path if needed
feature_cols = joblib.load("feature_columns.pkl")

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- UTILITIES ----------------
def predict_interview(company, role, website, status, notes):
    input_data = pd.DataFrame([{
        "company_name": company,
        "job_role": role,
        "website": website,
        "status": status,
        "notes": notes
    }])
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
    proba = model.predict_proba(input_encoded)[0][1]
    return proba

SKILLS_DB = [
    "Python", "R", "SQL", "Excel", "Tableau", "Power BI",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "Keras", "PyTorch", "Scikit-learn",
    "Data Analysis", "Data Visualization", "Statistics", "Probability",
    "AWS", "Azure", "GCP", "BigQuery", "Hadoop", "Spark",
    "Communication", "Teamwork", "Problem Solving", "Critical Thinking"
]

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def find_missing_skills(resume_text, jd_text, skills_db):
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    missing = []
    for skill in skills_db:
        if skill.lower() in jd_lower and skill.lower() not in resume_lower:
            missing.append(skill)
    return missing

def compute_match_score(resume_text, jd_text):
    embeddings = embedder.encode([resume_text, jd_text], convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    jd_skills = [s.lower() for s in SKILLS_DB if s.lower() in jd_text.lower()]
    resume_skills = [s.lower() for s in SKILLS_DB if s.lower() in resume_text.lower()]
    overlap = len(set(jd_skills) & set(resume_skills))
    skill_overlap_score = overlap / len(jd_skills) if jd_skills else 0
    final_score = (0.7 * semantic_similarity + 0.3 * skill_overlap_score) * 100
    return round(final_score, 2)

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Job Application Tracker", layout="centered")
st.title("💼 Job Application Tracker")

# Sidebar navigation
page = st.sidebar.radio(
    "Choose Feature",
    ["📄 Job Application Tracker", "📂 View & Update Applications", "📑 Resume–JD Match"]
)

# ---------------- PAGE 1: Job Application Tracker ----------------
if page == "📄 Job Application Tracker":
    with st.container():
        st.markdown(
            """
            <div style="background-color:#f9f9f9;padding:20px;border-radius:15px;box-shadow:0px 4px 8px rgba(0,0,0,0.1);">
                <h3 style="color:#333;">📝 Log a New Application</h3>
            </div>
            """, unsafe_allow_html=True
        )

        with st.form("job_form"):
            company = st.text_input("🏢 Company Name")
            role = st.selectbox("💼 Job Role", [
                "AI Engineer", "Data Scientist", "Data Analyst", "Business Analyst",
                "ML Engineer", "Research Scientist", "Deep Learning Engineer",
                "Computer Vision Engineer", "NLP Engineer", "Other"
            ])
            applied_date = st.date_input("📅 Applied Date", value=date.today())
            website = st.selectbox("🌐 Applied Website", [
                "LinkedIn", "Indeed", "Naukri", "Glassdoor", "Internshala",
                "Hirect", "Company Website", "AngelList", "Other"
            ])
            status = st.selectbox("📌 Status", [
                "Applied", "Interview", "Offer", "Rejected", "Followed-Up", "Other"
            ])
            notes = st.text_area("📝 Notes (Optional)", height=100)

            submitted = st.form_submit_button("🚀 Save Application")
            if submitted:
                if not company or not role or not website:
                    st.error("⚠️ Please fill all the fields.")
                else:
                    insert_job_application(company, role, applied_date, website, status, notes)
                    st.success("✅ Job application added successfully!")

                    # Prediction
                    proba = predict_interview(company, role, website, status, notes)
                    probability = round(proba * 100, 0)

                    fig = go.Figure(go.Pie(
                        values=[probability, 100 - probability],
                        hole=0.7,
                        marker=dict(colors=['#27c5ed', 'lightgray'], line=dict(color='white', width=2)),
                        textinfo='none'
                    ))

                    fig.update_layout(
                        width=400, height=400, showlegend=False,
                        margin=dict(t=50, b=0, l=0, r=0),
                        annotations=[dict(
                            text=f'{probability}%', x=0.5, y=0.5,
                            font_size=40, showarrow=False
                        )],
                        title_text="Chance of getting an Interview"
                    )
                    st.plotly_chart(fig)

# ---------------- PAGE 2: View & Update Applications ----------------
elif page == "📂 View & Update Applications":
    st.subheader("📂 Your Applications")
    df = get_all_applications()

    if not df.empty:
        for _, row in df.iterrows():
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color:#ffffff;padding:20px;margin:10px 0;
                                border-radius:12px;box-shadow:0px 4px 8px rgba(0,0,0,0.08);">
                        <h4 style="margin:0;color:#444;">🏢 {row['company_name']} - {row['job_role']}</h4>
                        <p style="margin:5px 0;color:#666;">📅 {row['applied_date']} | 🌐 {row['website']}</p>
                        <p style="margin:5px 0;"><b>Status:</b> {row['status']}</p>
                        <p style="margin:5px 0;color:#555;">📝 {row['notes'] or 'No notes added.'}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    new_status = st.selectbox(
                        f"Update Status (ID {row['id']})",
                        ["Applied", "Interview", "Offer", "Rejected", "Followed-Up", "Other"],
                        index=["Applied", "Interview", "Offer", "Rejected", "Followed-Up", "Other"].index(row["status"]),
                        key=f"status_{row['id']}"
                    )
                with col2:
                    if st.button("✅ Update", key=f"update_{row['id']}"):
                        update_status(row["id"], new_status)
                        st.success("Status updated!")
                        st.rerun()
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{row['id']}"):
                        delete_application(row["id"])
                        st.success("Application deleted!")
                        st.rerun()
    else:
        st.info("No applications found yet.")

# ---------------- PAGE 3: Resume-JD Match ----------------
elif page == "📑 Resume–JD Match":
    resume_file = st.file_uploader("Upload Your Resume (text or PDF)", type=["txt", "pdf"])
    jd_text = st.text_area("Paste Job Description", height=200)

    if resume_file and jd_text:
        if resume_file.name.endswith(".txt"):
            resume_text = resume_file.read().decode("utf-8")
        else:
            resume_text = extract_text_from_pdf(resume_file)

        match_score = compute_match_score(resume_text, jd_text)

        st.subheader("📊 Match Score")
        st.success(f"Your resume matches this job description by **{match_score}%**")

        missing_skills = find_missing_skills(resume_text, jd_text, SKILLS_DB)

        if missing_skills:
            st.warning("🚀 Suggested Skills to Add:")
            st.write(", ".join(missing_skills))
        else:
            st.info("✅ Your resume already covers most key skills for this role!")


























