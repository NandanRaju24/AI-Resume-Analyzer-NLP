import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text

# ✅ Cache BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

bert_model = load_model()

# ---------------- SKILLS ----------------
skills_list = [
    "python", "machine learning", "deep learning",
    "data science", "ai", "nlp", "tensorflow", "pandas"
]

skill_weights = {
    "python": 5,
    "machine learning": 5,
    "deep learning": 4,
    "ai": 4,
    "data science": 4,
    "nlp": 4,
    "pandas": 3
}

# ---------------- FUNCTIONS ----------------
def extract_text_from_pdf(file):
    return extract_text(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    return text

def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

def weighted_score(resume_skills, job_skills):
    score, total = 0, 0
    for skill in job_skills:
        weight = skill_weights.get(skill, 1)
        total += weight
        if skill in resume_skills:
            score += weight
    return score / total if total else 0

# ✅ NEW: Explanation function
def generate_explanation(matching, missing, score):
    return f"""
This job matches because you have skills like {', '.join(matching)}.
Your match score is {int(score*100)}%.
You can improve by adding {', '.join(missing) if missing else 'no missing skills'}.
"""

# ✅ NEW: Experience extraction
def extract_experience(text):
    match = re.search(r'(\d+)\s+years', text.lower())
    return int(match.group(1)) if match else 0

# ---------------- HYBRID MODEL ----------------
def match_resume_advanced(resume, jobs):
    resume_clean = clean_text(resume)
    jobs_clean = [clean_text(j) for j in jobs]

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean] + jobs_clean)
    cosine_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # BERT
    resume_emb = bert_model.encode([resume_clean])
    job_emb = bert_model.encode(jobs_clean)
    bert_scores = cosine_similarity(resume_emb, job_emb)[0]

    # Skills
    res_skills = extract_skills(resume_clean)

    results = []

    for i, job in enumerate(jobs_clean):
        job_skills = extract_skills(job)
        w_score = weighted_score(res_skills, job_skills)

        # Normalize BERT
        bert_norm = (bert_scores[i] + 1) / 2

        # Final score
        final = (0.4 * cosine_scores[i]) + (0.3 * bert_norm) + (0.3 * w_score)
        final = max(0, min(final, 1))

        matching = list(set(res_skills) & set(job_skills))
        missing = list(set(job_skills) - set(res_skills))

        results.append({
            "job": jobs[i],
            "final_score": round(final, 2),
            "bert": round(bert_scores[i], 2),
            "cosine": round(cosine_scores[i], 2),
            "weighted": round(w_score, 2),
            "matching_skills": matching,
            "missing_skills": missing
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)

# ---------------- UI ----------------
st.title("🚀 AI Resume Analyzer")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

resume = ""

if uploaded_file:
    try:
        resume = extract_text_from_pdf(uploaded_file)
        st.success("Resume uploaded successfully!")
    except:
        st.error("Error reading PDF")

jobs = [
    "Looking for AI engineer with Python and deep learning",
    "Hiring data analyst with Excel skills",
    "Machine learning engineer with Python required"
]

if st.button("Analyze", key="analyze_btn"):

    if resume.strip() == "":
        st.warning("Please upload a resume")
    else:
        # ✅ Experience
        exp = extract_experience(resume)
        st.write(f"🧑‍💼 Experience detected: {exp} years")

        results = match_resume_advanced(resume, jobs)
        best = results[0]

        st.success(f"🏆 BEST MATCH: {best['job']} ({int(best['final_score']*100)}%)")

        st.write("### 🔍 All Job Matches")

        for r in results:
            st.write(f"**Job:** {r['job']}")
            st.write(f"Score: {int(r['final_score']*100)}%")

            score_percent = max(0, min(int(r["final_score"] * 100), 100))
            st.progress(score_percent)

            st.write(f"Matching Skills: {', '.join(r['matching_skills'])}")
            st.write(f"Missing Skills: {', '.join(r['missing_skills'])}")

            # ✅ Explanation added
            explanation = generate_explanation(
                r['matching_skills'],
                r['missing_skills'],
                r['final_score']
            )

            st.write("💡 Why this job suits you:")
            st.write(explanation)

            st.write(f"BERT Score: {r['bert']}")
            st.write(f"TF-IDF Score: {r['cosine']}")
            st.write(f"Skill Score: {r['weighted']}")

            st.write("---")
            #c:/python314/python.exe -m streamlit run app.py
            #& "c:/project ai resume/.venv/Scripts/python.exe" -m streamlit run app.py 