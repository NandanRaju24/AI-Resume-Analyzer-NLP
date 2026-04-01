## Demo
# AI-Resume-Analyzer-NLP
Run the app:
streamlit run app.py

import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
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

def match_resume(resume, jobs):
    resume_clean = clean_text(resume)
    jobs_clean = [clean_text(j) for j in jobs]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean] + jobs_clean)

    cosine_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    res_skills = extract_skills(resume_clean)

    results = []

    for i, job in enumerate(jobs):
        job_skills = extract_skills(jobs_clean[i])
        w_score = weighted_score(res_skills, job_skills)

        final_score = (0.6 * cosine_scores[i]) + (0.4 * w_score)

        results.append({
            "job": job,
            "final_score": round(final_score, 2),
            "matching_skills": list(set(res_skills) & set(job_skills)),
            "missing_skills": list(set(job_skills) - set(res_skills))
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)

st.title("AI Resume Analyzer")

resume = st.text_area("Enter Resume")

jobs = [
    "Looking for AI engineer with Python and deep learning",
    "Hiring data analyst with Excel skills",
    "Machine learning engineer with Python required"
]

if st.button("Analyze"):
    results = match_resume(resume, jobs)

    for r in results:
        st.write("### Job:", r["job"])
        st.write("Final Score:", r["final_score"])
        st.write("Matching Skills:", ", ".join(r["matching_skills"]))
        st.write("Missing Skills:", ", ".join(r["missing_skills"]))
        st.progress(int(r["final_score"] * 100))
        st.write("---")    

