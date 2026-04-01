"""import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')   # 🔥 IMPORTANT (new fix)
nltk.download('stopwords')

# Run only once
nltk.download('punkt')
nltk.download('stopwords')

# Sample resume text
text = "I have experience in Python, Machine Learning and Data Science"

# Convert to lowercase
text = text.lower()

# Tokenization
tokens = word_tokenize(text)

# Load stopwords once (important optimization)
stop_words = set(stopwords.words('english'))

# Remove stopwords + punctuation
filtered = [word for word in tokens if word.isalnum() and word not in stop_words]

print("Tokens:", tokens)
print("Filtered:", filtered) 

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

filtered = [lemmatizer.lemmatize(word) for word in filtered]

print("Final Processed:", filtered)  

from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Python developer with AI skills",
    "Data science and machine learning expert",
    "Looking for AI engineer role"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(docs)

print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", X.toarray()) 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def preprocess(text):
    text = text.lower()
    text = text.replace("ml", "machine learning")
    text = text.replace("ai", "artificial intelligence")
    text = re.sub(r'\W', ' ', text)
    return text


skills = ["python", "machine learning", "data science", "ai"]

resume = "Data scientist with Python and machine learning"
job = "Looking for Python and machine learning engineer"

resume = resume.lower()
job = job.lower()

matched_skills = [skill for skill in skills if skill in resume and skill in job]

skill_score = len(matched_skills) / len(skills)

print("Matched Skills:", matched_skills)
print("Skill Score:", skill_score)
resume = preprocess(resume)
job = preprocess(job)

documents = [resume, job]

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
tfidf = vectorizer.fit_transform(documents)

score = cosine_similarity(tfidf[0:1], tfidf[1:2])

print("Improved Score:", score[0][0])  
 
similarity = 0.3875
skill_score = 0.5

final_score = (0.7 * similarity) + (0.3 * skill_score)

print(final_score)
if final_score > 0.7:
    print("Strong Match - Hire")
elif final_score > 0.4:
    print("Moderate Match - Consider")
else:
    print("Low Match - Reject") 




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = [
    "python developer skilled in ai and machine learning",
    "data scientist with deep learning experience",
    "civil engineer construction expert"
]

jobs = [
    "looking for python developer with ml skills",
    "hiring data scientist with deep learning",
    "construction company needs civil engineer"
]

vectorizer = TfidfVectorizer()

all_text = resumes + jobs
tfidf_matrix = vectorizer.fit_transform(all_text)

resume_vectors = tfidf_matrix[:len(resumes)]
job_vectors = tfidf_matrix[len(resumes):]
for i, res_vec in enumerate(resume_vectors):
    scores = []

    for j, job_vec in enumerate(job_vectors):
        score = cosine_similarity(res_vec, job_vec)[0][0]
        scores.append((j, score))

    # Sort jobs by score (descending)
    ranked_jobs = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nResume {i+1} Best Matches:")

    for rank, (job_index, score) in enumerate(ranked_jobs):
        print(f"Rank {rank+1} → Job {job_index+1} | Score: {score:.2f}") 

skills_list = [
    "python", "machine learning", "deep learning",
    "data science", "ai", "nlp", "tensorflow", "pandas"
]

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return found_skills

# Example (you should replace with yours)
resume = "Python developer with machine learning and AI skills"


job ="Looking for AI engineer with Python and deep learning",
"Hiring data analyst with Excel skills",
"Machine learning engineer with Python required"


res_skills = extract_skills(resume)
job_skills = extract_skills(job)

common = set(res_skills).intersection(set(job_skills))

print("Resume Skills:", res_skills)
print("Job Skills:", job_skills)
print("Matching Skills:", common)
results = match_resume(resume, jobs)

for r in results:
    print(r)

missing = set(job_skills) - set(res_skills)
print("Missing Skills:", missing) 


skill_weights = {
    "python": 5,
    "machine learning": 5,
    "deep learning": 4,
    "ai": 4,
    "data science": 4,
    "nlp": 4,
    "pandas": 3
}

def weighted_score(resume_skills, job_skills):
    score = 0
    total = 0

    for skill in job_skills:
        weight = skill_weights.get(skill, 1)
        total += weight

        if skill in resume_skills:
            score += weight

    return score / total if total != 0 else 0


resume_skills = ['python', 'machine learning', 'ai']
job_skills = ['python', 'deep learning', 'ai']

score = weighted_score(resume_skills, job_skills)
print("Weighted Match Score:", round(score, 2)) """





# =========================
# 1. IMPORTS
# =========================
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 2. DOWNLOAD (RUN ONCE)
# =========================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# 3. TEXT PREPROCESSING
# =========================
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# =========================
# 4. SKILL EXTRACTION
# =========================
skills_list = [
    "python", "machine learning", "deep learning",
    "data science", "ai", "nlp", "tensorflow", "pandas"
]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in skills_list if skill in text]

# =========================
# 5. WEIGHTED SCORING
# =========================
skill_weights = {
    "python": 5,
    "machine learning": 5,
    "deep learning": 4,
    "ai": 4,
    "data science": 4,
    "nlp": 4,
    "pandas": 3
}

def weighted_score(resume_skills, job_skills):
    score, total = 0, 0

    for skill in job_skills:
        weight = skill_weights.get(skill, 1)
        total += weight

        if skill in resume_skills:
            score += weight

    return score / total if total else 0

# =========================
# 6. MAIN PIPELINE FUNCTION
# =========================
def match_resume(resume, jobs):

    # Preprocess
    resume_clean = preprocess(resume)
    jobs_clean = [preprocess(j) for j in jobs]

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean] + jobs_clean)

    cosine_scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # Skills
    res_skills = extract_skills(resume_clean)

    results = []

    for i, job in enumerate(jobs):
        job_skills = extract_skills(jobs_clean[i])

        w_score = weighted_score(res_skills, job_skills)

        final_score = (0.6 * cosine_scores[i]) + (0.4 * w_score)

        results.append({
            "job": job,
            "cosine": round(cosine_scores[i], 2),
            "weighted": round(w_score, 2),
            "final_score": round(final_score, 2),
            "matching_skills": list(set(res_skills) & set(job_skills)),
            "missing_skills": list(set(job_skills) - set(res_skills))
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)

# =========================
# 7. TEST DATA
# =========================
resume = "Python developer with machine learning and AI skills"

jobs = [
    "Looking for AI engineer with Python and deep learning",
    "Hiring data analyst with Excel skills",
    "Machine learning engineer with Python required"
]

# =========================
# 8. RUN PIPELINE
# =========================
results = match_resume(resume, jobs)

for r in results:
    print("\n-----------------------------")
    print("Job:", r["job"])
    print("Cosine Score:", r["cosine"])
    print("Weighted Score:", r["weighted"])
    print("Final Score:", r["final_score"])
    print("Matching Skills:", r["matching_skills"])
    print("Missing Skills:", r["missing_skills"])
    results = match_resume(resume, jobs) 