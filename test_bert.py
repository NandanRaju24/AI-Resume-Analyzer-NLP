from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input
resume = "Python developer with machine learning and AI skills"

jobs = [
    "Looking for AI engineer with Python and deep learning",
    "Hiring data analyst with Excel skills",
    "Machine learning engineer with Python required"
]

# 🔹 Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    return text

resume = clean(resume)
jobs = [clean(j) for j in jobs]

# Convert to embeddings
resume_embedding = model.encode([resume])
job_embeddings = model.encode(jobs)

# Similarity
scores = cosine_similarity(resume_embedding, job_embeddings)[0]

print("BERT Scores:", scores)  