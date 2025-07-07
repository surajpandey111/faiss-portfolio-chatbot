from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("❌ GEMINI_API_KEY is missing in .env")

genai.configure(api_key=api_key)

# Load data
with open("portfolio_data.txt") as f:
    docs = [line.strip() for line in f if line.strip()]

vectorizer = TfidfVectorizer().fit(docs)
doc_vectors = vectorizer.transform(docs)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ TFIDF+Gemini server running. POST to /search"

@app.route("/search", methods=["POST"])
def search():
    q = request.json.get("question")
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, doc_vectors).flatten()
    top_idx = np.argsort(sims)[-4:][::-1]
    context = "\n".join([docs[i] for i in top_idx])

    prompt = f"Answer the question from this context:\n{context}\n\nQ: {q}"

    model = genai.GenerativeModel("gemini-2.0-flash")
    answer = model.generate_content(prompt)

    return jsonify({"answer": answer.text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
