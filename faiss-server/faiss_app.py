from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------
# SAFE ENV LOADING
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("⚠️ GEMINI_API_KEY not found. Gemini responses will fail.")

# -----------------------------
# SAFE FILE LOADING (IMPORTANT FIX)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "portfolio_data.txt")

try:
    with open(file_path, encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
except Exception as e:
    print("❌ portfolio_data.txt loading failed:", e)
    docs = ["Portfolio data not available."]

vectorizer = TfidfVectorizer().fit(docs)
doc_vectors = vectorizer.transform(docs)

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return "✅ Backend running. POST to /search"

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"answer": "Invalid request"}), 400

        question = data["question"]

        q_vec = vectorizer.transform([question])
        sims = cosine_similarity(q_vec, doc_vectors).flatten()
        top_idx = np.argsort(sims)[-4:][::-1]
        context = "\n".join([docs[i] for i in top_idx])

        prompt = f"Answer professionally using this context:\n{context}\n\nQ: {question}"

        if not api_key:
            return jsonify({"answer": "API key missing on server."})

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        return jsonify({"answer": response.text})

    except Exception as e:
        print("❌ ERROR in /search:", str(e))
        return jsonify({"answer": "Server error occurred."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

