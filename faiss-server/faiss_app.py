from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------
# ENV SETUP
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("⚠️ GEMINI_API_KEY not found!")

# Initialize Gemini client (NEW SDK)
client = genai.Client(api_key=api_key)

# -----------------------------
# LOAD DATA SAFELY
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
    return "✅ Backend running (Gemini 2.5 Flash Lite). POST to /search"


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"answer": "Invalid request"}), 400

        question = data["question"]

        # TF-IDF similarity
        q_vec = vectorizer.transform([question])
        sims = cosine_similarity(q_vec, doc_vectors).flatten()
        top_idx = np.argsort(sims)[-4:][::-1]
        context = "\n".join([docs[i] for i in top_idx])

        prompt = f"""
Answer professionally using the following context.

Context:
{context}

User Question:
{question}
"""

        # NEW SDK CALL
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )

        answer_text = response.text if response.text else "No response generated."

        return jsonify({"answer": answer_text})

    except Exception as e:
        print("❌ ERROR in /search:", str(e))
        return jsonify({"answer": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
