from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("❌ GEMINI_API_KEY is missing in .env")

genai.configure(api_key=api_key)

app = Flask(__name__)
CORS(app)

# Load once
with open("portfolio_data.txt") as f:
    data = f.read()

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(data)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(chunks, embedding=embeddings)

@app.route("/")
def home():
    return "✅ FAISS + Gemini server running. POST to /search"

@app.route("/search", methods=["POST"])
def search():
    q = request.json.get("question")
    results = db.similarity_search(q, k=4)
    context = "\n".join([r.page_content for r in results])
    
    prompt = f"Answer the question from this context:\n{context}\n\nQ: {q}"
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    answer = model.generate_content(prompt)
    
    return jsonify({"answer": answer.text})

if __name__ == "__main__":
    app.run(port=8000)
