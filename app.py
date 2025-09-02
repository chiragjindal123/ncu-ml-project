from flask import Flask, render_template, request, jsonify
import requests
import subprocess
import psycopg2
import json
from rag_utils import get_context, get_embedding
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx


app = Flask(__name__)

# --- CONFIG ---
GEMINI_API_KEY = "AIzaSyCb7gAoVeXmrascLigIfgXiXf2dCbhsYlc"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

def save_message(role, content):
    conn = psycopg2.connect(
        dbname="aidb",
        user="aiuser",
        password="aipassword",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (role, content) VALUES (%s, %s)",
        (role, content)
    )
    conn.commit()
    conn.close()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    model = data.get("model")  # "gemini" or "ollama"
    use_rag = data.get("use_rag")
    
    save_message("user", user_input)

    # Add RAG context
    rag_context = get_context(user_input) if use_rag else ""
    
    final_prompt = f"Context:\n{rag_context}\n\nQuestion:\n{user_input}"

    if model == "gemini":
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}]
        }
        response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
        reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    elif model == "ollama":
        result = subprocess.run(
            ["ollama", "run", "llama3:8b"],
            input=final_prompt,
            text=True,
            capture_output=True
        )
        reply = result.stdout.strip()

    else:
        reply = "Invalid model selection."

    save_message("ai", reply)
    return jsonify({"reply": reply})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    # Only allow .txt, .pdf, .docx
    if ext not in [".txt", ".pdf", ".docx"]:
        return jsonify({"message": "Only .txt, .pdf, and .docx files are supported."}), 400

    try:
        if ext == ".txt":
            content = file.read().decode("utf-8")
        elif ext == ".pdf":
            reader = PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
        elif ext == ".docx":
            doc = docx.Document(file)
            content = "\n".join([para.text for para in doc.paragraphs])
        else:
            return jsonify({"message": "Unsupported file type."}), 400
    except Exception as e:
        return jsonify({"message": f"Failed to process file: {str(e)}"}), 400

    if not content.strip():
        return jsonify({"message": "No extractable text found in the file."}), 400

    embedding = get_embedding(content)
    conn = psycopg2.connect(
        dbname="aidb",
        user="aiuser",
        password="aipassword",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)",
        (content, f"[{','.join(str(x) for x in embedding)}]")
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "File uploaded and embedded successfully."})


if __name__ == "__main__":
    app.run(debug=True)
