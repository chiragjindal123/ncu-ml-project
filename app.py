from flask import Flask, render_template, request, jsonify
import requests
import subprocess
import psycopg2
import json
from rag_utils import get_context, get_embedding, save_message, get_connection, chunk_text
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# --- CONFIG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = os.getenv("GEMINI_URL")

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    model = data.get("model")  # "gemini" or "ollama"
    use_rag = data.get("use_rag")
    
    save_message("user", user_input)

    prompt_lower = user_input.lower()
    if any(word in prompt_lower for word in ["quiz", "question", "test", "mcq"]):
        task = "quiz"
        # use_rag = True
    elif any(word in prompt_lower for word in ["implement", "practice", "code", "program", "exercise"]):
        task = "practice"
        # use_rag = True
    elif any(word in prompt_lower for word in ["review", "explain", "summarize", "summary"]):
        task = "review"
        # use_rag = True
    else:
        task = "general"
        # use_rag = False

    rag_context = get_context(user_input) if use_rag else ""

    # --- Prompt Engineering ---
    if task == "review":
        task_prompt = f"Review the following material and explain it simply:\n{user_input}"
    elif task == "quiz":
        task_prompt = (
            "Generate 5 quiz questions (with answers) in a numbered list:\n"
            f"{user_input}"
        )
    elif task == "practice":
        task_prompt = (
            "Give me a practical implementation exercise (with a brief solution):\n"
            f"{user_input}"
        )
    else:
        task_prompt = user_input

    final_prompt = f"Context:\n{rag_context}\n\nTask:\n{task_prompt}"


    if model == "gemini":
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [{"parts": [{"text": final_prompt}]}]
        }
        response = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload))
        try:
            data = response.json()
            if "candidates" in data and data["candidates"]:
                reply = data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                reply = f"Gemini API error: {data['error'].get('message', 'Unknown error')}"
            else:
                reply = "Gemini API returned an unexpected response."
        except Exception as e:
            reply = f"Failed to parse Gemini API response: {str(e)}"

    elif model == "ollama":
        result = subprocess.run(
            ["ollama", "run", "llama3:8b"],
            input=final_prompt,
            text=True,
            encoding="utf-8",
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

    chunks = chunk_text(content, chunk_size=1000, overlap=200)
    conn = get_connection()
    cur = conn.cursor()
    for chunk in chunks:
        embedding = get_embedding(chunk)
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)",
            (chunk, f"[{','.join(str(x) for x in embedding)}]")
        )
    conn.commit()
    conn.close()
    return jsonify({"message": f"File uploaded and embedded in {len(chunks)} chunks."})



if __name__ == "__main__":
    app.run(debug=True)
