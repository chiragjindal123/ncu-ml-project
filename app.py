from flask import Flask, render_template, request, jsonify
import requests
import subprocess
import psycopg2
import json
from rag_utils import get_context

app = Flask(__name__)

# --- CONFIG ---
GEMINI_API_KEY = "AIzaSyCb7gAoVeXmrascLigIfgXiXf2dCbhsYlc"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    model = data.get("model")  # "gemini" or "ollama"

    # Add RAG context
    rag_context = get_context(user_input)
    
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

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
