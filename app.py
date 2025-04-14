import subprocess
import sys
import os
import requests
import json
from datetime import datetime

# === Auto-install missing dependencies ===
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from flask import Flask, render_template, request, jsonify
except ImportError:
    install("flask")
    from flask import Flask, render_template, request, jsonify

try:
    from llama_cpp import Llama
except ImportError:
    install("llama-cpp-python")
    from llama_cpp import Llama

# === Constants ===
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "chatlog.jsonl")

# === Auto-download model if missing ===
def download_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.isfile(MODEL_PATH):
        print("Model not found. Downloading from Hugging Face...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Model downloaded successfully.")

download_model()

# === Initialize Flask + Load LLM ===
app = Flask(__name__)
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)

# === Ensure log folder ===
os.makedirs(LOG_DIR, exist_ok=True)

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_prompt = request.json.get("prompt")
    system_prompt = request.json.get("system")

    full_prompt = f"{system_prompt}\nUser: {user_prompt}\nAI:"
    response = llm(full_prompt, max_tokens=300, stop=["User:", "AI:"], echo=False)
    answer = response["choices"][0]["text"].strip()

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": answer
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
