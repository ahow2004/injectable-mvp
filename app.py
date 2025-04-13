import subprocess
import sys

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

import json
import os
from datetime import datetime
from llama_cpp import Llama
import json
import os
from datetime import datetime

app = Flask(__name__)

# === Load your model (.gguf) ===
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # <-- replace with your model filename
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)

# === Logs ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chatlog.jsonl")

# === Flask routes ===
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
