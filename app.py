# ==================================================
# SUMMARIZER SERVICE
# Port: 5008
# Route: POST /api/summarize
# ==================================================

import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Load .env from parent backend directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

app = Flask(__name__)
CORS(app)

SUMMARIZER_MODEL = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"

s_tokenizer = None
s_model = None

# --------------------------
# Lazy model loader
# --------------------------
def load_summarizer_model():
    global s_tokenizer, s_model
    if s_model is None:
        print(f"Loading summarizer model: {SUMMARIZER_MODEL}...")
        s_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
        s_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL).to(device)
        print("[SUCCESS] Summarizer model loaded.")

# --------------------------
# Route: POST /api/summarize
# --------------------------
@app.route('/api/summarize', methods=['POST'])
def handle_summarize():
    try:
        load_summarizer_model()
        data = request.json
        input_text = data.get('text', '')
        format_type = data.get('type', 'paragraph')
        length_preference = data.get('length', 'medium')

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        if len(input_text) > 50000:
            return jsonify({"error": "Text too long (max 50,000 characters)"}), 400

        if length_preference == "short":
            max_l, min_l = 50, 20
        elif length_preference == "long":
            max_l, min_l = 300, 100
        else:
            max_l, min_l = 150, 50

        inputs = s_tokenizer(
            input_text, max_length=1024,
            return_tensors="pt", truncation=True
        ).to(device)

        summary_ids = s_model.generate(
            inputs["input_ids"],
            max_length=max_l,
            min_length=min_l,
            do_sample=False,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary_text = s_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        if format_type == "bullets":
            sentences = summary_text.split(". ")
            summary_text = "\n".join([f"• {s.strip()}." for s in sentences if s.strip()])

        return jsonify({"summary": summary_text})

    except Exception as e:
        print(f"Summarizer Error: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------------
# Health check
# --------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "service": "summarizer",
        "model_loaded": s_model is not None,
        "device": device
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5008))
    app.run(host="0.0.0.0", port=port, debug=True)
