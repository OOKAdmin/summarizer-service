# ==================================================
# SUMMARIZER SERVICE - Abstractive with Tone Variety
# ==================================================

import os
import random
import requests as http_requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

app = Flask(__name__)
CORS(app)

# -- Tier 1: Gemini --
gemini_available = False
gemini_client = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_available = True
        print("[OK] Gemini ready")
    except Exception as e:
        print(f"[FAIL] Gemini init: {e}")

# -- Tier 2: HuggingFace --
hf_available = False
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = "facebook/bart-large-cnn"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
if HF_API_TOKEN:
    hf_available = True
    print("[OK] HF ready")

# -- Tier 3: Local sumy --
sumy_available = False
try:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    sumy_available = True
    print("[OK] Local ready")
except Exception as e:
    print(f"[FAIL] Local init: {e}")

# --- Prompt builder (aggressively abstractive, natural explanation style) ---
def build_prompt(input_text, format_type, length_preference):
    return f"""
Your task is to explain the following text in your own words, like a human explaining a concept clearly.

CRITICAL RULES:
- NEVER copy full sentences from the source. Always rephrase and restructure.
- Make the explanation feel natural, clear, and easy to understand.
- Do NOT include any meta-announcements, introductions (e.g. "Sure, here is the summary:"), or roleplay phrases.
- The explanation must be shorter than the original text and capture the core ideas.

Length:
- short: 2-3 sentences. Keep it brief.
- medium: 4-5 sentences. Provide a balanced explanation.
- long: 6-8 sentences. Explain in detail.

Format:
- paragraph: write a cohesive paragraph.
- bullets: list the key concepts as bullet points (use '-' at the start of each line).

Text to explain:
{input_text}

Explanation:
"""

# --- Gemini summarizer ---
def summarize_gemini(input_text, format_type, length_preference):
    prompt = build_prompt(input_text, format_type, length_preference)
    # We only try 2 key models to keep it fast
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None
    
    for model_name in models_to_try:
        try:
            print(f"[->] Trying Gemini model: {model_name}")
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.8, "top_p": 0.9, "max_output_tokens": 1024}
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[FAIL] Gemini model {model_name}: {e}")
            last_err = e
            
    if last_err:
        raise last_err
    raise Exception("All Gemini models failed")

# --- HuggingFace ---
def summarize_huggingface(input_text, format_type, length_preference):
    if length_preference == "short":
        max_length, min_length = 80, 20
    elif length_preference == "long":
        max_length, min_length = 300, 100
    else:
        max_length, min_length = 150, 50
        
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": input_text[:4000],
        "parameters": {
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    # Short timeout (5s) for fast response and fallback
    response = http_requests.post(HF_API_URL, headers=headers, json=payload, timeout=5)
    response.raise_for_status()
    result = response.json()
    summary = result[0]["summary_text"] if isinstance(result, list) and result else result.get("summary_text", "")
    
    if not summary:
        return ""
        
    if format_type == "bullets":
        sentences = [s.strip() for s in summary.split('. ') if s.strip()]
        return "\n".join(f"- {s.rstrip('.')}." for s in sentences) if len(sentences) > 1 else f"- {summary}"
    else:
        return summary

# --- Local summarizer ---
def summarize_local(input_text, format_type, length_preference):
    word_count = len(input_text.split())
    if length_preference == "short":
        sentence_count = max(2, word_count // 100)
    elif length_preference == "long":
        sentence_count = max(5, word_count // 30)
    else:
        sentence_count = max(3, word_count // 50)
    sentence_count = min(sentence_count, 15)
    
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    
    # Extract more candidates than needed for non-deterministic selection
    candidate_count = min(sentence_count + 4, 25)
    candidates = summarizer(parser.document, candidate_count)
    
    if len(candidates) > sentence_count:
        all_sentences = list(parser.document.sentences)
        candidate_with_indices = []
        for c in candidates:
            try:
                idx = all_sentences.index(c)
                candidate_with_indices.append((idx, c))
            except ValueError:
                pass
        
        # Randomly choose sentence_count items
        sampled = random.sample(candidate_with_indices, min(sentence_count, len(candidate_with_indices)))
        # Sort by original sentence index to preserve flow
        sampled.sort(key=lambda x: x[0])
        summary_sentences = [x[1] for x in sampled]
    else:
        summary_sentences = candidates
        
    if format_type == "bullets":
        return "\n".join(f"- {str(s)}" for s in summary_sentences)
    else:
        return " ".join(str(s) for s in summary_sentences)

# --- Route ---
@app.route('/api/summarize', methods=['POST'])
def handle_summarize():
    try:
        data = request.json
        input_text = data.get('text', '')
        format_type = data.get('type', 'paragraph')
        length_preference = data.get('length', 'medium')
        if not input_text:
            return jsonify({"error": "No text"}), 400
        if len(input_text) > 50000:
            return jsonify({"error": "Too long"}), 400

        # Gemini first
        if gemini_available:
            try:
                print("[->] Running Gemini summarization")
                summary = summarize_gemini(input_text, format_type, length_preference)
                return jsonify({"summary": summary, "engine": "gemini"})
            except Exception as e:
                print(f"[FAIL] Gemini overall pipeline failed: {e}")

        # HF
        if hf_available:
            try:
                print("[->] Falling back to HuggingFace")
                summary = summarize_huggingface(input_text, format_type, length_preference)
                return jsonify({"summary": summary, "engine": "huggingface"})
            except Exception as e:
                print(f"[FAIL] HF: {e}")

        # Local
        if sumy_available:
            try:
                print("[->] Falling back to Local Sumy")
                summary = summarize_local(input_text, format_type, length_preference)
                return jsonify({"summary": summary, "engine": "local"})
            except Exception as e:
                print(f"[FAIL] Local: {e}")

        return jsonify({"error": "All engines failed"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "engines": {
            "gemini": gemini_available,
            "huggingface": hf_available,
            "local": sumy_available
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5008))
    app.run(host="0.0.0.0", port=port, debug=True)