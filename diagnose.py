"""Diagnose which summarization tiers work."""
import os, sys, traceback
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

TEST_TEXT = "Artificial intelligence is intelligence demonstrated by machines. AI research studies intelligent agents that perceive their environment and take actions to achieve goals. Modern AI techniques have become pervasive and are used to solve many problems in computer science."

# --- Tier 1: Gemini ---
print("=" * 50)
print("TIER 1: Gemini API")
print("=" * 50)
try:
    from google import genai
    key = os.environ.get("GEMINI_API_KEY", "")
    print(f"  API Key: {key[:10]}...{key[-4:]}" if len(key) > 14 else f"  API Key: {key}")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model="gemini-2.0-flash", contents="Summarize briefly: " + TEST_TEXT)
    print(f"  Result: {resp.text[:200]}")
    print("  STATUS: OK")
except Exception:
    traceback.print_exc()
    print("  STATUS: FAILED")

# --- Tier 2: HuggingFace ---
print("\n" + "=" * 50)
print("TIER 2: HuggingFace Inference API")
print("=" * 50)
try:
    import requests
    token = os.environ.get("HF_API_TOKEN", "")
    print(f"  Token: {token[:10]}...{token[-4:]}" if len(token) > 14 else f"  Token: {token}")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": TEST_TEXT, "parameters": {"max_length": 80, "min_length": 20}}
    r = requests.post("https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
                       headers=headers, json=payload, timeout=30)
    print(f"  HTTP Status: {r.status_code}")
    print(f"  Response: {r.text[:300]}")
    print("  STATUS: OK" if r.status_code == 200 else "  STATUS: FAILED")
except Exception:
    traceback.print_exc()
    print("  STATUS: FAILED")

# --- Tier 3: Local sumy ---
print("\n" + "=" * 50)
print("TIER 3: Local sumy/LSA")
print("=" * 50)
try:
    import nltk
    nltk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_dir)
    nltk.download('punkt_tab', download_dir=nltk_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_dir, quiet=True)
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    parser = PlaintextParser.from_string(TEST_TEXT, Tokenizer("english"))
    summarizer = LsaSummarizer(Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")
    result = " ".join(str(s) for s in summarizer(parser.document, 2))
    print(f"  Result: {result[:200]}")
    print("  STATUS: OK")
except Exception:
    traceback.print_exc()
    print("  STATUS: FAILED")
