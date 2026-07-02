import requests, json

BASE = "http://localhost:5008"

# Test summarize
payload = {
    "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving. This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.",
    "type": "paragraph",
    "length": "short"
}

print("Testing POST /api/summarize ...")
r = requests.post(f"{BASE}/api/summarize", json=payload, timeout=30)
print(f"Status: {r.status_code}")
data = r.json()
print(f"Engine: {data.get('engine', 'N/A')}")
print(f"Summary: {data.get('summary', data.get('error', 'NO RESPONSE'))[:300]}")
print("PASS" if r.status_code == 200 and data.get("summary") else "FAIL")
