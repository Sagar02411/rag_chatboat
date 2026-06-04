"""
download_model.py
-----------------
One-time script to download the sentence-transformers embedding model
using requests (SSL verification disabled) for corporate proxy environments.

Run once:
    .\\venv\\Scripts\\python.exe download_model.py
"""

import os
import ssl

# Disable SSL globally BEFORE any import that triggers network calls
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL  = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
LOCAL_DIR = "./models/all-MiniLM-L6-v2"
POOLING_DIR = f"{LOCAL_DIR}/1_Pooling"

FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "sentence_bert_config.json",
    "modules.json",
    "model.safetensors",          # ~22 MB — the actual weights
    "1_Pooling/config.json",
]

def download(session: requests.Session, filename: str) -> None:
    local_path = os.path.join(LOCAL_DIR, filename.replace("/", os.sep))
    os.makedirs(os.path.dirname(local_path) or LOCAL_DIR, exist_ok=True)

    if os.path.exists(local_path):
        print(f"  [OK] Already exists: {filename}")
        return

    url = f"{BASE_URL}/{filename}"
    print(f"  >> Downloading {filename} ...", end="", flush=True)
    resp = session.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    size_kb = os.path.getsize(local_path) // 1024
    print(f" done ({size_kb} KB)")


if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(POOLING_DIR, exist_ok=True)

    session = requests.Session()
    session.verify = False          # Bypass SSL — required for corporate proxy

    print(f"\nDownloading all-MiniLM-L6-v2 to '{LOCAL_DIR}' ...\n")
    for f in FILES:
        download(session, f)

    # Quick sanity check — load the model
    print("\nVerifying model loads correctly ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_DIR)
    vec = model.encode("hello world")
    print(f"[OK] Model ready! Embedding dimension = {len(vec)}")
    print("\nAll done! The app will now use the local model.\n")
