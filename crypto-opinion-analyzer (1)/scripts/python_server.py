"""
Python API for Crypto Opinion Analysis

Endpoints:
- POST /classify  -> returns { level1, level2, level3, path }

This tries your trained Keras models (via predict_all_levels_path) and
gracefully falls back to a crypto-aware heuristic when models/tokenizer
aren't found yet.

How to run:
1) cd scripts
2) python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
3) pip install -r requirements.txt
4) uvicorn python_server:app --host 0.0.0.0 --port 8000

Frontend setup:
- In the Next.js app root, create .env.local with:
  CLASSIFIER_API_URL=http://localhost:8000/classify
- Restart `npm run dev` after setting env vars.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import os

# Ensure model files are discoverable no matter where server starts from
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.dirname(__file__))

# Import your existing predictor
try:
    # Note: this file is imported into scripts/ as transformer_wly89.py
    from transformer_wly89 import predict_all_levels_path  # type: ignore
except Exception as e:
    predict_all_levels_path = None  # will fallback
    print("[python_server] Could not import model module:", e)

app = FastAPI(title="Crypto Opinion Classifier", version="1.0.0")

# CORS for local Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyIn(BaseModel):
    text: str


def heuristic_predict(text: str) -> str:
    """
    Simple crypto-aware heuristic that mirrors the hierarchy:
      Level 1: noise | objective | subjective
      Level 2 (if subjective): neutral | negative | positive
      Level 3 (if neutral): neutral_sentiments | questions | advertisements | misc
    """
    t = (text or "").strip().lower()
    if len(t) < 3:
        return "noise"

    # Objective-like phrases (market data, price commentary)
    objective_kw = [
        "price is", "trading at", "market cap", "24h volume", "chart",
        "support at", "resistance", "ath", "atl", "up %", "down %", "$"
    ]
    if any(k in t for k in objective_kw):
        return "objective"

    # Advertisements/promotions often found in crypto
    ads_kw = ["giveaway", "airdrop", "promo", "promotion", "discount", "referral", "invite code", "bonus"]
    if any(k in t for k in ads_kw):
        return "subjective->neutral->advertisements"

    # Questions
    if "?" in t or any(k in t for k in ["any thoughts", "what do you think", "should i", "is it worth", "will it", "can it"]):
        return "subjective->neutral->questions"

    # Negative sentiment
    negative_kw = ["scam", "rugpull", "dump", "fear", "crash", "bearish", "fake", "bad", "hate", "angry", "down only", "avoid"]
    if any(k in t for k in negative_kw):
        return "subjective->negative"

    # Positive sentiment
    positive_kw = ["bullish", "buy", "moon", "pump", "great", "love", "good", "amazing", "boom", "surge", "green"]
    if any(k in t for k in positive_kw):
        return "subjective->positive"

    # Neutral subjective subtypes
    neutral_sent_kw = ["i think", "imo", "in my opinion", "feels like", "seems like", "i feel"]
    if any(k in t for k in neutral_sent_kw):
        return "subjective->neutral->neutral_sentiments"

    return "subjective->neutral->misc"


def path_to_levels(path: str) -> Dict[str, Optional[str]]:
    # Normalize to consistent lower-case labels
    parts = [p.strip().lower() for p in (path or "").split("->") if p.strip()]
    level1 = parts[0] if len(parts) >= 1 else None
    level2 = None
    level3 = None

    if level1 in ("noise", "objective"):
        # Terminal at level 1
        return {"level1": level1, "level2": None, "level3": None, "path": level1}

    # subjective flow
    if len(parts) >= 2:
        level2 = parts[1]
    if level2 == "neutral" and len(parts) >= 3:
        level3 = parts[2]

    # In case someone returns "subjective->positive/negative" without neutral
    return {
        "level1": level1 or "subjective",
        "level2": level2,
        "level3": level3,
        "path": "->".join(parts) if parts else None,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/classify")
def classify(inp: ClassifyIn):
    text = inp.text or ""
    # Try model first (if available and models exist), else heuristic
    if predict_all_levels_path is not None:
        try:
            # Ensure the working directory is where the model files live
            prev = os.getcwd()
            os.chdir(MODELS_DIR)
            try:
                path = predict_all_levels_path(text)
            finally:
                os.chdir(prev)
            return path_to_levels(path)
        except FileNotFoundError:
            # Models or tokenizer not present yet — fallback
            pass
        except Exception as e:
            # Any other runtime error — fallback but include info
            print("[python_server] model error:", e)

    # Fallback heuristic
    return path_to_levels(heuristic_predict(text))


if __name__ == "__main__":
    import uvicorn
    # Run with: python python_server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
