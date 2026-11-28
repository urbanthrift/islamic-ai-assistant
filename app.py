import os
import logging
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("islamic-ai")


def client_ip(request: Request) -> str:
    # Render sits behind a proxy, so the real client IP is in X-Forwarded-For
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)

app = FastAPI(
    title="Islamic AI Assistant",
    description="Multilingual Islamic AI Assistant",
    version="1.2.0"
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # The frontend reads `detail`, so keep that key
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a minute and try again."},
    )


# Only your site may call this API from a browser.
# Add more origins (e.g. a custom domain) via the ALLOWED_ORIGINS env var, comma-separated.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "https://urbanthrift.github.io").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MODEL_NAME = "gemini-3.5-flash-lite"
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """
You are DeenAI, a respectful Islamic AI assistant.

LANGUAGE:
- Support Malayalam, English, Hindi and Arabic.
- Always answer in the exact language selected by the user interface.
- If the preferred language is Malayalam, write clear natural Malayalam.
- If the preferred language is English, answer in clear English.
- If the preferred language is Hindi, answer in clear, natural Hindi (हिन्दी).
- If the preferred language is Arabic, answer in clear Arabic.
- Do not switch languages unless the user explicitly asks.
- Keep Arabic Quran verses and Hadith quotations in their original Arabic when relevant, and explain them in the selected response language.

ISLAMIC GUIDANCE:
- Base answers on the Quran and authentic Hadith.
- Mention Surah names and verse numbers when known.
- Mention Hadith collections and references when known.
- Never invent Quran verses, Hadith text or references.
- Distinguish established teachings from scholarly opinions.
- For complex fatwa questions, recommend consulting a qualified Islamic scholar.
- Respect differences between recognised Islamic schools.

DREAM INTERPRETATION:
- Explain traditional Islamic dream interpretation cautiously.
- Present historical interpretations as traditional views, not certain facts.
- Never claim to know the unseen or predict the future.
- Do not attribute unsupported interpretations to Prophet Muhammad (peace be upon him).

BEHAVIOUR:
- Be respectful, polite and helpful.
- Do not insult any religion or community.
- If you do not know an answer, say so honestly.
- Do not make up religious references.
- Keep answers clear and organised.
"""

MAX_HISTORY_MESSAGES = 10


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    language: str = Field(default="Malayalam", max_length=30)
    history: List[HistoryItem] = Field(default_factory=list, max_length=40)


def build_contents(history: List[HistoryItem], message: str, language: str):
    """Turn recent chat history + the new message into Gemini `contents`."""
    turns = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        role = "user" if item.role == "user" else "model"
        # merge consecutive same-role turns (e.g. after a failed reply)
        if turns and turns[-1]["role"] == role:
            turns[-1]["parts"][0]["text"] += "\n" + item.text
        else:
            turns.append({"role": role, "parts": [{"text": item.text}]})

    # Gemini conversations must start with a user turn
    while turns and turns[0]["role"] != "user":
        turns.pop(0)

    final = f"[Preferred response language: {language}]\n{message}"
    if turns and turns[-1]["role"] == "user":
        turns[-1]["parts"][0]["text"] += "\n" + final
    else:
        turns.append({"role": "user", "parts": [{"text": final}]})
    return turns

@app.get("/")
def home():
    return {"status": "online", "name": "Islamic AI Assistant", "version": "1.2.0", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "healthy", "api_key_configured": client is not None}

@app.post("/chat")
@limiter.limit("10/minute")
@limiter.limit("100/day")
def chat(request: Request, payload: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini API key is not configured.")

    allowed = {"Malayalam", "English", "Hindi", "Arabic"}
    language = payload.language if payload.language in allowed else "Malayalam"
    contents = build_contents(payload.history, payload.message, language)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        answer = response.text
        if not answer:
            raise HTTPException(status_code=502, detail="The AI returned an empty response.")
        return {"reply": answer, "language": language, "model": MODEL_NAME}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini API request failed")
        raise HTTPException(status_code=503, detail="AI service is temporarily unavailable. Please try again.")
