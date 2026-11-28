import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("islamic-ai")

app = FastAPI(
    title="Islamic AI Assistant",
    description="Multilingual Islamic AI Assistant",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    language: str = Field(default="Malayalam", max_length=30)

@app.get("/")
def home():
    return {"status": "online", "name": "Islamic AI Assistant", "version": "1.1.0", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "healthy", "api_key_configured": client is not None}

@app.post("/chat")
def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini API key is not configured.")

    allowed = {"Malayalam", "English", "Hindi", "Arabic"}
    language = request.language if request.language in allowed else "Malayalam"
    prompt = f"""{SYSTEM_PROMPT}

Preferred response language: {language}

User question:
{request.message}
"""
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        answer = response.text
        if not answer:
            raise HTTPException(status_code=502, detail="The AI returned an empty response.")
        return {"reply": answer, "language": language, "model": MODEL_NAME}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Gemini API request failed")
        raise HTTPException(status_code=503, detail="AI service is temporarily unavailable. Please try again.")
