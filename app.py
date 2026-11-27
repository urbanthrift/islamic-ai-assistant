
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

# -----------------------------
# Application setup
# -----------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("islamic-ai")

app = FastAPI(
    title="Islamic AI Assistant",
    description="Multilingual Islamic AI Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Gemini configuration
# -----------------------------

MODEL_NAME = "gemini-3.5-flash-lite"

API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None

# -----------------------------
# Islamic AI instructions
# -----------------------------

SYSTEM_PROMPT = """
You are Islamic AI Assistant.

Your purpose is to provide helpful Islamic information
in Malayalam, English and Arabic.

LANGUAGE:
- Reply in the language requested by the user.
- Support Malayalam, English and Arabic.
- Use simple and understandable language.

ISLAMIC GUIDANCE:
- Give answers based on the Quran and authentic Hadith.
- Mention Surah names and verse numbers when known.
- Mention Hadith collections and references when known.
- Never invent Quran verses or Hadith references.
- Clearly distinguish established Islamic teachings
  from scholarly opinions.
- For complex Fatwa questions, recommend consulting
  a qualified Islamic scholar.
- Respect differences between recognised Islamic schools.

DREAM INTERPRETATION:
- Explain traditional Islamic dream interpretation
  cautiously.
- Historical interpretations may be mentioned as
  traditional views, not certain facts.
- Never claim to know the unseen or predict the future.
- Explain that dreams can have different meanings
  and should not be used as the basis for major decisions.
- Do not attribute unsupported interpretations
  to Prophet Muhammad (peace be upon him).

BEHAVIOUR:
- Be respectful, polite and helpful.
- Do not insult any religion or community.
- If you do not know an answer, say so honestly.
- Do not make up religious references.
- Keep answers clear and organised.
"""

# -----------------------------
# Request model
# -----------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    language: str = Field(default="Malayalam", max_length=30)


# -----------------------------
# Home endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "status": "online",
        "name": "Islamic AI Assistant",
        "version": "1.0.0",
        "model": MODEL_NAME
    }


# -----------------------------
# Health endpoint
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api_key_configured": client is not None
    }


# -----------------------------
# Chat endpoint
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured."
        )

    prompt = f"""
{SYSTEM_PROMPT}

Preferred response language: {request.language}

User question:
{request.message}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        answer = response.text

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response."
            )

        return {
            "reply": answer,
            "language": request.language,
            "model": MODEL_NAME
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Gemini API request failed")

        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable. Please try again."
        )
        
