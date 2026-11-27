
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("islamic-ai")

app = FastAPI(title="Islamic AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str = "Malayalam"

SYSTEM_PROMPT = """
You are Islamic AI Assistant.
Support Malayalam, English and Arabic.

Answer using reliable Islamic sources:
- Quran
- Authentic Hadith
- Islamic worship
- Dua and Dhikr
- Islamic dream interpretation traditions

Rules:
1. Never claim knowledge of the unseen.
2. Never invent Quran verses or Hadith.
3. Mention differences of scholarly opinion.
4. Refer complex religious rulings to qualified scholars.
5. Reply in the requested language.
6. Be respectful and accurate.
"""

@app.get("/")
def home():
    return {"status": "Islamic AI Backend is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY is missing")
        return {"reply": "API Key is not configured."}

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                SYSTEM_PROMPT
                + "\nLanguage: " + request.language
                + "\nQuestion: " + request.message
            )
        )

        if not response.text:
            logger.error("Gemini returned an empty response")
            return {"reply": "AI returned an empty response."}

        return {"reply": response.text}

    except Exception as e:
        logger.exception("Gemini API request failed")
        return {"reply": "AI service error. Please check server logs."}
        
