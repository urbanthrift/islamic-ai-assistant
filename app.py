
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

SYSTEM_PROMPT = """
You are Islamic AI Assistant.

Your supported languages are Malayalam, English and Arabic.

Your purpose:
- Answer questions about the Quran and authentic Hadith.
- Explain Islamic beliefs and worship.
- Help with Dua, Dhikr and daily Islamic practices.
- Explain traditional Islamic dream interpretation.

Important rules:
1. Never claim to know the unseen or predict the future.
2. Never invent Quran verses, Hadith or religious references.
3. Clearly distinguish authentic sources from traditional interpretations.
4. Mention differences of scholarly opinion when relevant.
5. Refer complex religious rulings to qualified Islamic scholars.
6. Be respectful, accurate and easy to understand.
7. Always reply in the language requested by the user.
8. If you are uncertain, say so honestly.
"""

class ChatRequest(BaseModel):
    message: str
    language: str = "Malayalam"


@app.get("/")
def home():
    return {
        "status": "Islamic AI Backend is running",
        "assistant": "Islamic AI Assistant"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY is missing")
        return {"reply": "Gemini API Key is not configured."}

    if not request.message.strip():
        return {"reply": "Please enter your question."}

    try:
        client = genai.Client(api_key=api_key)

        prompt = (
            SYSTEM_PROMPT
            + "\n\nPreferred language: "
            + request.language
            + "\n\nUser question: "
            + request.message
        )

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )

        answer = interaction.output_text

        if not answer:
            logger.error("Gemini returned an empty response")
            return {"reply": "No answer received. Please try again."}

        return {"reply": answer}

    except Exception:
        logger.exception("Gemini API request failed")
        return {
            "reply": "AI service error. Please check Render logs."
        }
