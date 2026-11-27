
import os
import logging
import time

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

Support Malayalam, English and Arabic.

Your purpose:
- Quran and authentic Hadith explanations
- Islamic beliefs and worship
- Dua and Dhikr
- Islamic daily guidance
- Traditional Islamic dream interpretation

Important rules:
1. Never claim knowledge of the unseen.
2. Never predict the future.
3. Never invent Quran verses or Hadith.
4. Distinguish authentic sources from traditional interpretations.
5. Mention scholarly differences when relevant.
6. Refer complex religious rulings to qualified scholars.
7. Reply in the user's requested language.
8. Be respectful and accurate.
9. If uncertain, say so honestly.
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

        last_error = None

        for attempt in range(3):
            try:
                result = client.interactions.create(
                    model="gemini-3.5-flash-lite",
                    input=prompt
                )

                answer = result.output_text

                if answer:
                    return {"reply": answer}

                return {"reply": "No answer received. Please try again."}

            except Exception as e:
                last_error = e
                logger.warning(
                    "Gemini attempt %s failed: %s",
                    attempt + 1,
                    str(e)
                )

                if attempt < 2:
                    time.sleep(2 ** attempt)

        logger.error("All Gemini attempts failed: %s", str(last_error))

        return {
            "reply": "AI is temporarily busy. Please try again shortly."
        }

    except Exception:
        logger.exception("Gemini setup failed")
        return {
            "reply": "AI service error. Please check server logs."
        }
        
