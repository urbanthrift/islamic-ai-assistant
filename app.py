
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI(title="Islamic AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are Islamic AI Assistant.
You support Malayalam, English and Arabic.

Answer questions about:
- Quran and its meanings
- Authentic Hadith
- Dua and Dhikr
- Islamic worship and daily life
- Traditional Islamic dream interpretation

Rules:
1. Respect Islamic beliefs.
2. Never claim knowledge of the unseen or predict the future.
3. Never invent Quran verses or Hadith references.
4. Clearly mention uncertainty and differences of scholarly opinion.
5. Refer complex religious rulings to qualified scholars.
6. Reply in the language used by the user.
7. Be polite, clear and helpful.
"""

class ChatRequest(BaseModel):
    message: str
    language: str = "Malayalam"

@app.get("/")
def home():
    return {"status": "Islamic AI Backend is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        return {"reply": "AI API Key is not configured yet."}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                SYSTEM_PROMPT
                + "\nPreferred language: " + request.language
                + "\nUser question: " + request.message
            )
        )
        return {"reply": response.text or "Please try again."}
    except Exception:
        return {"reply": "Sorry, AI is temporarily unavailable. Please try again."}
      
