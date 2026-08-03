"""
Thin wrapper around the Gemini API for generation.

Kept separate from the RAG logic in rag.py so the LLM provider is swappable —
if you later want to compare Gemini vs. an open-source model via Hugging Face,
only this file changes.
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY")
_MODEL_NAME = "gemini-3.5-flash"  # stable GA model as of Aug 2026 — gemini-1.5-flash was shut down

if _API_KEY:
    genai.configure(api_key=_API_KEY)


def generate(prompt: str) -> str:
    """Send a prompt to Gemini and return the generated text."""
    if not _API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file — "
            "get a free key at https://aistudio.google.com/app/apikey"
        )
    model = genai.GenerativeModel(_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text
