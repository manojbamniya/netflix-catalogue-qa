"""Ask Gemini to answer from the retrieved titles, then keep only real sources."""

import os
import sqlite3
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from api import queries

load_dotenv()  # reads GEMINI_API_KEY from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TIMEOUT_MS = 30_000  # per Gemini call
RETRY_ATTEMPTS = 3  # when Gemini's servers are busy (5xx)

NO_MATCH_ANSWER = "I couldn't find a title in the catalogue that answers this."

PROMPT = """You answer questions about a Netflix catalogue.
Use ONLY the titles listed below. Never mention a title that isn't listed.
Recommend the titles that fit best. If a title only partly fits, you may still
suggest it and say what you're not sure about. Return no show_ids only if none
of the titles are related to the question.
Return JSON: "answer" (short, names the titles and why) and "show_ids"
(the show_id of every title your answer uses)."""


class LLMAnswer(BaseModel):
    """The JSON shape Gemini must reply with."""

    answer: str
    show_ids: list[str]


# Any function that takes (question, titles text) and returns an LLMAnswer.
# The real one is ask_gemini; tests pass a fake one.
AskLLM = Callable[[str, str], LLMAnswer]


def ask_gemini(question: str, titles_text: str) -> LLMAnswer:
    """Send the titles and question to Gemini and read its JSON reply."""
    # Give up on a call after 30 s. Try up to 3 times when Gemini's servers are
    # busy (5xx errors, e.g. "503 high demand"); not on 429 "quota used up",
    # because retrying can't help with that.
    retry = types.HttpRetryOptions(
        attempts=RETRY_ATTEMPTS, http_status_codes=[500, 502, 503, 504]
    )
    options = types.HttpOptions(timeout=TIMEOUT_MS, retry_options=retry)
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=options)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Titles:\n{titles_text}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=PROMPT,
            temperature=0.2,  # low = stick to the given titles
            response_mime_type="application/json",
            response_schema=LLMAnswer,
        ),
    )
    return LLMAnswer.model_validate_json(response.text)


def titles_text(titles: list[dict[str, Any]]) -> str:
    """One block per title, labelled with its show_id so Gemini can cite it."""
    return "\n\n".join(
        f"[show_id: {t['show_id']}] {t['title']} ({t['type']}, {t['year']}, "
        f"country: {t['country'] or 'Unknown'})\n{t['movie_description']}"
        for t in titles
    )


def answer_question(
    question: str, retrieved_ids: list[str], conn: sqlite3.Connection, ask: AskLLM
) -> dict[str, Any]:
    """Answer from the retrieved titles; drop any show_id the LLM made up."""
    titles = [queries.get_title(conn, show_id) for show_id in retrieved_ids]
    titles = [t for t in titles if t]
    reply = ask(question, titles_text(titles))

    by_id = {t["show_id"]: t for t in titles}
    used_ids = [i for i in dict.fromkeys(reply.show_ids) if i in by_id]
    if not used_ids:
        return {"answer": NO_MATCH_ANSWER, "show_ids": [], "sources": []}
    return {
        "answer": reply.answer,
        "show_ids": used_ids,
        "sources": [{"show_id": i, "title": by_id[i]["title"]} for i in used_ids],
    }
