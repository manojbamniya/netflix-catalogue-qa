"""Answer a question from retrieved titles with Gemini, and keep only real sources."""

import sqlite3
from typing import Any, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel

from api import queries
from rag.retriever import Retriever

NO_MATCH_ANSWER = "I couldn't find a title in the catalogue that answers this."

SYSTEM_PROMPT = """You answer questions about a Netflix catalogue.
Use ONLY the titles listed in the user's message. Do not use outside knowledge
about films, people or plots, and never mention a title that isn't listed.
If none of the listed titles fit the question, say so and return no show_ids.
Return JSON with:
  "answer": a short answer that names the titles you recommend and why,
  "show_ids": the show_id of every title your answer is based on."""


class LLMAnswer(BaseModel):
    """What the LLM must return (enforced with Gemini's JSON schema output)."""

    answer: str
    show_ids: list[str]


class LLM(Protocol):
    """Anything that can answer a question from a block of context text."""

    def generate(self, question: str, context: str) -> LLMAnswer: ...


class GeminiLLM:
    """Calls Gemini and asks for JSON matching LLMAnswer."""

    TEMPERATURE = 0.2  # low: stick to the given titles, little creativity

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, question: str, context: str) -> LLMAnswer:
        """Send the titles and question; parse the JSON reply."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"Titles:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.TEMPERATURE,
                response_mime_type="application/json",
                response_schema=LLMAnswer,
            ),
        )
        return LLMAnswer.model_validate_json(response.text)


def format_context(titles: list[dict[str, Any]]) -> str:
    """One block of text per retrieved title, labelled with its show_id."""
    blocks = [
        f"[show_id: {t['show_id']}] {t['title']} ({t['type']}, {t['year']}, "
        f"country: {t['country'] or 'Unknown'}, rating: {t['rating'] or 'Unrated'})\n"
        f"{t['movie_description']}"
        for t in titles
    ]
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    conn: sqlite3.Connection,
    retriever: Retriever,
    llm: LLM,
    top_k: int,
) -> dict[str, Any]:
    """Retrieve titles, ask the LLM, and return the answer with its sources.

    Any show_id the LLM returns that wasn't retrieved is dropped, so every
    source really is a title the answer was given. If no valid source is left,
    the answer is replaced with NO_MATCH_ANSWER.
    """
    retrieved_ids = retriever.search(question, k=top_k)
    titles = [t for i in retrieved_ids if (t := queries.get_title(conn, i))]
    reply = llm.generate(question, format_context(titles))

    by_id = {t["show_id"]: t for t in titles}
    sources = [by_id[i] for i in dict.fromkeys(reply.show_ids) if i in by_id]
    if not sources:
        return {"answer": NO_MATCH_ANSWER, "show_ids": [], "sources": []}
    return {
        "answer": reply.answer,
        "show_ids": [t["show_id"] for t in sources],
        "sources": [{"show_id": t["show_id"], "title": t["title"]} for t in sources],
    }
