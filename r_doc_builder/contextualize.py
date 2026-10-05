"""LLM chunk contextualizer (Anthropic-style contextual retrieval) and summarizer."""
import time

import yaml

from .config import CONTEXT_CHAR_LIMIT, PROMPTS_FILE, RETRIES, SUMMARY_MIN_CHARS, get_llm

with open(PROMPTS_FILE, encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f) or {}  # contextualize_chunk / summarize texts; placeholders filled below

for _k in ("contextualize_chunk", "summarize"):
    if _k not in PROMPTS:
        raise KeyError(f"{PROMPTS_FILE}: missing prompt {_k!r}")


def _invoke_llm(prompt, attempts=RETRIES):
    """Hosted gateways rate-limit sustained runs with transient errors the client does not
    retry. Back off and retry; give up rather than lose a whole run.
    ponytail: returns None on give-up so the caller falls back to raw text — a few
    uncontextualized chunks beats a dead 20-minute pipeline."""
    for attempt in range(attempts):
        try:
            return get_llm().invoke(prompt).content.strip()
        except Exception as e:
            if attempt == attempts - 1:
                print(f"  LLM call failed after {attempts} tries ({type(e).__name__}); using raw text")
                return None
            time.sleep(2 ** attempt)


def contextualize_chunk(chunk_content, whole_document):
    """Prepend an LLM-written situating sentence to the chunk (for embedding, not display)."""
    context_text = _invoke_llm(PROMPTS["contextualize_chunk"].format(
        context=str(whole_document)[:CONTEXT_CHAR_LIMIT], chunk_content=chunk_content[:CONTEXT_CHAR_LIMIT]))
    if context_text is None:
        return chunk_content
    return f"{context_text} {chunk_content}"


def get_summary(text, min_text=SUMMARY_MIN_CHARS):
    if len(text) < (min_text or 0):
        return text
    return _invoke_llm(PROMPTS["summarize"].format(text=text[:CONTEXT_CHAR_LIMIT])) or text
