import os
import re
import time

import requests

from ..config import get_llm

# gemma-3-12b-it serves an 8192-token window; a whole gitbook doc as context overflows it.
# ponytail: hard truncation of the situating context — raise this (or point COMPLETION_MODEL
# at a long-context model) if chunks near the end of long docs get situated poorly.
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "16000"))

# prompts verbatim from BDC_Chatbot utils/rag/chain.py:create_chunk_contextualizer_chain
_DOC_PROMPT = """<document>
{context}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""

_METADATA_PROMPT = """<metadata_context>
{context}
</metadata_context>
Here is the chunk we want to situate with the metadata context
<chunk>
{chunk_content}
</chunk>

Please give a short succinct natural language context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""

_SUMMARY_PROMPT = """<document_summary>
{context}
</document_summary>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct natural language context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""


def _invoke_llm(prompt, attempts=5):
    """The vLLM ingress rate-limits sustained runs with a transient 403, which the openai
    client does not retry. Back off and retry; give up rather than lose a whole run.
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


def event_datetime(date, time_str):
    """'2021-08-11' + '1:00 - 2:00 pm EDT' -> '2021-08-11T13:00' (sortable, local ET).
    Falls back to the date alone when the time is missing or unparseable."""
    if not date:
        return None
    m = re.search(r"(\d{1,2}):(\d{2})", time_str or "")
    if not m:
        return str(date)
    hour, minute = int(m.group(1)), int(m.group(2))
    # the meridiem may only appear after the end time ('1:00 - 2:00 pm EDT'),
    # in which case it applies to the start time too — first one after the start wins
    mer = re.search(r"([ap])\.?\s?m\b", time_str[m.end():], re.I)
    if mer:
        pm = mer.group(1).lower() == "p"
        if pm and hour != 12:
            hour += 12
        elif not pm and hour == 12:
            hour = 0
    return f"{date}T{hour:02d}:{minute:02d}"


def youtube_upload_date(video_url):
    """Upload date ('YYYY-MM-DD') scraped from the watch page JSON — no API key needed.
    Returns None for non-YouTube URLs or on any fetch/parse failure."""
    if "youtu" not in (video_url or ""):
        return None
    try:
        resp = requests.get(video_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        m = re.search(r'"uploadDate":"([^"]+)"', resp.text)
        return m.group(1)[:10] if m else None
    except Exception as e:
        print(f"  upload date fetch failed for {video_url}: {type(e).__name__}: {e}")
        return None


def concat_paths(*paths):
    result = paths[0].rstrip("/") if paths else ""
    for path in paths[1:]:
        result = f"{result}/{path.lstrip('/')}"
    return result


def paths_to_urls(base_url, file_paths):
    urls = []
    for path in file_paths:
        clean = path.replace("\\", "/").replace(".mdx", "")
        clean = re.sub(r"/index$", "", clean)
        urls.append(f"{base_url.rstrip('/')}/{clean}")
    return urls


def split_by_sections(text, return_dict=False):
    """Split markdown text on '##' headers."""
    sections = []
    current_section = []
    current_header = None

    for line in text.split("\n"):
        if line.startswith("##"):
            if current_section:
                sections.append({"header": current_header, "content": "\n".join(current_section).strip()})
            current_header = line.strip()
            current_section = []
        else:
            current_section.append(line)

    if current_section:
        sections.append({"header": current_header, "content": "\n".join(current_section).strip()})

    if return_dict:
        return sections
    return [f"{s['header'] or ''}\n{s['content'] or ''}" for s in sections]


def contextualize_chunk(chunk_content, whole_document=None, metadata_context=None,
                        is_markdown=False, return_context_only=False, is_doc_summary=False):
    """Anthropic-style contextual retrieval: prepend an LLM-written situating sentence
    to the chunk before embedding."""
    if metadata_context is not None:
        prompt, context = _METADATA_PROMPT, metadata_context
        if is_doc_summary:
            prompt = _SUMMARY_PROMPT
    elif is_doc_summary:
        prompt, context = _SUMMARY_PROMPT, whole_document
    else:
        prompt, context = _DOC_PROMPT, whole_document

    context_text = _invoke_llm(
        prompt.format(context=str(context)[:CONTEXT_CHAR_LIMIT], chunk_content=chunk_content[:CONTEXT_CHAR_LIMIT])
    )
    if context_text is None:
        return "" if return_context_only else chunk_content

    if return_context_only:
        return context_text
    return f"{context_text}\n\n{chunk_content}" if is_markdown else f"{context_text} {chunk_content}"


def get_summary(text, min_text=300):
    """Ported from BDC_Chatbot utils/rag/chain.py:get_summary."""
    if len(text) < (min_text or 0):
        return text
    return _invoke_llm(
        "Write a concise summary of the following text in 1-3 sentences, return the summary ONLY, "
        f"This is NOT a conversation. \n\n{text[:CONTEXT_CHAR_LIMIT]}"
    ) or text
