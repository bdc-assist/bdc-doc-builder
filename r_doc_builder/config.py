import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # real env vars win over .env, so shell overrides work as expected

CONFIG_DIR = os.getenv("CONFIG_DIR", "config")  # folder holding sources.yaml, prompts.yaml, build.yaml; examples/bdc for BDC
PROMPTS_FILE = os.path.join(CONFIG_DIR, "prompts.yaml")  # contextualizer / summarizer texts
SOURCES_FILE = os.path.join(CONFIG_DIR, "sources.yaml")  # root source list, the --yaml default
BUILD_FILE = os.path.join(CONFIG_DIR, "build.yaml")  # how the sources become chunks, below

# chunking, html reading and captions (sources.py) belong to the project, not the machine: build.yaml
with open(BUILD_FILE, encoding="utf-8") as f:
    _build = yaml.safe_load(f) or {}
CHUNK_SIZE = int(_build.get("chunk_size", 1500))  # characters per chunk
CHUNK_OVERLAP = int(_build.get("chunk_overlap", 200))  # overlap between neighbouring chunks
HTML_DROP_TAGS = list(_build.get("html_drop_tags", ["script", "style", "nav", "header", "footer", "noscript"]))
HTML_MAIN_TAGS = list(_build.get("html_main_tags", ["main", "article", "body"]))  # first one present holds the text
YOUTUBE_LANG = str(_build.get("youtube_lang", "en"))  # preferred caption language for youtube sources
for _k in ("CHUNK_SIZE", "CHUNK_OVERLAP", "HTML_DROP_TAGS", "HTML_MAIN_TAGS", "YOUTUBE_LANG"):
    if os.getenv(_k):  # these used to be env vars: say so rather than silently build something else
        print(f"warning: {_k} in the environment is ignored; set {_k.lower()} in {BUILD_FILE}", file=sys.stderr)

REPOS_DIR = os.getenv("REPOS_DIR", "./data/repos/")  # where git sources are cloned (shared: clones are named by repo)
# where pipeline writes the .pkl files (--data-dir default): data/<CONFIG_DIR folder name>/, so examples don't overwrite each other
PREPROC_DATA_DIR = os.getenv("PREPROC_DATA_DIR") or f"./data/{Path(CONFIG_DIR).resolve().name}/"
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0")  # sent with page, freshdesk and transcript downloads
COMPLETION_TEMPERATURE = float(os.getenv("COMPLETION_TEMPERATURE", "0"))  # every get_llm() provider
# unset for non-reasoning models (gpt-4o, gpt-4o-mini); none|low|medium|high for reasoning ones
# (gpt-6-luna). See _openai_kwargs. OpenAI-family providers only.
COMPLETION_REASONING_EFFORT = os.getenv("COMPLETION_REASONING_EFFORT") or None
RETRIES = int(os.getenv("RETRIES", "5"))  # attempts per LLM / embedding call before giving up
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))  # one source download (sources.py)
SUMMARY_MIN_CHARS = int(os.getenv("SUMMARY_MIN_CHARS", "300"))  # ingest --summarize: shorter texts are embedded as-is
# r-doc-mcp ingest API (ingest.py): where built content is pushed, and how
DOC_MCP_URL = os.getenv("DOC_MCP_URL", "http://127.0.0.1:8000").rstrip("/")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")  # must match INGEST_TOKEN on the r-doc-mcp side
PUSH_BATCH = int(os.getenv("PUSH_BATCH", "200"))  # chunks per upsert request
PUSH_TIMEOUT = int(os.getenv("PUSH_TIMEOUT", "300"))  # one ingest API request
EMBEDDING_BATCH_TOKENS = int(os.getenv("EMBEDDING_BATCH_TOKENS", "6000"))  # per-request token budget when embedding
# small-window models (8k) overflow on a whole doc as context.
# ponytail: hard truncation of the situating context — raise this (or point COMPLETION_MODEL
# at a long-context model) if chunks near the end of long docs get situated poorly.
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT", "16000"))


def _self_hosted_key():
    # self-hosted vLLM ignores the key, but the openai client refuses to start without one
    return os.getenv("OPENAI_API_KEY") or "EMPTY"


def _provider(kind: str) -> str:
    explicit = os.getenv(f"{kind}_MODEL_PROVIDER")
    if explicit:
        # split on '#': a trailing .env comment can survive into the value and is unreadable as an error
        return explicit.split("#")[0].strip().lower()
    url = os.getenv(f"{kind}_URL")
    if url:
        # ponytail: URL substring heuristic; set *_MODEL_PROVIDER explicitly to override
        # :11434 is Ollama's port — covers kubectl-tunneled remote Ollama at localhost
        if "ollama" in url or ":11434" in url:
            return "ollama"
        return "azure" if "azure.com" in url else "vllm"
    return "openai"


class FakeEmbeddings:
    """EMBEDDING_MODEL_PROVIDER=fake: a vector made from a hash of the text. Offline, free and
    deterministic, for tests and the fixture rehearsal (tests/test_fixture_cli.py). Search over it only
    finds exact repeats. r-doc-mcp has the same class, so query and document vectors agree."""
    model = "fake"  # part of every embed_hash: switching to or from a real model re-embeds

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [b / 255 for b in hashlib.sha256(text.encode()).digest()]


@lru_cache
def get_emb():
    provider = _provider("EMBEDDING")
    url = os.getenv("EMBEDDING_URL")
    model = os.getenv("EMBEDDING_MODEL")
    print(f"embeddings: provider={provider} model={model} url={url}", file=sys.stderr)
    if provider == "fake":
        return FakeEmbeddings()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        # check_embedding_ctx_length=False sends raw strings, not token arrays —
        # required by OpenAI-compatible gateways; chunks are small so no length risk
        return OpenAIEmbeddings(model=model or "text-embedding-3-small", check_embedding_ctx_length=False)
    if provider == "vllm":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(base_url=url, model=model, api_key=_self_hosted_key(),
                                check_embedding_ctx_length=False)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=url, model=model)
    raise ValueError(f"Unsupported EMBEDDING_MODEL_PROVIDER: {provider}")


def _openai_kwargs() -> dict:
    """Temperature and reasoning for the OpenAI-family clients, by COMPLETION_REASONING_EFFORT.
    Each rule is what the API returned for that model (tests/test_contextualize.py):
      unset   temperature only: gpt-4o-mini rejects any reasoning_effort, even "none"
      none    reasoning off: gpt-6-luna then takes COMPLETION_TEMPERATURE
      other   no temperature: a reasoning gpt-6-luna takes only its default. Contextualizing makes
              plain calls, which chat completions serves while reasoning (r-assist's tool calls need
              the Responses API; not here, so replies stay plain strings)."""
    if not COMPLETION_REASONING_EFFORT:
        return {"temperature": COMPLETION_TEMPERATURE}
    if COMPLETION_REASONING_EFFORT == "none":
        return {"temperature": COMPLETION_TEMPERATURE, "reasoning_effort": "none"}
    return {"reasoning_effort": COMPLETION_REASONING_EFFORT}


@lru_cache
def get_llm():
    provider = _provider("COMPLETION")
    url = os.getenv("COMPLETION_URL")
    model = os.getenv("COMPLETION_MODEL")
    print(f"llm: provider={provider} model={model} url={url} reasoning_effort={COMPLETION_REASONING_EFFORT}",
          file=sys.stderr)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o-mini", **_openai_kwargs())
    if provider == "azure" and url and url.rstrip("/").endswith("/openai/v1"):
        # Azure's OpenAI-compatible v1 gateway speaks plain OpenAI — the deployments
        # client would stack its own path on top and 404
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(base_url=url, model=model, **_openai_kwargs())
    if provider == "azure":
        # COMPLETION_MODEL is the *deployment* name here, and COMPLETION_URL the resource root
        # (no /openai/v1 suffix) — the client appends the deployment path itself
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=url, azure_deployment=model, **_openai_kwargs(),
            api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )
    if provider == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(base_url=url, model=model, **_openai_kwargs(), api_key=_self_hosted_key())
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(base_url=url, model=model, temperature=COMPLETION_TEMPERATURE)
    raise ValueError(f"Unsupported COMPLETION_MODEL_PROVIDER: {provider}")
