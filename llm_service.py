"""
llm_service.py
==============
Search-agent LLM service for the multilingual voice assistant.

Flow (matches the requested behaviour):
    1. The user's speech (any language / Tanglish / Hinglish) is sent
       to the LLM.
    2. The LLM internally translates the intent into ENGLISH search
       queries.
    3. It calls the `web_search` tool (DuckDuckGo, no API key) as many
       times as it needs to cover every part of the question.
    4. The tool results are fed back to the LLM.
    5. The LLM produces the final answer in the SAME language the user
       spoke (Tanglish -> Tanglish, Tamil -> Tamil, Hindi -> Hindi, ...).

The model does NOT access the internet itself. This file is the bridge
between the model and the search engine.
"""

import sys
import json
import os
from concurrent.futures import ThreadPoolExecutor

# Force UTF-8 on Windows terminals (cp1252 can't print emoji)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY")
OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
LLM_MODEL = os.environ.get("OPENCODE_MODEL", "nemotron-3-ultra-free")

MAX_SEARCH_RESULTS = 5
MAX_TOKENS = 1200
TEMPERATURE = 0.3
MAX_SEARCH_ROUNDS = 6  # max number of tool-call rounds per user query
MAX_PARALLEL_SEARCHES = 4  # searches run concurrently within one round


# ============================================================
# SEARCH ENGINE
# ============================================================

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()
SERPER_API_URL = "https://google.serper.dev/search"
SERPER_QPS_LIMIT = 10.0  # max requests/second; soft cap for short bursts


class SearchEngine:
    """
    Web search using the Serper.dev API when a SERPER_API_KEY is
    configured, with DuckDuckGo (ddgs) as a keyless fallback.

    Serper is a no-billing-required eval tier (2,500 free queries)
    and returns Google results fast over a single connection.

    One persistent httpx.Client is reused for every query so a single
    HTTPS connection pool serves all searches (no per-query TLS/session
    setup). Returns a list of dicts with 'title', 'url' and 'snippet'.
    """

    def __init__(self):

        import threading

        self._client = None
        self._client_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._rate_lock = threading.Lock()

    # -- connection management ----------------------------------

    def _get_client(self):
        """Return the shared httpx.Client, creating it once on first use."""

        if self._client is None:

            with self._client_lock:

                if self._client is None:

                    import httpx

                    self._client = httpx.Client(
                        timeout=httpx.Timeout(10.0, connect=5.0),
                        headers={
                            "Accept": "application/json",
                        },
                    )

        return self._client

    def close(self):
        """Close the shared connection pool."""

        if self._client is not None:

            with self._client_lock:

                if self._client is not None:

                    self._client.close()

                    self._client = None

    # -- search ---------------------------------------------------

    def search(self, query, max_results=MAX_SEARCH_RESULTS):
        """Search the web for `query` and return top results."""

        if SERPER_API_KEY:

            results = self._search_serper(query, max_results)

            if results:

                return results

            print("ℹ️ Serper returned nothing; falling back to DuckDuckGo.")

        return self._search_ddg(query, max_results)

    # -- Serper ---------------------------------------------------

    def _search_serper(self, query, max_results):
        """Prefer Serper (fast, one reused connection, no billing)."""

        import time

        try:

            # Soft throttle for parallel callers: keep roughly SERPER_QPS_LIMIT
            # requests/second, but never sleep longer than ~1s so short
            # bursts still go out immediately.
            with self._rate_lock:

                min_interval = 1.0 / SERPER_QPS_LIMIT

                wait = min_interval - (
                    time.monotonic() - self._last_request_ts
                )

                if wait > 0:

                    time.sleep(min(wait, 1.0))

                self._last_request_ts = time.monotonic()

            response = self._get_client().post(
                SERPER_API_URL,
                headers={
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": min(max_results, 20),
                    "gl": "in",
                    "hl": "en",
                },
            )

            response.raise_for_status()

            payload = response.json()

            raw_results = payload.get("organic") or []

            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in raw_results
                if item.get("link")
            ]

            return results[:max_results]

        except Exception as e:

            print(f"⚠️ Serper search failed: {e}")

            return []

    # -- DuckDuckGo fallback --------------------------------------

    def _search_ddg(self, query, max_results):
        """Keyless fallback (one persistent ddgs session, not per-query)."""

        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                raw_results = list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                    )
                )

            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
                for item in raw_results
                if item.get("body")
            ]

            return results

        except Exception as e:

            print(f"⚠️ DuckDuckGo search failed: {e}")

            return []

    def format_results(self, results):
        """Format search results into a readable context block."""

        if not results:

            return (
                "No web search results were found "
                "for this query."
            )

        lines = []

        for i, result in enumerate(results, start=1):

            lines.append(
                f"{i}. {result['title']}\n"
                f"   URL: {result['url']}\n"
                f"   {result['snippet']}"
            )

        return "\n\n".join(lines)


# ============================================================
# TOOL SCHEMA
# ============================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for fresh facts about government schemes, "
            "eligibility, age limits, application steps, amounts, etc. "
            "Use this when information is not found in the local document RAG knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query in ENGLISH. "
                        "Translate the user's intent to English first."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

RAG_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Search the local RAG knowledge base (PACS Bye-Laws, PMFBY guidelines, "
            "uploaded scheme documents, PDFs, scanned image text). "
            "Always call this tool FIRST for specific scheme guidelines or uploaded files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or search phrase in English or native language.",
                }
            },
            "required": ["query"],
        },
    },
}

GRIEVANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "file_grievance",
        "description": (
            "Initiate or update a formal citizen grievance form. "
            "Call this whenever the user wants to lodge a complaint, report loan delay, "
            "unpaid insurance claims, PACS issues, or request official dispute resolution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_complaint": {
                    "type": "string",
                    "description": "Summary of user's complaint or answer to follow-up question.",
                }
            },
            "required": ["user_complaint"],
        },
    },
}


# ============================================================
# LLM SERVICE
# ============================================================

class LLMService:
    """
    Search-agent that answers in the user's language.

    Pipeline:
        1. Build a system prompt (language rules + agent behaviour).
        2. Run an agent loop: the model decides when to call
           `web_search`, the service executes it and returns results.
        3. When the model has enough info it writes the final answer
           in the user's language.
    """

    def __init__(self):

        from datetime import datetime

        self.today_date = datetime.now().strftime("%d %B %Y")
        self.current_year = str(datetime.now().year)

        self.search_engine = SearchEngine()

        self.client = None

        if OPENCODE_API_KEY:

            self.client = OpenAI(
                api_key=OPENCODE_API_KEY,
                base_url=OPENCODE_BASE_URL,
                default_headers={"x-opencode-session": "multilingual-rag-session"},
            )

            print(
                f"LLM Service ready "
                f"(OpenCode Zen / {LLM_MODEL})."
            )

        else:

            print(
                "\n❌ OPENCODE_API_KEY is not configured."
            )

            print(
                "Set it before running the program:"
            )

            print(
                "export OPENCODE_API_KEY='your-key-here'"
            )

    def _build_system_prompt(
        self,
        base_system_prompt,
        language_instruction,
    ):

        return (
            base_system_prompt
            + f"""

TODAY'S DATE: {self.today_date}

SEARCH AGENT BEHAVIOUR
======================

You are a research agent with a `web_search` tool.

- The user may speak ANY language, including Tanglish / Hinglish
  (Indian languages written in Latin script).
- First UNDERSTAND the user's intent in their own language.
- Then TRANSLATE the intent into precise ENGLISH search queries.
- IMPORTANT: ALWAYS search for the LATEST data. Include the current
  year ({self.current_year}) in your search queries (for example
  "... {self.current_year} ..." or "... {self.current_year} latest ...").
  Prefer results dated in {self.current_year} or the most recent year
  available. Reject stale information if newer results contradict it.
- Call `web_search` as many times as needed to cover every part of
  the question. For multi-part questions, search each part separately.
- If the first results are thin, refine the query and search again.
- Once you have enough information, write the final answer.

FINAL ANSWER RULES
==================

- Write the final answer ONLY in the language the user spoke.
  (Tanglish -> Tanglish, Tamil -> Tamil script, Hindi -> Hindi
  script, English -> English, and so on.)
- Mirror the script style: if the user wrote Tamil in Latin letters
  (Tanglish), reply in Tanglish Latin letters.
- Keep the answer concise and conversational (2-4 short sentences)
  because it will be read aloud by a text-to-speech engine.
- Mention key facts like amounts, eligibility, age limits and
  how to apply when available.
- When citing amounts, dates or eligibility rules, use the LATEST
  figures from the search results. If a year is involved, state it
  clearly.
"""
            + language_instruction
        )

    def _run_agent_loop(self, system_prompt, user_text, prior_turns, session_id, cancel_event=None):
        """
        Run the tool-calling loop until the model answers or the
        round limit is reached. Returns the final text reply.
        """

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Prior multi-turn context (user/assistant pairs)
        messages.extend(prior_turns)

        messages.append(
            {
                "role": "user",
                "content": (
                    "USER MESSAGE:\n"
                    + user_text
                ),
            }
        )

        for _ in range(MAX_SEARCH_ROUNDS):

            if (
                cancel_event
                and cancel_event.is_set()
            ):
                return ""

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=[WEB_SEARCH_TOOL, RAG_SEARCH_TOOL, GRIEVANCE_TOOL],
                    tool_choice="auto",
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
            )

            message = response.choices[0].message

            tool_calls = getattr(
                message,
                "tool_calls",
                None,
            )

            # The model produced a final answer (no tool call)
            if not tool_calls:

                reply = (message.content or "").strip()

                if reply:

                    return reply

                # Empty response — nudge the model to retry instead
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You returned an empty reply. Please answer the "
                            "user's question now. Use the tools if "
                            "you need more information, then give your final "
                            "answer in the user's language."
                        ),
                    }
                )

                continue

            # Execute each requested tool call and feed results back
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            from rag_service import rag_service
            from grievance_service import grievance_service

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    query = args.get("query") or args.get("user_complaint") or user_text
                except Exception:
                    query = user_text

                if fn_name == "rag_search":
                    print(f"📚 RAG Searching: {query}")
                    rag_docs = rag_service.hybrid_search(query, top_k=4)
                    rag_content = rag_service.format_rag_context(rag_docs) or "No relevant local documents found."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": rag_content,
                    })
                elif fn_name == "file_grievance":
                    print(f"📝 Grievance Filing: {query}")
                    # [Fix 2] Pass session_id so draft persists across tool calls
                    res = grievance_service.process_grievance_turn(query, session_id=session_id)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": res["prompt"],
                    })
                elif fn_name == "web_search":
                    print(f"🔎 Web Searching: {query}")
                    web_results = self.search_engine.search(query)
                    web_content = self.search_engine.format_results(web_results)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": web_content,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Unknown tool function.",
                    })

        # Round limit reached without a final answer
        return ""

    def generate_response(
        self,
        user_text,
        base_system_prompt,
        language_instruction,
        chat_history,
        cancel_event=None,
    ):
        """
        Search the web (as many rounds as the model wants) and return
        the reply in the user's language.

        chat_history is mutated in place (user + assistant turns
        are appended) so multi-turn context is preserved.

        If cancel_event (a threading.Event) is set while searching,
        the loop stops early and returns "" (nothing is added to
        chat_history).
        """

        if self.client is None:

            return (
                "Sorry, the assistant is not configured. "
                "Please set the OPENCODE_API_KEY."
            )

        # ----------------------------------------
        # [Fix 2] Unique session ID for grievance draft tracking
        # ----------------------------------------
        import uuid as _uuid
        session_id = _uuid.uuid4().hex

        # ----------------------------------------
        # 1. Build the system prompt
        # ----------------------------------------

        system_prompt = self._build_system_prompt(
            base_system_prompt=base_system_prompt,
            language_instruction=language_instruction,
        )

        # ----------------------------------------
        # 2. Prior turns (everything except index 0)
        # ----------------------------------------

        prior_turns = chat_history[1:] if chat_history else []

        # ----------------------------------------
        # 3. Run the search-agent loop
        # ----------------------------------------

        try:

            reply = self._run_agent_loop(
                system_prompt=system_prompt,
                user_text=user_text,
                prior_turns=prior_turns,
                session_id=session_id,
                cancel_event=cancel_event,
            )

        except Exception as e:

            print(f"⚠️ LLM request failed: {e}")

            return (
                "Sorry, I couldn't reach "
                "the assistant service right now."
            )

        # Interrupted mid-search: discard the partial reply
        # and DO NOT remember this turn.
        if (
            cancel_event
            and cancel_event.is_set()
        ):

            return ""

        if not reply:

            # [Fix 8] Include language hint so TTS can read the message correctly
            reply = (
                "I couldn't find an answer for that right now. "
                "Please try rephrasing or ask again."
            )

        # ----------------------------------------
        # 4. Remember the user + assistant turns
        # ----------------------------------------

        chat_history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        return reply


# Shared instance used by the main chatbot loop
llm_service = LLMService()
