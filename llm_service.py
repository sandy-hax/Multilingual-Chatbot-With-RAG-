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

import json
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY")
OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
LLM_MODEL = os.environ.get("OPENCODE_MODEL", "deepseek-v4-flash-free")

MAX_SEARCH_RESULTS = 5
MAX_TOKENS = 1200
TEMPERATURE = 0.3
MAX_SEARCH_ROUNDS = 6  # max number of tool-call rounds per user query
MAX_PARALLEL_SEARCHES = 4  # searches run concurrently within one round


# ============================================================
# SEARCH ENGINE
# ============================================================

class SearchEngine:
    """
    Lightweight web search using DuckDuckGo (ddgs).

    Free, fast, and requires no API key. Returns a list of
    dicts with 'title', 'url' and 'snippet' for the top hits.
    """

    def search(self, query, max_results=MAX_SEARCH_RESULTS):
        """Search the web for `query` and return top results."""

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

            print(f"⚠️ Web search failed: {e}")

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
            "Search the web for information. Use this whenever you need "
            "fresh or detailed facts about government schemes, "
            "eligibility, age limits, application steps, amounts, etc. "
            "You may call it multiple times with different queries to "
            "gather everything the user asked about. Always include the "
            "current year in the query to get the latest data."
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

    def _run_agent_loop(self, system_prompt, user_text, prior_turns, cancel_event=None):
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
                    tools=[WEB_SEARCH_TOOL],
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
                # of giving up silently.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You returned an empty reply. Please answer the "
                            "user's question now. Use the web_search tool if "
                            "you need more information, then give your final "
                            "answer in the user's language."
                        ),
                    }
                )

                continue

            # Execute each requested search and feed results back
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

            jobs = []

            for tc in tool_calls:

                try:

                    args = json.loads(
                        tc.function.arguments
                        or "{}"
                    )

                    query = args.get("query", user_text)

                except Exception:

                    query = user_text

                jobs.append((tc, query))

            print(
                f"🔎 Searching x{len(jobs)} (parallel): "
                + " | ".join(q for _, q in jobs)
            )

            # Run all searches for this round concurrently.
            # Slowest single search now bounds the round, not the sum.
            # Results are stashed and appended in the ORIGINAL tool-call
            # order so tool messages always line up with tool_calls.
            ordered_results = [None] * len(jobs)

            with ThreadPoolExecutor(
                max_workers=min(
                    len(jobs),
                    MAX_PARALLEL_SEARCHES,
                )
            ) as pool:

                future_by_index = {}

                for index, (tc, query) in enumerate(jobs):

                    future_by_index[index] = pool.submit(
                        self.search_engine.search,
                        query,
                    )

                for index, future in future_by_index.items():

                    try:

                        ordered_results[index] = future.result()

                    except Exception as e:

                        print(f"⚠️ Search failed: {e}")

                        ordered_results[index] = []

            for index, (tc, query) in enumerate(jobs):

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            self.search_engine.format_results(
                                ordered_results[index]
                            )
                        ),
                    }
                )

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

            reply = (
                "I couldn't find an answer "
                "for that right now. Please try again."
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
