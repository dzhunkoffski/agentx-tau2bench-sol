import json
import logging
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
import litellm
from langgraph.graph import StateGraph, START, END

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

log = logging.getLogger(__name__)

load_dotenv()

MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "2"))

SYSTEM_PROMPT = """\
You are a customer service agent. Follow the domain policy and tool instructions \
provided in the conversation exactly.

CRITICAL RULES:
1. ALWAYS respond with a single valid JSON object, nothing else.
2. Format: {"name": "<action>", "arguments": {<params>}}
3. Use tool names from the provided list, or "respond" to reply to the user.
4. ONE action per turn — never combine tool calls and responses.
5. Follow the domain policy strictly — do not deviate or make exceptions.
6. Output ONLY raw JSON — no markdown, no code blocks, no extra text.

TOOL-USE DISCIPLINE (most important):
- You MUST call tools to look up or verify any information before acting on it or
  reporting it to the user. Never assume or fabricate user IDs, booking numbers,
  account details, or prices.
- You MUST call the appropriate action tool (cancel, update, refund, etc.) to
  actually make a change. Telling the user "I've done X" without calling the tool
  is a hard failure.
- Only use {"name": "respond", ...} AFTER all required tools have been called and
  you are ready to give the user a final, accurate confirmation.
- If uncertain what to do next, call a tool to gather more information rather than
  responding to the user prematurely."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_to_json(text: str) -> str:
    """Extract JSON from a response that may contain markdown or extra text."""
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    if not text.startswith("{") and not text.startswith("["):
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1)
    return text


# ---------------------------------------------------------------------------
# LangGraph state & nodes
# ---------------------------------------------------------------------------

class TurnState(TypedDict):
    messages: list[dict]       # LLM message dicts (system + history + new)
    response: str              # Raw LLM response text
    parsed_response: str       # Cleaned JSON string
    retry_count: int           # Number of retries so far


RESPOND_ACTION_NAME = "respond"


def _extract_action(parsed: dict | list) -> dict | None:
    """Mirror the green agent's _parse_response logic: accept dict or list."""
    if isinstance(parsed, list):
        if not parsed:
            return None
        parsed = parsed[0]
    if isinstance(parsed, dict) and "name" in parsed:
        return parsed
    return None


async def call_llm(state: TurnState) -> dict:
    """Call the LLM with the current message history."""
    result = await litellm.acompletion(
        model=MODEL,
        messages=state["messages"],
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    content = result.choices[0].message.content
    updated_messages = state["messages"] + [{"role": "assistant", "content": content}]
    return {"messages": updated_messages, "response": content}


async def parse_response(state: TurnState) -> dict:
    """Strip markdown / extra text to extract raw JSON."""
    cleaned = strip_to_json(state["response"])

    # Log whether the LLM decided to call a tool or respond to the user.
    try:
        action = _extract_action(json.loads(cleaned))
        if action:
            name = action.get("name", "?")
            if name == RESPOND_ACTION_NAME:
                log.info("[purple-agent] ACTION=respond  content=%r",
                         action.get("arguments", {}).get("content", "")[:120])
            else:
                log.info("[purple-agent] ACTION=tool_call  tool=%r  args=%s",
                         name, json.dumps(action.get("arguments", {}))[:200])
    except Exception:
        log.warning("[purple-agent] Could not parse LLM output as action: %r",
                    cleaned[:200])

    return {"parsed_response": cleaned}


def check_json(state: TurnState) -> str:
    """Route to END if valid JSON action, otherwise retry or give up.

    Mirrors the green agent's _parse_response: accepts both dict and list.
    """
    try:
        action = _extract_action(json.loads(state["parsed_response"]))
        if action is not None:
            return "valid"
    except (json.JSONDecodeError, TypeError):
        pass
    if state.get("retry_count", 0) >= MAX_RETRIES:
        log.error("[purple-agent] Max retries reached — giving up. raw=%r",
                  state["parsed_response"][:200])
        return "give_up"
    log.warning("[purple-agent] Invalid JSON on retry %d, retrying…",
                state.get("retry_count", 0))
    return "invalid"


async def add_retry(state: TurnState) -> dict:
    """Append a corrective prompt and bump the retry counter."""
    retry_msg = {
        "role": "user",
        "content": (
            "Your response was not valid JSON with the required format. "
            'Respond with ONLY a JSON object: {"name": "...", "arguments": {...}}'
        ),
    }
    return {
        "messages": state["messages"] + [retry_msg],
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph agent graph.

    Graph layout:

        START ──► call_llm ──► parse_response ──► (check_json)
                                                    │  valid ──► END
                                                    │  give_up ──► END
                                                    └─ invalid ──► retry ──► call_llm
    """
    builder = StateGraph(TurnState)

    builder.add_node("call_llm", call_llm)
    builder.add_node("parse_response", parse_response)
    builder.add_node("retry", add_retry)

    builder.add_edge(START, "call_llm")
    builder.add_edge("call_llm", "parse_response")
    builder.add_conditional_edges(
        "parse_response",
        check_json,
        {"valid": END, "invalid": "retry", "give_up": END},
    )
    builder.add_edge("retry", "call_llm")

    return builder.compile()


# ---------------------------------------------------------------------------
# A2A Agent wrapper
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.history: list[dict] = []
        self.graph = build_graph()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        # First turn: initialise history with system prompt
        if not self.history:
            self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.history.append({"role": "user", "content": input_text})

        # Run one turn through the graph (retries are internal)
        result = await self.graph.ainvoke(
            {
                "messages": list(self.history),   # copy so retries don't leak
                "response": "",
                "parsed_response": "",
                "retry_count": 0,
            }
        )

        response_text = result["parsed_response"]

        # Keep only the final answer in persistent history (not retry prompts)
        self.history.append({"role": "assistant", "content": response_text})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="Response",
        )
