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
MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "4"))


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Call 4 — action execution: clean JSON-only prompt, reasoning arrives as context
SYSTEM_PROMPT = """\
You are a customer service agent executing a pre-determined action plan.

CRITICAL RULES:
1. ALWAYS respond with a single valid JSON object, nothing else.
2. Format: {"name": "<action>", "arguments": {<params>}}
3. Use tool names from the provided list, or "respond" to reply to the user.
4. ONE action per turn — never combine tool calls and responses.
5. Follow the domain policy strictly — do not deviate or make exceptions.
6. Output ONLY raw JSON — no markdown, no code blocks, no extra text.
7. Execute the ACTION PLAN provided to you exactly — do not deviate from it."""

# Call 1 — situation analysis
SITUATION_PROMPT = """\
You are analyzing a customer service conversation to assess the current situation.

Review the full conversation history and answer concisely:
1. What did the most recent message contain? (tool result, user message, or error)
2. What steps has the agent already completed?
3. What is the customer's ultimate goal?

Be factual and specific. 3-5 sentences maximum."""

# Call 2 — policy resolution
POLICY_PROMPT = """\
You are a policy compliance expert reviewing a customer service interaction.

The conversation history contains the full domain policy document.
Review it carefully and answer:
1. Which specific policy rule(s) apply to the customer's request?
2. Is the request permitted under the policy? State yes or no and why.
3. What conditions, limits, or constraints apply?

Be precise. Quote the relevant policy text where possible. 3-5 sentences maximum."""

# Call 3 — action planning
PLAN_PROMPT = """\
You are a task planner for a customer service agent.

Based on the situation and policy analysis, determine the single best next action.

Decision rules (apply in order):
1. If required information has NOT been looked up yet → call the appropriate lookup tool
2. If a change is needed AND the lookup was done → call the action tool \
(cancel / update / refund / etc.)
3. If the action completed successfully → respond to the customer to confirm
4. If the request violates policy → respond to decline and explain why
5. If a tool returned an error → diagnose and try a corrected approach

State your plan in ONE sentence: name the exact tool to call with its key parameters,
OR state that the agent should respond and summarize what to say."""

# Layer 3 — pre-respond critic
CRITIC_SYSTEM_PROMPT = """\
You are a quality-control reviewer for a customer service agent.
You will be shown a conversation history and the agent's proposed final response.
Your job is to verify the agent completed ALL required steps before responding.

Check ALL of the following:
1. Did the agent look up the relevant record (order, booking, account) with a tool?
2. Did the agent call the appropriate action tool (cancel, update, refund, etc.)
   if the user requested a change?
3. Did the last action tool return a success result — not an error?
4. Is the proposed response accurate and consistent with what the tools returned?

Respond with ONLY a JSON object — no other text:
{"ok": true}
OR
{"ok": false, "reason": "<which specific step is missing or incorrect>"}"""


# ---------------------------------------------------------------------------
# Error detection helpers
# ---------------------------------------------------------------------------

ERROR_KEYWORDS = [
    "error", "failed", "failure", "invalid", "not found", "exception",
    "cannot", "can't", "unable", "denied", "rejected", "unauthorized",
    "does not exist", "no such", "bad request", "missing required",
]


def _last_user_message(messages: list[dict]) -> str | None:
    """Return the content of the last user message, or None."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def _detect_tool_error(text: str) -> bool:
    """Return True if the message looks like a tool error response."""
    text_lower = text.lower()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if "error" in parsed:
                return True
            if parsed.get("status") in ("error", "failed", "failure"):
                return True
    except Exception:
        pass
    return any(kw in text_lower for kw in ERROR_KEYWORDS)


# ---------------------------------------------------------------------------
# JSON helpers
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
# Shared LLM call helper
# ---------------------------------------------------------------------------

async def _llm_call(messages: list[dict]) -> str:
    """Single LLM completion, returns raw content string."""
    result = await litellm.acompletion(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return result.choices[0].message.content


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class TurnState(TypedDict):
    messages: list[dict]         # Messages for Call 4 (augmented with reasoning)
    response: str                # Raw LLM response from Call 4
    parsed_response: str         # Cleaned JSON string (think field stripped)
    retry_count: int             # Retries for JSON errors + verification failures
    tools_called: list[str]      # Tool names called so far in this conversation
    verification_passed: bool    # Whether the critic approved the last respond
    situation: str               # Call 1 output
    policy: str                  # Call 2 output
    plan: str                    # Call 3 output


RESPOND_ACTION_NAME = "respond"


def _extract_action(parsed: dict | list) -> dict | None:
    """Mirror the green agent's _parse_response logic: accept dict or list.

    Strips the ``think`` field so it never reaches the evaluator.
    """
    if isinstance(parsed, list):
        if not parsed:
            return None
        parsed = parsed[0]
    if isinstance(parsed, dict) and "name" in parsed:
        parsed.pop("think", None)
        return parsed
    return None


# ---------------------------------------------------------------------------
# Layer 2: Tool error detection
# ---------------------------------------------------------------------------

async def check_tool_error(state: TurnState) -> dict:
    """Detect a tool error in the latest message and inject a recovery hint.

    Only activates after at least one tool has been called, so the initial
    policy message from the green agent is never flagged as an error.
    """
    if not state.get("tools_called"):
        return {}

    last_msg = _last_user_message(state["messages"])
    if last_msg and _detect_tool_error(last_msg):
        log.warning("[purple-agent] Tool error detected — injecting recovery hint.")
        hint = {
            "role": "user",
            "content": (
                "⚠️ SYSTEM NOTE: The previous tool call returned an error. "
                "Do NOT proceed as if it succeeded. "
                "The situation analysis must note this error explicitly. "
                "The action plan must describe a concrete recovery approach."
            ),
        }
        return {"messages": state["messages"] + [hint]}
    return {}


# ---------------------------------------------------------------------------
# Layer 1: Reasoning architecture — Calls 1, 2, 3
# ---------------------------------------------------------------------------

async def analyze_situation(state: TurnState) -> dict:
    """Call 1 — determine what just happened and where we are in the task."""
    messages = [
        {"role": "system", "content": SITUATION_PROMPT},
        {
            "role": "user",
            "content": (
                f"Conversation history "
                f"(last {min(len(state['messages']), 20)} messages):\n"
                f"{json.dumps(state['messages'][-20:], indent=2)}"
            ),
        },
    ]
    situation = await _llm_call(messages)
    log.info("[purple-agent] SITUATION: %s", situation[:300])
    return {"situation": situation}


async def check_policy(state: TurnState) -> dict:
    """Call 2 — resolve which policy rules apply and whether the request is allowed."""
    messages = [
        {"role": "system", "content": POLICY_PROMPT},
        {
            "role": "user",
            "content": (
                f"Conversation history "
                f"(last {min(len(state['messages']), 20)} messages):\n"
                f"{json.dumps(state['messages'][-20:], indent=2)}\n\n"
                f"Situation analysis:\n{state['situation']}"
            ),
        },
    ]
    policy = await _llm_call(messages)
    log.info("[purple-agent] POLICY: %s", policy[:300])
    return {"policy": policy}


async def plan_action(state: TurnState) -> dict:
    """Call 3 — decide the single next action and inject reasoning into messages.

    The three reasoning outputs are appended to ``state["messages"]`` as
    explicit grounded context for Call 4.  They are ephemeral — they never
    enter the persistent per-conversation history stored in ``Agent``.
    """
    messages = [
        {"role": "system", "content": PLAN_PROMPT},
        {
            "role": "user",
            "content": (
                f"Conversation history "
                f"(last {min(len(state['messages']), 20)} messages):\n"
                f"{json.dumps(state['messages'][-20:], indent=2)}\n\n"
                f"Situation:\n{state['situation']}\n\n"
                f"Policy:\n{state['policy']}"
            ),
        },
    ]
    plan = await _llm_call(messages)
    log.info("[purple-agent] PLAN: %s", plan[:300])

    # Inject all three reasoning outputs as grounded context for Call 4.
    # Placed as user messages so they cannot be ignored or contradicted.
    reasoning_context = [
        {
            "role": "user",
            "content": f"=== SITUATION ANALYSIS ===\n{state['situation']}",
        },
        {
            "role": "user",
            "content": f"=== POLICY ANALYSIS ===\n{state['policy']}",
        },
        {
            "role": "user",
            "content": (
                f"=== ACTION PLAN ===\n{plan}\n\n"
                f"Execute this plan now. "
                f'Output ONLY the JSON action: {{"name": "...", "arguments": {{...}}}}'
            ),
        },
    ]
    return {"plan": plan, "messages": state["messages"] + reasoning_context}


# ---------------------------------------------------------------------------
# Call 4: Action execution
# ---------------------------------------------------------------------------

async def call_llm(state: TurnState) -> dict:
    """Call 4 — execute the action plan as a single JSON action."""
    content = await _llm_call(state["messages"])
    updated_messages = state["messages"] + [{"role": "assistant", "content": content}]
    return {"messages": updated_messages, "response": content}


async def parse_response(state: TurnState) -> dict:
    """Strip markdown / extra text, track tool calls, strip the think field."""
    cleaned = strip_to_json(state["response"])
    tools_called = list(state.get("tools_called") or [])

    try:
        parsed = json.loads(cleaned)
        action = _extract_action(parsed)
        if action:
            name = action.get("name", "?")
            if name == RESPOND_ACTION_NAME:
                log.info("[purple-agent] ACTION=respond  content=%r",
                         action.get("arguments", {}).get("content", "")[:120])
            else:
                log.info("[purple-agent] ACTION=tool_call  tool=%r  args=%s",
                         name, json.dumps(action.get("arguments", {}))[:200])
                tools_called.append(name)
            cleaned = json.dumps(parsed if isinstance(parsed, list) else action)
    except Exception:
        log.warning("[purple-agent] Could not parse LLM output as action: %r",
                    cleaned[:200])

    return {"parsed_response": cleaned, "tools_called": tools_called}


def check_json(state: TurnState) -> str:
    """Route based on JSON validity and action type.

    valid   → tool call confirmed, forward to green agent
    verify  → agent wants to respond, run critic first  (Layer 3)
    invalid → JSON malformed, retry
    give_up → max retries exhausted
    """
    try:
        parsed = json.loads(state["parsed_response"])
        action = _extract_action(parsed)
        if action is not None:
            args = action.get("arguments")
            if args is not None and not isinstance(args, dict):
                log.warning("[purple-agent] 'arguments' is not a dict — retrying")
            else:
                if action.get("name") == RESPOND_ACTION_NAME:
                    return "verify"
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
    """Append a corrective prompt for a JSON formatting error and bump counter."""
    failed_output = state.get("response", "")[:500]
    retry_msg = {
        "role": "user",
        "content": (
            f"Your response was not valid JSON with the required format. "
            f"You said:\n{failed_output}\n\n"
            f"Fix it. Respond with ONLY a JSON object: "
            f'{{"name": "...", "arguments": {{...}}}}'
        ),
    }
    return {
        "messages": state["messages"] + [retry_msg],
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Layer 3: Pre-respond critic
# ---------------------------------------------------------------------------

async def verify_respond(state: TurnState) -> dict:
    """Second LLM call to verify all required steps were completed.

    Runs only when the agent emits a ``respond`` action.
    Fail-open: if the critic call itself errors, the response is approved.
    """
    critic_messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Conversation (last {min(len(state['messages']), 20)} messages):\n"
                f"{json.dumps(state['messages'][-20:], indent=2)}\n\n"
                f"Tools called so far: {state.get('tools_called', [])}\n\n"
                f"Agent's proposed response:\n{state['parsed_response']}"
            ),
        },
    ]
    try:
        content = await _llm_call(critic_messages)
        verdict = json.loads(strip_to_json(content))

        if verdict.get("ok") is True:
            log.info("[purple-agent] Critic approved respond.")
            return {"verification_passed": True}

        reason = verdict.get("reason", "unknown issue")
        log.warning("[purple-agent] Critic rejected respond: %s", reason)
        correction = {
            "role": "user",
            "content": (
                f"Do not respond to the user yet. "
                f"You missed a required step: {reason}\n"
                f"Complete that step first, then respond."
            ),
        }
        return {
            "messages": state["messages"] + [correction],
            "verification_passed": False,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    except Exception as exc:
        log.warning("[purple-agent] Critic call failed (%s) — approving.", exc)
        return {"verification_passed": True}


def route_after_verify(state: TurnState) -> str:
    """After critic: approved → end, rejected → re-run call_llm."""
    if state.get("verification_passed", True):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        log.error("[purple-agent] Critic rejected too many times — giving up.")
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph agent graph.

    Full layout per turn:

        START
          │
          ▼
      check_tool_error          ← Layer 2: detect & annotate tool errors
          │
          ▼
      analyze_situation         ← Call 1: what just happened?
          │
          ▼
      check_policy              ← Call 2: what does policy say?
          │
          ▼
      plan_action               ← Call 3: what is the next step?
          │                        (injects reasoning context into messages)
          ▼
        call_llm                ← Call 4: execute plan as JSON action
          │
          ▼
      parse_response
          │
          ▼
       check_json
          ├──► "valid"   ──────────────────────────────────────────► END
          │
          ├──► "verify"  ──► verify_respond   ← Layer 3: critic call
          │                       ├── "end"  ────────────────────── END
          │                       └── "retry" ──────────────────────┐
          │                                                          │
          ├──► "invalid" ──► add_retry ───────────────────────────► call_llm
          │
          └──► "give_up" ──────────────────────────────────────────► END
    """
    builder = StateGraph(TurnState)

    builder.add_node("check_tool_error", check_tool_error)
    builder.add_node("analyze_situation", analyze_situation)
    builder.add_node("check_policy", check_policy)
    builder.add_node("plan_action", plan_action)
    builder.add_node("call_llm", call_llm)
    builder.add_node("parse_response", parse_response)
    builder.add_node("retry", add_retry)
    builder.add_node("verify_respond", verify_respond)

    builder.add_edge(START, "check_tool_error")
    builder.add_edge("check_tool_error", "analyze_situation")
    builder.add_edge("analyze_situation", "check_policy")
    builder.add_edge("check_policy", "plan_action")
    builder.add_edge("plan_action", "call_llm")
    builder.add_edge("call_llm", "parse_response")
    builder.add_conditional_edges(
        "parse_response",
        check_json,
        {"valid": END, "verify": "verify_respond", "invalid": "retry", "give_up": END},
    )
    builder.add_edge("retry", "call_llm")
    builder.add_conditional_edges(
        "verify_respond",
        route_after_verify,
        {"end": END, "retry": "call_llm"},
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# A2A Agent wrapper
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.history: list[dict] = []
        self.tools_called: list[str] = []
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

        # Run one turn through the graph (retries are internal).
        # state["messages"] is a working copy — reasoning context and retry
        # messages are added inside the graph but never written back to
        # self.history, keeping the persistent history clean.
        result = await self.graph.ainvoke(
            {
                "messages": list(self.history),
                "response": "",
                "parsed_response": "",
                "retry_count": 0,
                "tools_called": list(self.tools_called),
                "verification_passed": True,
                "situation": "",
                "policy": "",
                "plan": "",
            }
        )

        response_text = result["parsed_response"]

        # Persist tool tracking across turns
        self.tools_called = result.get("tools_called", self.tools_called)

        # Only the final JSON action enters the persistent history —
        # reasoning context and retry messages remain ephemeral.
        self.history.append({"role": "assistant", "content": response_text})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="Response",
        )
