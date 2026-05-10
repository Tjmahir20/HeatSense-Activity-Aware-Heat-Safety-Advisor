"""
HeatSense agentic shift-plan generator.

Orchestrates a two-turn OpenAI chat flow: turn one selects validated tool calls
(WBGT forecast + guideline retrieval); turn two emits JSON parsed into
``ShiftPlan``. Tool JSON Schemas follow the Module 5 pattern; structured JSON
output follows the Module 3 pattern.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from openai import OpenAI

from .rag import retrieve
from .schema import ShiftPlan
from .weather import fetch_hourly_wbgt

# Loading ``.env`` so ``OPENAI_API_KEY`` is available before client construction.
load_dotenv()

_DISCLAIMER = (
    "HeatSense provides general guidance only. Consult a qualified occupational health "
    "professional for site-specific programs."
)


def _get_openai_client() -> OpenAI:
    """
    Construct an OpenAI client after verifying an API key exists.

    Returns:
        Configured ``openai.OpenAI`` instance.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is unset or empty.
    """
    # Reading the secret from the environment (never hard-coded in source).
    api_key = os.getenv("OPENAI_API_KEY")
    # Failing fast with a clear message instead of an opaque 401 later.
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
    # Returning a client bound to the user’s default OpenAI project settings.
    return OpenAI(api_key=api_key)


TOOL_FETCH_WBGT: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_wbgt_forecast",
        "description": (
            "Fetch the hourly Wet-Bulb Globe Temperature (WBGT) forecast for a work "
            "shift at a given location. Returns an hour-by-hour WBGT table (°C)."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Site latitude in decimal degrees"},
                "longitude": {"type": "number", "description": "Site longitude in decimal degrees"},
                "shift_start": {"type": "integer", "description": "Shift start hour 0-23"},
                "shift_end": {"type": "integer", "description": "Shift end hour 1-23, must exceed shift_start"},
                "direct_sun": {"type": "boolean", "description": "True if the work site is fully exposed to sun"},
            },
            "required": ["latitude", "longitude", "shift_start", "shift_end", "direct_sun"],
            "additionalProperties": False,
        },
    },
}

TOOL_SEARCH_GUIDELINES: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_heat_guidelines",
        "description": (
            "Search the NIOSH/OSHA heat safety knowledge base. Returns relevant passages "
            "on WBGT exposure limits, work/rest ratios, hydration, and heat illness."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic to look up, e.g. 'Heavy work WBGT ceiling and rest ratio'",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

TOOLS: List[Dict[str, Any]] = [TOOL_FETCH_WBGT, TOOL_SEARCH_GUIDELINES]

TOOL_SCHEMAS: Dict[str, Any] = {
    "fetch_wbgt_forecast": TOOL_FETCH_WBGT["function"]["parameters"],
    "search_heat_guidelines": TOOL_SEARCH_GUIDELINES["function"]["parameters"],
}


def _exec_fetch_wbgt(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the WBGT tool: call weather math and format a compact text table.

    Args:
        args: Validated tool arguments (lat/lon/shift/sun flags).

    Returns:
        Dict with key ``wbgt_table`` mapping to a multi-line hour/WBGT string.
    """
    # Delegating numeric WBGT computation to the weather pipeline module.
    hourly = fetch_hourly_wbgt(
        latitude=args["latitude"],
        longitude=args["longitude"],
        shift_start=args["shift_start"],
        shift_end=args["shift_end"],
        direct_sun=args["direct_sun"],
    )
    # Formatting each hour as a fixed-width clock label plus one decimal WBGT.
    rows = [f"{int(r['hour']):02d}:00 | {float(r['WBGT']):.1f}°C" for r in hourly]
    # Returning a JSON-serializable payload the model can read on turn two.
    return {"wbgt_table": "\n".join(rows)}


def _exec_search_guidelines(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the guideline search tool via hybrid RAG retrieval.

    Args:
        args: Validated tool arguments; must include ``query`` string.

    Returns:
        Dict with key ``context`` holding top passages joined by blank lines.
    """
    # Pulling a fixed number of passages so prompts stay bounded in size.
    context = retrieve(args["query"], k=4)
    # Wrapping the string so the chat API always sees JSON tool content.
    return {"context": context}


TOOL_EXECUTORS = {
    "fetch_wbgt_forecast": _exec_fetch_wbgt,
    "search_heat_guidelines": _exec_search_guidelines,
}


def _execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate tool arguments with JSON Schema, then dispatch to the executor.

    Args:
        tool_name: OpenAI function name emitted by the model.
        args:      Parsed JSON object of arguments for that function.

    Returns:
        Tool result dict from the matching executor.

    Raises:
        KeyError: If ``tool_name`` is not registered in ``TOOL_SCHEMAS``.
    """
    # Rejecting unknown tool names before validation to avoid silent mistakes.
    if tool_name not in TOOL_SCHEMAS:
        raise KeyError(f"Unknown tool: '{tool_name}'")
    # Validating argument shapes against the schema bundled with the tool spec.
    validate(instance=args, schema=TOOL_SCHEMAS[tool_name])
    # Dispatching to the Python callable registered for this tool name.
    return TOOL_EXECUTORS[tool_name](args)


def _parse_shift_plan(text: str) -> ShiftPlan:
    """
    Parse model JSON text into a strict ``ShiftPlan`` Pydantic model.

    Args:
        text: Raw assistant message body expected to be JSON only.

    Returns:
        Validated ``ShiftPlan`` instance.

    Raises:
        pydantic.ValidationError: If JSON is invalid or fails model checks.
    """
    # Choosing the Pydantic v2 API when available for stricter JSON parsing.
    if hasattr(ShiftPlan, "model_validate_json"):
        return ShiftPlan.model_validate_json(text)
    # Falling back to v1 ``parse_raw`` for older environments if needed.
    return ShiftPlan.parse_raw(text)


def run_agent(
    *,
    lat: float,
    lon: float,
    job_type: str,
    direct_sun: bool,
    shift_start: int,
    shift_end: int,
) -> ShiftPlan:
    """
    Run the two-turn agent: tool calls first, then structured shift-plan JSON.

    Turn one forces at least one tool invocation via ``tool_choice="required"``.
    Tool outputs are appended as ``role: tool`` messages. Turn two requests a
    JSON object response, then parses it into ``ShiftPlan``.

    Args:
        lat:          Site latitude in decimal degrees (already geocoded).
        lon:          Site longitude in decimal degrees.
        job_type:     One of Light / Moderate / Heavy / Very Heavy.
        direct_sun:   True when the outdoor sun WBGT path applies.
        shift_start:  Inclusive shift start hour (0–23).
        shift_end:    Exclusive shift end hour (1–23).

    Returns:
        Validated ``ShiftPlan`` ready for template rendering.

    Raises:
        EnvironmentError: When the OpenAI API key is missing.
        RuntimeError: When turn one produces no tool calls or bad tool JSON.
        pydantic.ValidationError: When turn-two JSON does not match ``ShiftPlan``.
    """
    # Opening a client shared by both chat completion rounds.
    client = _get_openai_client()

    # Building the system string that constrains tool usage and output shape.
    system_prompt = (
        "You are HeatSense, a NIOSH/OSHA-compliant heat safety advisor for outdoor workers.\n"
        "When given a shift planning request:\n"
        "  1. Call fetch_wbgt_forecast to obtain the hourly WBGT forecast.\n"
        "  2. Call search_heat_guidelines to retrieve work/rest and hydration guidance.\n"
        "  3. Using ONLY the tool results, return a single valid JSON object:\n"
        "     {\"worker_summary\": str,\n"
        "      \"hours\": [{\"hour\":int,\"wbgt\":float,\"risk_level\":str,\"risk_color\":str,\n"
        "                   \"work_rest_ratio\":str,\"water_intake\":str,\"notes\":str}],\n"
        "      \"disclaimer\": str}\n"
        "     risk_level ∈ {safe, caution, danger, stop_work}\n"
        "     risk_color ∈ {green, yellow, red, black}\n"
        f'     Always set disclaimer to: "{_DISCLAIMER}"\n'
        "  4. Do not add extra keys. Do not invent WBGT values."
    )

    # Composing the user goal from resolved coordinates and form-derived flags.
    goal = (
        f"Generate an hourly shift plan for {job_type} work "
        f"({'direct sun' if direct_sun else 'shade/covered'}) "
        f"at site ({lat:.4f}°, {lon:.4f}°), "
        f"shift window {shift_start:02d}:00 – {shift_end:02d}:00."
    )

    # Initializing the chat transcript with system + user roles only.
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]

    # Starting turn one: asking the model to emit tool calls (not free text).
    resp1 = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
    )

    # Extracting the assistant message object containing tool call metadata.
    assistant_msg = resp1.choices[0].message
    # Appending the assistant turn so follow-up tool rows attach correctly.
    messages.append(assistant_msg)

    # Verifying the model actually requested tools (defensive guard).
    if not assistant_msg.tool_calls:
        raise RuntimeError("Agent did not invoke any tools on turn 1.")

    # Iterating each tool invocation returned in parallel by the API.
    for tc in assistant_msg.tool_calls:
        tool_name = tc.function.name
        try:
            # Parsing the arguments JSON string into a Python dict.
            tool_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Agent returned malformed tool arguments: {exc}") from exc

        try:
            # Executing the tool after JSON Schema validation inside _execute_tool.
            result = _execute_tool(tool_name, tool_args)
        except ValidationError as exc:
            # Surfacing validation failures as tool-visible errors for turn two.
            result = {"error": f"Argument validation failed: {exc.message}"}
        except Exception as exc:
            # Catching any runtime error so one bad tool does not crash the request.
            result = {"error": str(exc)}

        # Appending the OpenAI-required tool message shape with linkage by ID.
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            }
        )

    # Starting turn two: requesting strict JSON matching ShiftPlan fields.
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )

    # Reading the final assistant text channel (may be empty on edge failures).
    content = resp2.choices[0].message.content or ""
    # Parsing and validating JSON into the Pydantic return type for Flask.
    return _parse_shift_plan(content)
