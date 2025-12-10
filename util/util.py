"""Utility helpers for the workspace.

This module provides small helper functions used by agents and scripts:
- `load_env` — load environment variables from a `.env` file (searches sensible
  locations if a path is not provided) and validate required keys.
- `create_anthropic_model` — create and return an Anthropic/Claude model
  instance in a way that's robust across common LangChain / adapter versions.
- `create_openai_model` — helper to create an OpenAI Chat model (small wrapper).

The implementation prefers lazy imports to avoid hard runtime dependencies at
import-time; errors are surfaced as clear exceptions.

Usage:
    from util.util import load_env, create_anthropic_model
    load_env(required_vars=["ANTHROPIC_API_KEY"])  # loads .env if present
    model = create_anthropic_model(temperature=0.7)

"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv as _load_dotenv

logger = logging.getLogger(__name__)


def _find_dotenv(dotenv_path: Optional[Path | str] = None) -> Optional[Path]:
    """Resolve a .env file path.

    Order of lookup when `dotenv_path` is None:
    - current working directory / .env
    - repository root (parent of this file) / .env

    Returns the first matching Path, or None when no file is found.
    """
    if dotenv_path:
        p = Path(dotenv_path)
        return p if p.is_file() else None

    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [cwd / ".env", repo_root / ".env"]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def load_env(
    dotenv_path: Optional[Path | str] = None,
    required_vars: Optional[Iterable[str]] = None,
    override: bool = False,
) -> Dict[str, str]:
    """Load environment variables from a .env file and validate required keys.

    Parameters
    - dotenv_path: explicit path to a .env file. If omitted, a search is performed.
    - required_vars: an iterable of environment variable names that must be present
      after loading. If any are missing a ``ValueError`` is raised.
    - override: pass to `python-dotenv.load_dotenv` to allow `.env` to overwrite
      existing environment variables.

    Returns
    - A dict mapping variable names (only those present/required) to their values.
    """
    dotenv_file = _find_dotenv(dotenv_path)
    if dotenv_file:
        logger.debug("Loading environment from %s", dotenv_file)
        _load_dotenv(dotenv_path=str(dotenv_file), override=override)
    else:
        logger.debug("No .env file found (searched current dir and repo root)")

    result: Dict[str, str] = {}
    if required_vars:
        missing: List[str] = []
        for name in required_vars:
            val = os.getenv(name)
            if val is None:
                missing.append(name)
            else:
                result[name] = val
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    else:
        # Return a small set of commonly used keys if present.
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DATABASE_URL"):
            val = os.getenv(key)
            if val is not None:
                result[key] = val

    return result


def create_anthropic_model(
    model_name: Optional[str] = None, temperature: float = 0.0, require_api_key: bool = True
) -> Any:
    """Instantiate an Anthropic (Claude) chat model for use with LangChain.

    This function attempts to find a compatible model class in the following
    order (lazy imports):
    1. `langchain_anthropic.ChatAnthropic` (recommended adapter package)
    2. `langchain.chat_models.Anthropic` (LangChain's built-in wrapper, if present)

    It requires `ANTHROPIC_API_KEY` to be set in the environment unless
    `require_api_key=False`.

    Returns an instantiated model object (the exact type depends on installed
    packages). Raises RuntimeError if no compatible client is available.
    """
    # Lazy import chain to avoid hard dependency at import time
    ChatAnthropic = None
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore
    except Exception:
        try:
            from langchain.chat_models import Anthropic as ChatAnthropic  # type: ignore
        except Exception:
            ChatAnthropic = None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        if require_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Call load_env(...) or set the environment variable."
            )
        return None

    # Keep the env var available for downstream SDKs
    os.environ["ANTHROPIC_API_KEY"] = api_key

    if ChatAnthropic is None:
        raise RuntimeError(
            "No Anthropic/Claude client available. Install `langchain-anthropic` or ensure LangChain exposes an Anthropic model class."
        )

    model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    # Construct with common arguments; adapter classes should accept these.
    return ChatAnthropic(model=model, temperature=temperature)


def create_openai_model(temperature: float = 0.0, require_api_key: bool = True) -> Any:
    """Create a ChatOpenAI model instance (LangChain wrapper).

    This helper lazily imports `ChatOpenAI` from `langchain.chat_models` and
    ensures `OPENAI_API_KEY` is present when `require_api_key=True`.
    """
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("ChatOpenAI not available in installed LangChain") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if require_api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Call load_env(...) or set it manually.")
        return None

    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(temperature=temperature)


__all__ = ["load_env", "create_anthropic_model", "create_openai_model", "handle_tool_calls_and_respond"]


def extract_response_text(resp: Any) -> Optional[str]:
    """Return the assistant's textual content from various response shapes.

    This mirrors the logic used by agent scripts to extract a human-friendly
    text body from objects returned by LangChain / Anthropic adapters.
    """
    if resp is None:
        return None
    if hasattr(resp, "content"):
        return resp.content
    if hasattr(resp, "text"):
        return resp.text
    gens = getattr(resp, "generations", None)
    if gens:
        try:
            first = gens[0]
            if isinstance(first, list):
                first = first[0]
            if hasattr(first, "text"):
                return first.text
        except Exception:
            pass
    msg = getattr(resp, "message", None) or getattr(resp, "messages", None)
    if msg:
        try:
            if hasattr(msg, "content"):
                return msg.content
            if isinstance(msg, list) and len(msg) > 0 and hasattr(msg[0], "content"):
                return msg[0].content
        except Exception:
            pass
    try:
        return str(resp)
    except Exception:
        return None


def print_beautiful(text: Optional[str]) -> None:
    """Nicely print multi-line or markdown assistant text to the console.

    Uses `rich` when available for prettier rendering and falls back to
    simple framed plain text output when `rich` is not installed.
    """
    if not text:
        print("(no response)")
        return

    try:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.rule("Assistant")
        console.print(Markdown(text))
        console.rule()
    except Exception:
        print("\n" + "=" * 40)
        print(text)
        print("=" * 40 + "\n")

__all__.extend(["extract_response_text", "print_beautiful"])


def create_agent(
    tools: list,
    provider: str = "auto",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    require_api_key: bool = True,
) -> Any:
    """Create and return a tool-bound agent (model with tools attached).

    Parameters
    - tools: list of LangChain `tool` objects to bind to the model.
    - provider: "anthropic", "openai", or "auto". When "auto", Anthropic is
      preferred if `ANTHROPIC_API_KEY` is present, otherwise OpenAI is used.
    - model_name: optional model id to pass to the underlying model creator.
    - temperature: sampling temperature passed to the model constructor.
    - require_api_key: whether to require an API key for the chosen provider.

    Returns
    - A model instance with `.bind_tools(tools)` already applied.
    """
    provider_choice = provider
    if provider_choice == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            provider_choice = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider_choice = "openai"
        else:
            provider_choice = "anthropic"  # default to anthropic, let create_* raise if missing

    if provider_choice == "anthropic":
        model = create_anthropic_model(model_name=model_name, temperature=temperature, require_api_key=require_api_key)
    elif provider_choice == "openai":
        model = create_openai_model(temperature=temperature, require_api_key=require_api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Bind tools and return
    try:
        bound = model.bind_tools(tools)
    except Exception as e:
        raise RuntimeError(f"Failed to bind tools to model: {e}") from e
    return bound


def handle_tool_calls_and_respond(model_with_tools: Any, messages: list, response: Any, tool_map: Dict[str, Any]) -> Any:
    """Execute tool calls returned in `response`, send results back to the model, and return the final response.

    - `model_with_tools` is the LangChain model instance with tools bound.
    - `messages` is the original list of messages sent to the model.
    - `response` is the assistant response object that may contain `tool_calls`.
    - `tool_map` maps tool names to callables that accept the tool args dict.

    This helper prints tool usage and results, constructs `ToolMessage` blocks
    expected by Anthropic / LangChain, invokes the model with the tool results,
    and handles one level of recursive tool calls (if the model calls more tools
    after receiving results).
    """
    # Lazy import to avoid hard dependency
    try:
        from langchain_core.messages import ToolMessage, HumanMessage
    except Exception:  # pragma: no cover - defensive
        # Fallback: try older import path
        from langchain.messages import ToolMessage, HumanMessage  # type: ignore

    # Collect and execute tool calls
    tool_results = []
    if not getattr(response, "tool_calls", None):
        return response

    print("Claude is using tools...\n")
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool_use_id = tool_call.get("id")
        print(f"  → {tool_name}({tool_args})")

        tool_fn = tool_map.get(tool_name)
        if tool_fn:
            # If the tool is a StructuredTool (from langchain) it will expose the
            # underlying python function as `.func`. Prefer calling that. If not
            # present, try callable, then fall back to `.run` if available.
            callable_fn = None
            if hasattr(tool_fn, "func"):
                callable_fn = getattr(tool_fn, "func")
            elif callable(tool_fn):
                callable_fn = tool_fn
            elif hasattr(tool_fn, "run"):
                callable_fn = getattr(tool_fn, "run")

            if callable_fn:
                try:
                    result = callable_fn(**tool_args)
                    print(f"     Result: {result}")
                    tool_results.append((tool_use_id, tool_name, tool_args, result))
                except Exception as e:
                    tool_results.append((tool_use_id, tool_name, tool_args, {"error": str(e)}))
            else:
                tool_results.append((tool_use_id, tool_name, tool_args, {"error": "Tool not callable"}))
        else:
            tool_results.append((tool_use_id, tool_name, tool_args, {"error": "Unknown tool"}))

    # Build message chain with tool_result blocks
    tool_messages: list = [response]
    for tool_use_id, tool_name, tool_args, result in tool_results:
        tool_msg = ToolMessage(content=json.dumps(result), tool_call_id=tool_use_id)
        tool_messages.append(tool_msg)

    # Ask for final response
    tool_messages.append(HumanMessage(content="Based on the tool results above, provide a helpful weather summary."))

    # Invoke again with tool results
    final_response = model_with_tools.invoke(messages + tool_messages)

    # If the model called more tools, execute them and ask for final summary again
    if getattr(final_response, "tool_calls", None):
        print("Claude is using more tools...\n")
        more_results = []
        for tool_call in final_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_use_id = tool_call.get("id")
            print(f"  → {tool_name}({tool_args})")
            tool_fn = tool_map.get(tool_name)
            if tool_fn:
                callable_fn = None
                if hasattr(tool_fn, "func"):
                    callable_fn = getattr(tool_fn, "func")
                elif callable(tool_fn):
                    callable_fn = tool_fn
                elif hasattr(tool_fn, "run"):
                    callable_fn = getattr(tool_fn, "run")

                if callable_fn:
                    try:
                        result = callable_fn(**tool_args)
                        print(f"     Result: {result}")
                        more_results.append((tool_use_id, tool_name, tool_args, result))
                    except Exception as e:
                        more_results.append((tool_use_id, tool_name, tool_args, {"error": str(e)}))
                else:
                    more_results.append((tool_use_id, tool_name, tool_args, {"error": "Tool not callable"}))

        # Append final_response and the more tool_result blocks then ask for summary
        tool_messages.append(final_response)
        for tool_use_id, tool_name, tool_args, result in more_results:
            tool_msg = ToolMessage(content=json.dumps(result), tool_call_id=tool_use_id)
            tool_messages.append(tool_msg)
        tool_messages.append(HumanMessage(content="Now provide the weather summary based on all the data."))
        final_response = model_with_tools.invoke(messages + tool_messages)

    return final_response



