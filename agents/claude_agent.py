#!/usr/bin/env python3
"""
A LangChain agent using Claude (Anthropic) as the LLM.

This agent demonstrates basic usage of Claude via LangChain with tool calling.

Run:
  cp .env.template .env
  # edit .env and set ANTHROPIC_API_KEY
  python agents/claude_agent.py
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# Load .env from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Ensure API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("No ANTHROPIC_API_KEY set. Copy `.env.template` to `.env` and set the key.")
    raise SystemExit(1)

os.environ["ANTHROPIC_API_KEY"] = api_key

model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
# Initialize Claude via LangChain
llm = ChatAnthropic(model=model, temperature=0)


# Define simple tools for demonstration
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


def main():
    """Run a simple Claude agent interaction."""
    
    # Create tools list
    tools = [add_numbers, multiply_numbers]
    
    # Create the agent with tool binding
    llm_with_tools = llm.bind_tools(tools)
    
    # Simple conversation
    messages = [
        HumanMessage(content="What is 5 + 3 and then multiply the result by 2?")
    ]
    
    print("User: What is 5 + 3 and then multiply the result by 2?\n")
    
    response = llm_with_tools.invoke(messages)
    
    print(f"Claude response:\n{response}\n")
    
    # If Claude suggested using tools, process them
    if response.tool_calls:
        print("Claude called the following tools:")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call['name']}({tool_call['args']})")
    else:
        print("Claude responded without calling tools.")


if __name__ == "__main__":
    main()
