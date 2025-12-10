import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
#!/usr/bin/env python3
"""
A LangChain agent using Claude (Anthropic) as the LLM.

This agent demonstrates basic usage of Claude via LangChain with tool calling.

Run:
  cp .env.template .env
  # edit .env and set ANTHROPIC_API_KEY
  python agents/claude_agent.py
"""


from util.util import (
    load_env,
    create_anthropic_model,
    create_agent,
    extract_response_text,
    print_beautiful,
)
from langchain.tools import tool
from langchain_core.messages import HumanMessage


# Load environment and create model using util
load_env()

# Create agent with tools using util
tools = []



# Define simple tools for demonstration

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

# tools = [add_numbers, multiply_numbers]



def main():
    """Run a simple Claude agent interaction using util helpers."""
    agent = create_agent(tools=tools, provider="anthropic", require_api_key=True)
    messages = [HumanMessage(content="What is 5 + 3 and then multiply the result by 2?")]
    print("User: What is 5 + 3 and then multiply the result by 2?\n")
    response = agent.invoke(messages)
    print_beautiful(extract_response_text(response))
    if getattr(response, "tool_calls", None):
        print("Claude called the following tools:")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call['name']}({tool_call['args']})")
    else:
        print("Claude responded without calling tools.")


if __name__ == "__main__":
    main()
