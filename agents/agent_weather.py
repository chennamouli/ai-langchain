#!/usr/bin/env python3
"""
Weather agent using Anthropic Claude.

This agent uses tool calling to:
1. Geocode a location (city name or ZIP code) to latitude/longitude using Open-Meteo.
2. Fetch current weather and hourly forecast using the coordinates.

Usage:
  python agents/agent_weather.py "San Francisco"
  python agents/agent_weather.py "78641"
  python agents/agent_weather.py --location "Austin, TX"

No API keys required; Open-Meteo is a free, rate-limited public API.
"""

import argparse
import json
import logging
import sys
from typing import Any, Optional

import requests
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# Add parent directory to path so we can import util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from util.util import (
    load_env,
    create_agent,
    handle_tool_calls_and_respond,
    extract_response_text,
    print_beautiful,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@tool
def geocode_location(location: str) -> dict:
    """Geocode a location (city name or ZIP code) to get latitude and longitude.

    Uses Open-Meteo Geocoding API (free, no auth required).
    Returns a dict with latitude, longitude, name, country, etc.
    Raises an error if the location is not found.
    """
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("results"):
            return {"error": f"Location '{location}' not found"}

        result = data["results"][0]
        response = {
            "name": result.get("name", "Unknown"),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "country": result.get("country", ""),
            "admin1": result.get("admin1", ""),
        }
        logger.info(f"Geocoded '{location}' to {response}")
        return response
    except Exception as e:
        return {"error": f"Geocoding failed: {e}"}


@tool
def get_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather and hourly forecast for given lat/lon coordinates.

    Uses Open-Meteo Forecast API (free, no auth required).
    Returns current temperature, weather code, and hourly forecast.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "hourly": "temperature_2m,precipitation",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        return {
            "current": data.get("current", {}),
            "hourly": {
                "time": data.get("hourly", {}).get("time", [])[:12],  # first 12 hours
                "temperature_2m": data.get("hourly", {}).get("temperature_2m", [])[:12],
                "precipitation": data.get("hourly", {}).get("precipitation", [])[:12],
            },
            "timezone": data.get("timezone", ""),
        }
    except Exception as e:
        return {"error": f"Weather lookup failed: {e}"}


def main():
    parser = argparse.ArgumentParser(
        description="Weather agent: ask Claude about weather for a location"
    )
    parser.add_argument(
        "--location", "-l", default="San Francisco", help="City name or ZIP code"
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help="Custom query (default is to get weather summary)",
    )
    args = parser.parse_args()

    # Load environment (for future API keys if needed)
    try:
        load_env()  # Load .env if present, no required vars
    except Exception as e:
        logger.warning("Could not load .env: %s", e)

    # Create agent (model + bound tools)
    tools = [geocode_location, get_weather]
    try:
        agent = create_agent(tools=tools, provider="auto", require_api_key=True)
    except Exception as e:
        print(f"Error creating agent: {e}")
        sys.exit(1)

    # Construct the user query
    query = args.query or f"What is the current weather in {args.location}? Also tell me about the next few hours."
    print(f"\nUser: {query}\n")

    # Invoke the model
    messages = [HumanMessage(content=query)]
    response = agent.invoke(messages)

    # Check if the model called any tools
    # Response extraction and pretty-printing moved to util.util

    if response.tool_calls:
        tool_map = {"geocode_location": geocode_location, "get_weather": get_weather}
        final_response = handle_tool_calls_and_respond(agent, messages, response, tool_map)
        print_beautiful(extract_response_text(final_response))
    else:
        print("Claude's response:\n")
        print_beautiful(extract_response_text(response))


if __name__ == "__main__":
    main()
