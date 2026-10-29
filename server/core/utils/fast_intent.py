"""
Rule-based fast answers (time / weather / location).

These questions have a fixed answer that doesn't need the LLM. Patterns are
deliberately strict (whole-utterance matches): anything ambiguous, such as
"does the weather affect my glucose?", falls through to the main LLM, which
also has the time and weather in its <context>.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

_END = r"[\s?？.。!！]*$"

_PATTERNS = {
    "time": [
        re.compile(r"^(?:what(?:'s| is)? the time|what time is it|what's the current time|time)(?: now| right now)?" + _END),
        re.compile(r"^(?:现在)?(?:几点了?|几点钟|什么时间)" + _END),
    ],
    "weather": [
        re.compile(r"^(?:how(?:'s| is)|what(?:'s| is)) the weather(?: like)?(?: today| now| outside| right now)?" + _END),
        re.compile(r"^(?:今天|现在)?(?:的)?天气(?:怎么样|如何|好吗)?" + _END),
    ],
    "location": [
        re.compile(r"^where am i(?: now| right now)?" + _END),
        re.compile(r"^我(?:现在)?在哪(?:里|儿)?" + _END),
    ],
}


def match_fast_intent(text: str) -> Optional[str]:
    """Return "time" | "weather" | "location" if the whole utterance is one of those questions."""
    normalized = (text or "").strip().lower()
    if not normalized or len(normalized) > 60:
        return None
    for intent, patterns in _PATTERNS.items():
        if any(p.match(normalized) for p in patterns):
            return intent
    return None


def get_fast_answer(intent: str, text: str, client_id: str, config: dict) -> Optional[str]:
    if intent == "time":
        return _time_answer(text, config)
    if intent == "weather":
        return _weather_answer(config)
    if intent == "location":
        return _location_answer(client_id)
    return None


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text or ""))


def _time_answer(text: str, config: dict) -> str:
    tz_offset = config.get("server", {}).get("timezone_offset", 0)
    if tz_offset == -5:
        tz = ZoneInfo("America/New_York")
    elif tz_offset == -8:
        tz = ZoneInfo("America/Los_Angeles")
    else:
        tz = timezone(timedelta(hours=tz_offset))
    now = datetime.now(tz)
    if _is_chinese(text):
        return f"现在是{now.strftime('%H:%M')}。"
    return f"It's {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %B %d')}."


def _weather_answer(config: dict) -> str:
    try:
        from plugins_func.functions.get_weather import get_weather

        location = config.get("plugins", {}).get("get_weather", {}).get("default_location", "")
        if not location:
            return "I don't have your location set."

        class _ConfigOnlyConn:
            def __init__(self, cfg):
                self.config = cfg

        result = get_weather(_ConfigOnlyConn(config), location=location)
        if result and getattr(result, "result", None):
            return result.result
        return f"I couldn't get the weather for {location}."
    except Exception as e:
        logger.bind(tag=TAG).error(f"Weather fast answer failed: {e}")
        return "Weather service is currently unavailable."


def _location_answer(client_id: str) -> str:
    try:
        config_path = os.path.join("data", client_id, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                location = json.load(f).get("location")
                if location:
                    return f"We're in {location}."
        return "Location not set in your config."
    except Exception:
        return "Unable to determine your location."
