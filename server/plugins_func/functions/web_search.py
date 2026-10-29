import asyncio
from typing import TYPE_CHECKING

from config.logger import setup_logging
from core.utils.web_search import perform_web_search
from plugins_func.register import register_function, ToolType, ActionResponse, Action

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

MAX_RESULT_CHARS = 2000

WEB_SEARCH_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for external knowledge, public information or recent news. Do NOT use it for "
            "greetings, small talk, questions about yourself, or the user's own glucose / pump data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, in the user's language",
                }
            },
            "required": ["query"],
        },
    },
}


@register_function("web_search", WEB_SEARCH_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def web_search(conn: "ConnectionHandler", query: str = None):
    logger.bind(tag=TAG).info(f"web_search called | query={query}")
    if not query:
        return ActionResponse(Action.REQLLM, "No search query was given.", None)

    results = await asyncio.to_thread(perform_web_search, query, conn.config)
    if len(results) > MAX_RESULT_CHARS:
        results = results[:MAX_RESULT_CHARS] + "... [truncated]"
    return ActionResponse(Action.REQLLM, results, None)
