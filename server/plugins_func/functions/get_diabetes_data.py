import asyncio
import json
from typing import TYPE_CHECKING

from config.logger import setup_logging
from core.utils.coach_pipeline import run_insight_pipeline
from plugins_func.register import register_function, ToolType, ActionResponse, Action

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

GET_DIABETES_DATA_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_diabetes_data",
        "description": (
            "Analyze the user's CGM (blood glucose) and insulin pump data. Call this when the user asks about "
            "glucose trends, patterns, highs, lows, time in range, how their control has been, insulin, bolus, "
            "basal, or whether insulin worked. Do NOT call it for small talk, or when the user only wants the "
            "current reading and it is already in the message context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["glucose", "insulin", "both"],
                    "description": "glucose = CGM only; insulin = pump only; both = how insulin affected glucose",
                },
                "question": {
                    "type": "string",
                    "description": "The user's question, in their own words and language",
                },
            },
            "required": ["topic", "question"],
        },
    },
}


@register_function("get_diabetes_data", GET_DIABETES_DATA_FUNCTION_DESC, ToolType.SYSTEM_CTL)
async def get_diabetes_data(conn: "ConnectionHandler", topic: str = "both", question: str = ""):
    if topic not in ("glucose", "insulin", "both"):
        topic = "both"
    logger.bind(tag=TAG).info(f"get_diabetes_data called | topic={topic} | question={question}")

    # Nightscout 同步和 decision / interpretation 两次 LLM 调用都是阻塞的，放到线程里，不卡住事件循环
    result = await asyncio.to_thread(run_insight_pipeline, conn, question, topic)
    return ActionResponse(Action.REQLLM, json.dumps(result, ensure_ascii=False, default=str), None)
