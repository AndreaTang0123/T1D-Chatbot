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

    def send_to_client(payload: dict) -> None:
        # 在工作线程里调用，交回事件循环发送；发送失败（如已断线）不影响分析本身
        try:
            asyncio.run_coroutine_threadsafe(
                conn.websocket.send(json.dumps(payload, ensure_ascii=False, default=str)), conn.loop
            )
        except Exception as e:
            logger.bind(tag=TAG).warning(f"Failed to send coach message: {e}")

    def on_step(step: str) -> None:
        send_to_client({"type": "coach", "state": "step", "step": step, "topic": topic})

    # Nightscout 同步和 decision / interpretation 两次 LLM 调用都是阻塞的，放到线程里，不卡住事件循环
    result = await asyncio.to_thread(run_insight_pipeline, conn, question, topic, on_step)
    send_to_client(build_evidence(topic, result))
    return ActionResponse(Action.REQLLM, json.dumps(result, ensure_ascii=False, default=str), None)


def build_evidence(topic: str, result: dict) -> dict:
    """浏览器「Data behind this answer」折叠区：只放本次真正交给模型的数据，不额外计算。"""
    analysis = result.get("analysis_result") or {}
    patterns = analysis.get("patterns") or {}
    return {
        "type": "coach",
        "state": "evidence",
        "topic": topic,
        "cgm": {
            "status": analysis.get("cgm_status"),
            "glucose_now": analysis.get("glucose_now"),
            "last_glucose": analysis.get("last_glucose"),
            "direction": analysis.get("glucose_direction"),
            "age_min": analysis.get("glucose_age_min"),
            "metrics": analysis.get("cgm_metrics") or {},
            "context": patterns.get("daily") or [],
        },
        "pump": {
            "data_status": analysis.get("pump_data_status") or {},
            "summary": (analysis.get("pump_status") or {}).get("summary"),
            "patterns": analysis.get("pump_patterns") or [],
            "insulin_effect": analysis.get("pump_glucose_joint") or [],
            "anomalies": analysis.get("anomalies") or [],
        },
        "decision": result.get("decision_result") or {},
    }
