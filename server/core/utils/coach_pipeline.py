"""
Diabetes coach pipeline: analysis -> decision -> interpretation.

Called by the get_diabetes_data tool. The main LLM (persona prompt) turns the
returned insight into the final, streamed reply, so the persona layer is no
longer a separate call here.
"""
import json
import os
import uuid
from typing import Any, Dict, TYPE_CHECKING

from config.logger import setup_logging
from core.utils.dialogue import Dialogue, Message
from core.utils.util import extract_json_from_string

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()


def llm_single_shot(conn: "ConnectionHandler", system_prompt: str, user_payload: Any) -> str:
    """Run a one-shot LLM call without touching the main dialogue history."""
    if not conn.llm:
        return ""

    try:
        if isinstance(user_payload, str):
            user_content = user_payload
        else:
            user_content = json.dumps(user_payload, ensure_ascii=False)

        temp_dialogue = Dialogue()
        temp_dialogue.update_system_message(system_prompt)
        temp_dialogue.put(Message(role="user", content=user_content))
        messages = temp_dialogue.get_llm_dialogue_with_memory(None, conn.config.get("voiceprint", {}))

        chunks = []
        for chunk in conn.llm.response(str(uuid.uuid4()), messages):
            if chunk:
                chunks.append(chunk)
        return "".join(chunks).strip()
    except Exception as e:
        logger.bind(tag=TAG).error(f"Single-shot LLM call failed: {e}")
        return ""


def safe_parse_json(raw_text: str) -> Dict[str, Any]:
    """Parse JSON response safely, with fallback extraction from free text."""
    if not raw_text:
        return {}

    try:
        return json.loads(raw_text)
    except Exception:
        pass

    extracted = extract_json_from_string(raw_text)
    if extracted:
        try:
            return json.loads(extracted)
        except Exception:
            pass

    logger.bind(tag=TAG).warning(f"Failed to parse JSON response: {raw_text[:200]}")
    return {}


def build_analysis_result(conn: "ConnectionHandler", query: str, topic: str) -> Dict[str, Any]:
    """Build structured CGM / pump analysis for the decision and interpretation layers."""
    result = {
        "user_query": query,
        "language_hint": getattr(conn, "current_language_tag", None),
        "topic": topic,  # glucose / insulin / both，由主模型调用工具时给出
        "cgm_status": "no_data",  # ok / stale / no_data
        "glucose_now": None,  # 只有 cgm_status == "ok" 时才有值
        "glucose_direction": None,
        "glucose_age_min": None,
        "last_glucose": None,  # stale 时的最后一次读数，不能当作当前值
        "cgm_note": None,
        "cgm_metrics": {},
        "events": {},
        "patterns": {
            "daily": [],
            "weekly": [],
            "monthly": [],
        },
        "pump_data_status": {"status": "no_data"},  # ok / stale / no_data，见 PumpManager.get_data_status
        "pump_status": {},
        "pump_patterns": [],
        "pump_glucose_joint": [],
        "anomalies": [],
    }

    client_id = conn.headers.get("client-id") if conn.headers else None
    if not client_id:
        return result

    try:
        config_path = os.path.join("data", client_id, "config.json")
        client_cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                client_cfg = json.load(f)

        cgm_enabled = bool(client_cfg.get("cgm"))
        pump_enabled = bool(client_cfg.get("pump"))

        from core.utils.cgm_manager import CGMManager, format_cgm_status, NO_DATA_NOTE
        result["cgm_note"] = NO_DATA_NOTE

        if cgm_enabled:
            cgm_manager = CGMManager()
            cgm_manager.fetch_and_update(client_id)

            cgm_state = cgm_manager.get_current_status(
                client_id, int(conn.config.get("cgm_stale_after_minutes", 15))
            )
            result["cgm_status"] = cgm_state["status"]
            result["glucose_direction"] = cgm_state["direction"]
            result["glucose_age_min"] = cgm_state["age_min"]
            result["cgm_note"] = format_cgm_status(cgm_state)
            if cgm_state["status"] == "ok":
                result["glucose_now"] = cgm_state["glucose"]
            elif cgm_state["status"] == "stale":
                result["last_glucose"] = cgm_state["glucose"]

            # 这里尽量兼容你现有 manager，如果某个方法不存在就跳过
            if hasattr(cgm_manager, "get_metrics_summary"):
                metrics = cgm_manager.get_metrics_summary(client_id)
                if isinstance(metrics, dict):
                    result["cgm_metrics"] = metrics.get("metrics", metrics)
                    if "events" in metrics and isinstance(metrics["events"], dict):
                        result["events"] = metrics["events"]

            context_summary = cgm_manager.get_context_summary(client_id)
            if context_summary:
                result["patterns"]["daily"].append(context_summary)

        if pump_enabled:
            from core.utils.pump_manager import PumpManager
            pump_manager = PumpManager()
            if topic in ("insulin", "both"):
                pump_manager.fetch_and_update(client_id)

            pump_context = pump_manager.get_authoritative_summary(client_id)
            if pump_context:
                result["pump_status"]["summary"] = pump_context

            result["pump_data_status"] = pump_manager.get_data_status(
                client_id, float(conn.config.get("pump_stale_after_hours", 6))
            )
            result["pump_patterns"] = pump_manager.analyze_recent_patterns(client_id)
            result["pump_glucose_joint"] = pump_manager.analyze_insulin_effectiveness(client_id)
            result["anomalies"] = pump_manager.detect_pump_anomalies(client_id)

    except Exception as e:
        logger.bind(tag=TAG).error(f"Failed to build pipeline analysis result: {e}")

    return result


def run_insight_pipeline(conn: "ConnectionHandler", query: str, topic: str) -> Dict[str, Any]:
    """analysis -> decision -> interpretation. Returns what the persona layer needs."""
    analysis_result = build_analysis_result(conn, query, topic)

    prompts = conn.pipeline_prompts or {}
    decision_prompt = prompts.get("decision")
    interpretation_prompt = prompts.get("interpretation")
    if not decision_prompt or not interpretation_prompt:
        # 这个用户没有分层 prompt：直接把结构化数据交给主模型
        return {"analysis_result": analysis_result}

    decision_result = safe_parse_json(llm_single_shot(conn, decision_prompt, analysis_result))
    insight_text = llm_single_shot(
        conn,
        interpretation_prompt,
        {"analysis_result": analysis_result, "decision_result": decision_result},
    )
    logger.bind(tag=TAG).info(f"Coach insight: {insight_text[:200]}")

    return {
        "insight_text": insight_text,
        "decision_result": decision_result,
        "analysis_result": analysis_result,
    }
