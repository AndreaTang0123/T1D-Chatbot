"""浏览器顶部血糖胶囊和「Today」面板用的状态快照（{"type":"status"} 消息的回复内容）。

只读 data/<client_id>/ 下已经同步好的 cgm.csv / pump_events.csv，不调用 LLM。
CGM 读数过旧时会顺手从 Nightscout 补拉一次（每个 client 最多每分钟一次），
因为后台任务 15 分钟才拉一次，而读数超过 15 分钟就算 stale。
"""
import csv
import json
import os
import statistics
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

LOW = 70
HIGH = 180
REFRESH_IF_OLDER_THAN_MIN = 5
MIN_FETCH_INTERVAL_S = 60

_fetch_lock = threading.Lock()
_last_fetch: Dict[str, float] = {}
# path -> (mtime, rows)：pump_events.csv 有几 MB，前端每分钟请求一次，不能每次都整表重读
_csv_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def _read_csv_cached(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    mtime = os.path.getmtime(path)
    cached = _csv_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    _csv_cache[path] = (mtime, rows)
    return rows


def _load_client_config(client_id: str) -> Dict[str, Any]:
    path = os.path.join("data", client_id, "config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.bind(tag=TAG).error(f"Failed to load config for {client_id}: {e}")
        return {}


def _range_stats(values: List[float]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    stats = {
        "n": n,
        "mean": round(mean),
        "tir": round(sum(LOW <= v <= HIGH for v in values) / n * 100, 1),
        "tbr": round(sum(v < LOW for v in values) / n * 100, 1),
        "tar": round(sum(v > HIGH for v in values) / n * 100, 1),
        "min": round(min(values)),
        "max": round(max(values)),
    }
    if n > 1:
        stats["cv"] = round(statistics.pstdev(values) / mean * 100, 1)
    return stats


def _maybe_refresh_cgm(cgm_manager, client_id: str, age_min: Optional[int]) -> bool:
    if age_min is not None and age_min < REFRESH_IF_OLDER_THAN_MIN:
        return False
    with _fetch_lock:
        now = time.time()
        if now - _last_fetch.get(client_id, 0) < MIN_FETCH_INTERVAL_S:
            return False
        _last_fetch[client_id] = now
        cgm_manager.fetch_and_update(client_id)
    return True


def _cgm_section(client_id: str, cgm_cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    from core.utils.cgm_manager import CGMManager

    manager = CGMManager(config)
    stale_after = int(config.get("cgm_stale_after_minutes", 15))
    current = manager.get_current_status(client_id, stale_after)
    if _maybe_refresh_cgm(manager, client_id, current["age_min"]):
        current = manager.get_current_status(client_id, stale_after)

    latest = manager.get_latest_reading(client_id) or {}
    section = {
        "enabled": True,
        "status": current["status"],
        "glucose": current["glucose"],
        "direction": current["direction"],
        "age_min": current["age_min"],
        "unix_s": latest.get("unix_s"),
        "stale_after_min": stale_after,
    }

    rows = _read_csv_cached(os.path.join("data", client_id, "cgm.csv"))
    tz = ZoneInfo(cgm_cfg.get("user_tz", "US/Eastern"))
    now = time.time()
    midnight = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    cutoff_14d = now - 14 * 86400

    today: List[float] = []
    last14: List[float] = []
    days14 = set()
    for r in rows:
        try:
            ts = int(r["unix_s"])
            v = float(r["sgv"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts >= cutoff_14d:
            last14.append(v)
            days14.add((r.get("time") or "")[:10])
            if ts >= midnight:
                today.append(v)

    section["today"] = _range_stats(today)
    stats14 = _range_stats(last14)
    if stats14:
        # 与 cgm_manager.analyze_long_term_control 相同的 GMI 公式
        stats14["gmi"] = round(3.31 + 0.02392 * (sum(last14) / len(last14)), 1)
        stats14["days"] = len(days14)
    section["days14"] = stats14
    return section


def _pump_section(client_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    from core.utils.pump_manager import PumpManager

    manager = PumpManager()
    section = {"enabled": True}
    section.update(manager.get_data_status(client_id, float(config.get("pump_stale_after_hours", 6))))

    rows = _read_csv_cached(manager._events_csv_path(client_id))
    last_bolus = None
    for r in reversed(rows):
        if str(r.get("category") or "").strip().lower() == "bolus":
            last_bolus = r
            break
    if last_bolus:
        section["last_bolus"] = {
            "unix_s": int(float(last_bolus.get("unix_s") or 0)),
            "insulin_units": manager._safe_float(last_bolus.get("insulin_units")),
            "carbs_g": manager._safe_float(last_bolus.get("carbs_g")),
        }
    return section


def build_status_snapshot(client_id: Optional[str], config: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "type": "status",
        "server_time": int(time.time()),
        "cgm": {"enabled": False},
        "pump": {"enabled": False},
    }
    if not client_id:
        return snapshot

    client_cfg = _load_client_config(client_id)
    if client_cfg.get("cgm"):
        try:
            snapshot["cgm"] = _cgm_section(client_id, client_cfg["cgm"], config)
        except Exception as e:
            logger.bind(tag=TAG).error(f"CGM status snapshot failed for {client_id}: {e}")
            snapshot["cgm"] = {"enabled": True, "status": "no_data"}
    if client_cfg.get("pump"):
        try:
            snapshot["pump"] = _pump_section(client_id, config)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Pump status snapshot failed for {client_id}: {e}")
            snapshot["pump"] = {"enabled": True, "status": "no_data"}
    return snapshot
