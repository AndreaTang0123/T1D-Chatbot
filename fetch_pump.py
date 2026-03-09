from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Union, Optional
from collections import Counter
import requests
import pytz
import json

BASE_URL = "https://night--nightscout--bfbsr6d8dlk9.code.run"
API_SECRET_SHA1 = "893c9b0d94d5b20431cb5b786216cd096d3813f5"

EASTERN_TZ = pytz.timezone("US/Eastern")


def to_utc_ms(value: Union[str, datetime]) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)

    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)

    raise TypeError("value must be str or datetime")


def parse_iso_to_dt(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fetch_treatments(
    start: Union[str, datetime],
    end: Union[str, datetime],
    *,
    base_url: str,
    api_secret_sha1: str,
    page_size: int = 500,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    start_ms = to_utc_ms(start)
    end_ms = to_utc_ms(end)

    if end_ms < start_ms:
        raise ValueError("end must be >= start")

    headers = {"api-secret": api_secret_sha1}
    url = f"{base_url.rstrip('/')}/api/v1/treatments.json"

    all_rows: List[Dict[str, Any]] = []
    last_ms = start_ms - 1

    while True:
        params = {
            "count": page_size,
            "find[created_at][$gt]": datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc).isoformat(),
            "find[created_at][$lte]": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
        }

        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"Nightscout error {r.status_code}: {r.text[:300]}")

        batch = r.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected response type: {type(batch)}")

        if not batch:
            break

        batch.sort(key=lambda x: parse_iso_to_dt(x["created_at"]).timestamp() if "created_at" in x else 0)

        for row in batch:
            created_at = row.get("created_at")
            if not isinstance(created_at, str):
                continue
            ms = int(parse_iso_to_dt(created_at).astimezone(timezone.utc).timestamp() * 1000)
            if start_ms <= ms <= end_ms:
                all_rows.append(row)

        new_last_ms = int(parse_iso_to_dt(batch[-1]["created_at"]).astimezone(timezone.utc).timestamp() * 1000)

        if new_last_ms <= last_ms:
            break

        last_ms = new_last_ms

        if last_ms >= end_ms:
            break

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in sorted(all_rows, key=lambda x: x.get("created_at", "")):
        key = row.get("_id") or (
            row.get("created_at"),
            row.get("eventType"),
            row.get("pump_event_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def fetch_profile(*, base_url: str, api_secret_sha1: str, timeout: int = 20) -> Any:
    headers = {"api-secret": api_secret_sha1}
    url = f"{base_url.rstrip('/')}/api/v1/profile.json"

    r = requests.get(url, headers=headers, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"Nightscout profile error {r.status_code}: {r.text[:300]}")
    return r.json()


def normalize_basal(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "temp_basal",
        "time": row.get("created_at"),
        "time_et": parse_iso_to_dt(row["created_at"]).astimezone(EASTERN_TZ).isoformat() if row.get("created_at") else None,
        "rate_u_per_hr": row.get("absolute") if row.get("absolute") is not None else row.get("rate"),
        "duration_min": row.get("duration"),
        "reason": row.get("reason"),
        "pump_event_id": row.get("pump_event_id"),
        "entered_by": row.get("enteredBy"),
    }


def normalize_bolus(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "bolus",
        "subtype": row.get("eventType"),
        "time": row.get("created_at"),
        "time_et": parse_iso_to_dt(row["created_at"]).astimezone(EASTERN_TZ).isoformat() if row.get("created_at") else None,
        "insulin_units": row.get("insulin"),
        "carbs_g": row.get("carbs"),
        "glucose_mgdl": row.get("glucose"),
        "notes": row.get("notes"),
        "pump_event_id": row.get("pump_event_id"),
        "entered_by": row.get("enteredBy"),
    }


def normalize_pump_event(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "pump_event",
        "subtype": row.get("eventType"),
        "time": row.get("created_at"),
        "time_et": parse_iso_to_dt(row["created_at"]).astimezone(EASTERN_TZ).isoformat() if row.get("created_at") else None,
        "entered_by": row.get("enteredBy"),
        "pump_event_id": row.get("pump_event_id"),
        "raw": row,
    }


def extract_active_profile(profile_payload: Any) -> Dict[str, Any]:
    """
    Returns a simplified active profile object from Nightscout profile.json.
    Prefers the first profile object's defaultProfile.
    """
    if not isinstance(profile_payload, list) or not profile_payload:
        return {}

    first = profile_payload[0]
    if not isinstance(first, dict):
        return {}

    default_profile_name = first.get("defaultProfile")
    store = first.get("store", {})

    if not isinstance(store, dict):
        return {}

    # prefer defaultProfile if present
    if isinstance(default_profile_name, str) and default_profile_name in store:
        active_name = default_profile_name
    else:
        active_name = next(iter(store.keys()), None)

    if active_name is None:
        return {}

    profile = store.get(active_name, {})
    if not isinstance(profile, dict):
        return {}

    return {
        "active_profile_name": active_name,
        "timezone": profile.get("timezone"),
        "units": profile.get("units"),
        "dia_hours": profile.get("dia"),
        "basal_schedule": profile.get("basal", []),
        "carbratio_schedule": profile.get("carbratio", []),
        "sensitivity_schedule": profile.get("sens", []),
        "target_low_schedule": profile.get("target_low", []),
        "target_high_schedule": profile.get("target_high", []),
        "raw_profile": profile,
    }


def fetch_pump_data(
    start: Union[str, datetime],
    end: Union[str, datetime],
    *,
    base_url: str,
    api_secret_sha1: str,
) -> Dict[str, Any]:
    treatments = fetch_treatments(
        start,
        end,
        base_url=base_url,
        api_secret_sha1=api_secret_sha1,
    )
    profile_payload = fetch_profile(
        base_url=base_url,
        api_secret_sha1=api_secret_sha1,
    )

    pump_rows = [x for x in treatments if x.get("enteredBy") == "Pump (tconnectsync)"]

    basal_rows = [x for x in pump_rows if x.get("eventType") == "Temp Basal"]
    bolus_rows = [x for x in pump_rows if "Bolus" in str(x.get("eventType"))]
    other_rows = [
        x for x in pump_rows
        if x.get("eventType") != "Temp Basal" and "Bolus" not in str(x.get("eventType"))
    ]

    normalized = {
        "summary": {
            "start": start if isinstance(start, str) else start.isoformat(),
            "end": end if isinstance(end, str) else end.isoformat(),
            "total_treatments": len(treatments),
            "pump_treatments": len(pump_rows),
            "event_type_counts": dict(Counter(x.get("eventType", "UNKNOWN") for x in pump_rows)),
            "basal_count": len(basal_rows),
            "bolus_count": len(bolus_rows),
            "other_event_count": len(other_rows),
        },
        "basal_events": [normalize_basal(x) for x in basal_rows],
        "bolus_events": [normalize_bolus(x) for x in bolus_rows],
        "other_pump_events": [normalize_pump_event(x) for x in other_rows],
        "active_profile": extract_active_profile(profile_payload),
        "raw_profile_payload": profile_payload,
    }

    return normalized


if __name__ == "__main__":
    pump_data = fetch_pump_data(
        "2026-03-09T00:00:00Z",
        "2026-03-09T23:59:59Z",
        base_url=BASE_URL,
        api_secret_sha1=API_SECRET_SHA1,
    )

    print("=== summary ===")
    print(json.dumps(pump_data["summary"], indent=2))

    print("\n=== first 3 basal events ===")
    print(json.dumps(pump_data["basal_events"][:3], indent=2))

    print("\n=== bolus events ===")
    print(json.dumps(pump_data["bolus_events"], indent=2))

    print("\n=== active profile ===")
    print(json.dumps(pump_data["active_profile"], indent=2)[:4000])