from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Tuple, Union, Optional
import requests
import pytz

BASE_URL = "https://night--nightscout--bfbsr6d8dlk9.code.run"
API_SECRET_SHA1 = "893c9b0d94d5b20431cb5b786216cd096d3813f5"

# Timezone converter: UTC epoch ms -> US/Eastern datetime (EST/EDT automatically)
EASTERN_TZ = pytz.timezone("US/Eastern")

def utc_ms_to_est(ms_utc: int) -> datetime:
    """Convert UTC epoch milliseconds (Nightscout `date`) to US/Eastern datetime."""
    dt_utc = datetime.fromtimestamp(ms_utc / 1000, tz=timezone.utc)
    return dt_utc.astimezone(EASTERN_TZ)

def fetch_cgm(
    start: Union[str, datetime], end: Union[str, datetime], *, base_url: str,
    api_secret_sha1: str,
    return_format: Literal["dict", "tuples"] = "tuples",
    page_size: int = 100,
    timeout: int = 20,
) -> Union[List[Dict[str, Any]], List[Tuple[datetime, int, Optional[str]]]]:
    """
    Fetch CGM entries from Nightscout within [start, end] time range (inclusive).

    Args:
      start, end: ISO str or datetime. Interpreted/converted to UTC.
      return_format:
        - "dict": returns list of raw dict entries
        - "tuples": returns list of (datetime_utc, sgv, direction) (UTC datetime for storage/processing)
      page_size: Nightscout count per request (use <=1000 for safety)
      timeout: requests timeout (seconds)

    Returns:
      List of entries in requested format, sorted by time ascending.

    Raises:
      RuntimeError for HTTP errors or unexpected responses.
    """
    # Convert start/end to UTC epoch milliseconds (Nightscout uses UTC internally)
    if isinstance(start, datetime):
        start_dt = start if start.tzinfo is not None else start.replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.astimezone(timezone.utc).timestamp() * 1000)
    elif isinstance(start, str):
        s = start.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        start_dt = datetime.fromisoformat(s)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.astimezone(timezone.utc).timestamp() * 1000)
    else:
        raise TypeError("start must be a str or datetime")

    if isinstance(end, datetime):
        end_dt = end if end.tzinfo is not None else end.replace(tzinfo=timezone.utc)
        end_ms = int(end_dt.astimezone(timezone.utc).timestamp() * 1000)
    elif isinstance(end, str):
        s = end.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        end_dt = datetime.fromisoformat(s)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        end_ms = int(end_dt.astimezone(timezone.utc).timestamp() * 1000)
    else:
        raise TypeError("end must be a str or datetime")
    if end_ms < start_ms:
        raise ValueError("end must be >= start")

    headers = {"api-secret": api_secret_sha1}
    url = f"{base_url.rstrip('/')}/api/v1/entries.json"

    all_entries: List[Dict[str, Any]] = []
    # We page forward by repeatedly requesting entries with date > last_ms
    last_ms = start_ms - 1

    while True:
        params = {
            "count": page_size,
            "find[date][$gt]": last_ms,
            "find[date][$lte]": end_ms,
        }

        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"Nightscout error {r.status_code}: {r.text[:300]}")

        batch = r.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected response type: {type(batch)}")

        if not batch:
            break

        # Nightscout may return newest-first; we sort by date asc to page safely
        batch.sort(key=lambda x: x.get("date", 0))

        # Append, but ensure we don't include anything outside range
        for e in batch:
            d = e.get("date")
            if isinstance(d, int) and start_ms <= d <= end_ms:
                all_entries.append(e)

        last_ms = batch[-1].get("date", last_ms)

        # If the newest item in this batch is already >= end, we're done
        if isinstance(last_ms, int) and last_ms >= end_ms:
            break

        # Safety: if server returns same last_ms repeatedly, avoid infinite loop
        if len(batch) == 1 and batch[0].get("date") == last_ms:
            break

    # Final sort and de-dup by _id (or date+sgv fallback)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for e in sorted(all_entries, key=lambda x: x.get("date", 0)):
        key = e.get("_id") or (e.get("date"), e.get("sgv"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    if return_format == "dict":
        return deduped

    # tuples
    out: List[Tuple[datetime, int, Optional[str]]] = []
    for e in deduped:
        d = e.get("date")
        sgv = e.get("sgv")
        direction = e.get("direction")
        if isinstance(d, int) and isinstance(sgv, int):
            dt_utc = datetime.fromtimestamp(d / 1000, tz=timezone.utc)
            out.append((dt_utc, sgv, direction if isinstance(direction, str) else None))
    return out


data = fetch_cgm(
    "2026-01-27T00:00:00Z",
    "2026-01-27T23:59:59Z",
    base_url=BASE_URL,
    api_secret_sha1=API_SECRET_SHA1,
    return_format="tuples",
)
data_dict = fetch_cgm(
    "2026-03-07T00:00:00Z",
    "2026-03-07T23:59:59Z",
    base_url=BASE_URL,
    api_secret_sha1=API_SECRET_SHA1,
    return_format="dict",
)

print("done")
print("data rows:", len(data_dict))
print("first row:", data_dict[0] if data_dict else None)
dt_et = data[0][0].astimezone(EASTERN_TZ)
print(dt_et)