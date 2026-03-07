from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fetch_cgm import API_SECRET_SHA1, BASE_URL, EASTERN_TZ, fetch_cgm


# -----------------------------
# Data loading / preprocessing
# -----------------------------
def fetch_day_entries_local(target_date: str) -> List[Dict[str, Any]]:
    """
    Fetch one local US/Eastern calendar day of Nightscout CGM data.

    Example:
        target_date = "2026-03-06"

    The input date is interpreted as a local US/Eastern day.
    We convert the local day's start/end into UTC before querying Nightscout.
    """
    local_date = pd.to_datetime(target_date).date()

    start_local = EASTERN_TZ.localize(datetime.combine(local_date, time.min))
    end_local = EASTERN_TZ.localize(datetime.combine(local_date, time.max.replace(microsecond=0)))

    return fetch_cgm(
        start_local,
        end_local,
        base_url=BASE_URL,
        api_secret_sha1=API_SECRET_SHA1,
        return_format="dict",
    )


def entries_to_dataframe(entries: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert Nightscout raw dict entries into a clean DataFrame."""
    if not entries:
        return pd.DataFrame(columns=["timestamp", "glucose", "direction"])

    df = pd.DataFrame(entries).copy()

    if "dateString" in df.columns:
        df["timestamp"] = pd.to_datetime(df["dateString"], utc=True)
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    else:
        raise ValueError("Entries must contain either 'dateString' or 'date'.")

    df["timestamp"] = df["timestamp"].dt.tz_convert(EASTERN_TZ)
    df["glucose"] = pd.to_numeric(df["sgv"], errors="coerce")
    df["direction"] = df["direction"] if "direction" in df.columns else None

    df = df.dropna(subset=["timestamp", "glucose"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "glucose", "direction"]]


def filter_local_day(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """Filter the DataFrame to one local US/Eastern calendar day."""
    local_date = pd.to_datetime(target_date).date()
    day_df = df[df["timestamp"].dt.date == local_date].copy()

    if day_df.empty:
        raise ValueError(f"No CGM data found for local date {target_date}.")

    return day_df


# -----------------------------
# Metrics
# -----------------------------
def compute_ranges(df: pd.DataFrame) -> Dict[str, float]:
    """Compute TIR / TBR / TAR percentages."""
    g = df["glucose"]

    return {
        "TIR_70_180": ((g >= 70) & (g <= 180)).mean() * 100,
        "TBR_lt70": (g < 70).mean() * 100,
        "TBR_lt54": (g < 54).mean() * 100,
        "TAR_gt180": (g > 180).mean() * 100,
        "TAR_gt250": (g > 250).mean() * 100,
    }


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute basic daily glucose statistics."""
    g = df["glucose"]

    mean_glucose = float(g.mean())
    sd = float(g.std(ddof=0))
    cv = float(sd / mean_glucose * 100) if mean_glucose != 0 else np.nan
    gmi = float(3.31 + 0.02392 * mean_glucose)
    return {
        "num_readings": int(len(df)),
        "mean_glucose": mean_glucose,
        "median_glucose": float(g.median()),
        "min_glucose": float(g.min()),
        "max_glucose": float(g.max()),
        "sd": sd,
        "cv": cv,
        "gmi": gmi,
    }


# -----------------------------
# Event detection
# -----------------------------
def detect_events(
    df: pd.DataFrame,
    threshold: float,
    mode: str = "low",
    min_readings: int = 3,
) -> List[Dict[str, Any]]:
    """
    Detect consecutive low/high glucose events.

    min_readings=3 is about 15 minutes if data is sampled every 5 minutes.
    """
    work = df.copy()

    if mode == "low":
        work["flag"] = work["glucose"] < threshold
    elif mode == "high":
        work["flag"] = work["glucose"] > threshold
    else:
        raise ValueError("mode must be 'low' or 'high'.")

    work["group"] = (work["flag"] != work["flag"].shift()).cumsum()

    events: List[Dict[str, Any]] = []
    for _, group_df in work.groupby("group"):
        if not bool(group_df["flag"].iloc[0]):
            continue

        if len(group_df) < min_readings:
            continue

        start = group_df["timestamp"].iloc[0]
        end = group_df["timestamp"].iloc[-1]

        duration_minutes = 0.0
        if len(group_df) > 1:
            diffs = group_df["timestamp"].diff().dropna().dt.total_seconds() / 60
            if not diffs.empty:
                duration_minutes = float(diffs.sum() + diffs.median())

        events.append(
            {
                "start": start,
                "end": end,
                "duration_minutes": round(duration_minutes, 1),
                "min_glucose": float(group_df["glucose"].min()),
                "max_glucose": float(group_df["glucose"].max()),
                "num_readings": int(len(group_df)),
            }
        )

    return events


# -----------------------------
# Summary assembly / printing
# -----------------------------
def build_daily_summary(df: pd.DataFrame, target_date: str) -> Dict[str, Any]:
    """Assemble all daily metrics and events into one summary dict."""
    day_df = filter_local_day(df, target_date)

    stats = compute_basic_stats(day_df)
    ranges = compute_ranges(day_df)

    hypo_events = detect_events(day_df, threshold=70, mode="low", min_readings=3)
    severe_hypo_events = detect_events(day_df, threshold=54, mode="low", min_readings=3)
    hyper_events = detect_events(day_df, threshold=180, mode="high", min_readings=6)
    severe_hyper_events = detect_events(day_df, threshold=250, mode="high", min_readings=6)

    summary: Dict[str, Any] = {
        "date": target_date,
        **stats,
        **ranges,
        "hypo_event_count_lt70": len(hypo_events),
        "severe_hypo_event_count_lt54": len(severe_hypo_events),
        "hyper_event_count_gt180": len(hyper_events),
        "severe_hyper_event_count_gt250": len(severe_hyper_events),
        "hypo_events": hypo_events,
        "hyper_events": hyper_events,
    }
    return summary


def print_daily_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print the daily summary."""
    print(f"\n=== Daily CGM Summary for {summary['date']} ===")
    print(f"Readings: {summary['num_readings']}")
    print(f"Mean glucose: {summary['mean_glucose']:.1f} mg/dL")
    print(f"Median glucose: {summary['median_glucose']:.1f} mg/dL")
    print(
        f"Min / Max glucose: {summary['min_glucose']:.1f} / {summary['max_glucose']:.1f} mg/dL"
    )
    print(f"SD: {summary['sd']:.1f}")
    print(f"CV: {summary['cv']:.1f}%")
    print(f"GMI: {summary['gmi']:.1f}")
    
    print("\nTime in range:")
    print(f"  TIR 70-180: {summary['TIR_70_180']:.1f}%")
    print(f"  TBR <70:    {summary['TBR_lt70']:.1f}%")
    print(f"  TBR <54:    {summary['TBR_lt54']:.1f}%")
    print(f"  TAR >180:   {summary['TAR_gt180']:.1f}%")
    print(f"  TAR >250:   {summary['TAR_gt250']:.1f}%")

    print("\nEvents:")
    print(f"  Hypo events <70:   {summary['hypo_event_count_lt70']}")
    print(f"  Severe hypo <54:   {summary['severe_hypo_event_count_lt54']}")
    print(f"  Hyper events >180: {summary['hyper_event_count_gt180']}")
    print(f"  Severe hyper >250: {summary['severe_hyper_event_count_gt250']}")

    if summary["hypo_events"]:
        print("\nHypo event details:")
        for i, event in enumerate(summary["hypo_events"], start=1):
            print(
                f"  {i}. {event['start']} -> {event['end']} "
                f"({event['duration_minutes']} min, min glucose={event['min_glucose']:.1f})"
            )

    if summary["hyper_events"]:
        print("\nHyper event details:")
        for i, event in enumerate(summary["hyper_events"], start=1):
            print(
                f"  {i}. {event['start']} -> {event['end']} "
                f"({event['duration_minutes']} min, max glucose={event['max_glucose']:.1f})"
            )


# -----------------------------
# Script entry point
# -----------------------------
def main() -> None:
    target_date = "2026-03-07"

    entries = fetch_day_entries_local(target_date)
    df = entries_to_dataframe(entries)
    summary = build_daily_summary(df, target_date)
    print_daily_summary(summary)
    print(df.head())
    print(df.tail())
    print(len(df))


if __name__ == "__main__":
    main()