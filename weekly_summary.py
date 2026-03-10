from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fetch_cgm import API_SECRET_SHA1, BASE_URL, EASTERN_TZ, fetch_cgm


# -----------------------------
# Data loading / preprocessing
# -----------------------------
def fetch_week_entries_local(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetch Nightscout CGM data for a local US/Eastern date range.

    Example:
        start_date = "2026-03-01"
        end_date   = "2026-03-07"

    The input dates are interpreted as local US/Eastern calendar dates.
    We convert the full local date range into UTC before querying Nightscout.
    """
    start_local_date = pd.to_datetime(start_date).date()
    end_local_date = pd.to_datetime(end_date).date()

    start_local = EASTERN_TZ.localize(datetime.combine(start_local_date, time.min))
    end_local = EASTERN_TZ.localize(datetime.combine(end_local_date, time.max.replace(microsecond=0)))

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


def filter_local_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filter the DataFrame to an inclusive local US/Eastern date range."""
    start_local_date = pd.to_datetime(start_date).date()
    end_local_date = pd.to_datetime(end_date).date()

    range_df = df[
        (df["timestamp"].dt.date >= start_local_date)
        & (df["timestamp"].dt.date <= end_local_date)
    ].copy()

    if range_df.empty:
        raise ValueError(f"No CGM data found for local date range {start_date} to {end_date}.")

    return range_df


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
    """Compute glucose statistics."""
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
# Daily breakdown inside week
# -----------------------------
def build_daily_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build daily metrics for each day in the week."""
    results: List[Dict[str, Any]] = []

    for date_value, day_df in df.groupby(df["timestamp"].dt.date):
        day_df = day_df.copy().sort_values("timestamp").reset_index(drop=True)

        stats = compute_basic_stats(day_df)
        ranges = compute_ranges(day_df)

        hypo_events = detect_events(day_df, threshold=70, mode="low", min_readings=3)
        severe_hypo_events = detect_events(day_df, threshold=54, mode="low", min_readings=3)
        hyper_events = detect_events(day_df, threshold=180, mode="high", min_readings=6)
        severe_hyper_events = detect_events(day_df, threshold=250, mode="high", min_readings=6)

        results.append(
            {
                "date": str(date_value),
                **stats,
                **ranges,
                "hypo_event_count_lt70": len(hypo_events),
                "severe_hypo_event_count_lt54": len(severe_hypo_events),
                "hyper_event_count_gt180": len(hyper_events),
                "severe_hyper_event_count_gt250": len(severe_hyper_events),
            }
        )

    return results


# -----------------------------
# Summary assembly / printing
# -----------------------------
def build_weekly_summary(df: pd.DataFrame, start_date: str, end_date: str) -> Dict[str, Any]:
    """Assemble weekly metrics and events into one summary dict."""
    week_df = filter_local_range(df, start_date, end_date)

    stats = compute_basic_stats(week_df)
    ranges = compute_ranges(week_df)

    hypo_events = detect_events(week_df, threshold=70, mode="low", min_readings=3)
    severe_hypo_events = detect_events(week_df, threshold=54, mode="low", min_readings=3)
    hyper_events = detect_events(week_df, threshold=180, mode="high", min_readings=6)
    severe_hyper_events = detect_events(week_df, threshold=250, mode="high", min_readings=6)

    daily_breakdown = build_daily_breakdown(week_df)

    summary: Dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        **stats,
        **ranges,
        "hypo_event_count_lt70": len(hypo_events),
        "severe_hypo_event_count_lt54": len(severe_hypo_events),
        "hyper_event_count_gt180": len(hyper_events),
        "severe_hyper_event_count_gt250": len(severe_hyper_events),
        "hypo_events": hypo_events,
        "hyper_events": hyper_events,
        "daily_breakdown": daily_breakdown,
    }
    return summary


def print_weekly_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print the weekly summary."""
    print(f"\n=== Weekly CGM Summary ({summary['start_date']} to {summary['end_date']}) ===")
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

    print("\nDaily breakdown:")
    for day in summary["daily_breakdown"]:
        print(
            f"  {day['date']}: "
            f"mean={day['mean_glucose']:.1f}, "
            f"TIR={day['TIR_70_180']:.1f}%, "
            f"TBR<70={day['TBR_lt70']:.1f}%, "
            f"TAR>180={day['TAR_gt180']:.1f}%, "
            f"hypo={day['hypo_event_count_lt70']}, "
            f"hyper={day['hyper_event_count_gt180']}"
        )

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
    end_date = "2026-03-07"
    start_date = str((pd.to_datetime(end_date) - timedelta(days=6)).date())

    entries = fetch_week_entries_local(start_date, end_date)
    df = entries_to_dataframe(entries)
    summary = build_weekly_summary(df, start_date, end_date)
    print_weekly_summary(summary)
    print(df.head())
    print(df.tail())
    print(len(df))


if __name__ == "__main__":
    main()