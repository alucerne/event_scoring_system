# lead_scoring_system.py (field filtering + single-value fields enforced)
from fastapi import FastAPI, Request
from fastapi.params import Query
from datetime import datetime
import pandas as pd
import uvicorn
import numpy as np
from typing import Optional

app = FastAPI()

# -----------------------------
# Scoring Config
# -----------------------------
event_weights = {
    "all_form_submissions": 10,
    "file_downloads": 9,
    "copy": 8,
    "deep_scroll": 6,
    "video_play": 3,
    "video_pause": 2,
    "all_clicks": 4,
    "exit_intent": 3,
    "user_idle": 2,
    "page_view": 1
}

def recency_multiplier(event_time, ref_time):
    days_old = (ref_time - event_time).days
    if days_old <= 1:
        return 1.2
    elif days_old <= 3:
        return 1.0
    elif days_old <= 7:
        return 0.7
    elif days_old <= 14:
        return 0.5
    return 0.2

def burst_bonus(timestamps):
    for i in range(len(timestamps)):
        window = [t for t in timestamps if 0 <= (t - timestamps[i]).total_seconds() <= 600]
        if len(window) >= 3:
            return 5
    return 0

def velocity_bonus(event_count, duration_min):
    if duration_min == 0:
        return 0
    rate = event_count / duration_min
    if rate >= 2:
        return 3
    elif rate >= 1:
        return 2
    elif rate >= 0.5:
        return 1
    return 0

def extract_first_value(val):
    if isinstance(val, str) and "," in val:
        return val.split(",")[0].strip()
    return val

def extract_events_from_payload(payload):
    all_events = []

    if isinstance(payload, dict) and "events" in payload:
        payload = [payload]

    for block in payload:
        for event in block.get("events", []):
            if not event.get("event_type") or not event.get("hem_sha256"):
                continue  # Must include both event_type and hem_sha256

            resolution = event.get("resolution", {})
            raw_email = resolution.get("PERSONAL_EMAILS", "")
            email = raw_email.split(",")[0].strip() if raw_email else None

            flat_event = {
                "hem_sha256": event.get("hem_sha256"),
                "event_type": event.get("event_type"),
                "event_timestamp": event.get("event_timestamp"),
                "personal_emails": email,
            }

            for k, v in resolution.items():
                flat_event[k.lower()] = extract_first_value(v)

            all_events.append(flat_event)

    return pd.DataFrame(all_events)

@app.post("/score")
async def score_events(request: Request, fields: Optional[str] = Query(None)):
    try:
        payload = await request.json()
    except Exception as e:
        return {"error": "JSON decode failed", "reason": str(e)}

    df = extract_events_from_payload(payload)
    df.dropna(subset=["hem_sha256", "event_type", "event_timestamp"], inplace=True)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    df["event_score"] = df["event_type"].map(event_weights).fillna(0)

    now = df["event_timestamp"].max()
    df["recency_multiplier"] = df["event_timestamp"].apply(lambda x: recency_multiplier(x, now))
    df["adjusted_score"] = df["event_score"] * df["recency_multiplier"]

    burst_map = {}
    for hem, group in df.groupby("hem_sha256"):
        ts = group["event_timestamp"].sort_values().tolist()
        burst_map[hem] = burst_bonus(ts)
    df["burst_bonus"] = df["hem_sha256"].map(burst_map)

    agg = df.groupby("hem_sha256").agg(
        start=("event_timestamp", "min"),
        end=("event_timestamp", "max"),
        count=("event_score", "count")
    ).reset_index()
    agg["duration_min"] = (agg["end"] - agg["start"]).dt.total_seconds() / 60
    agg["velocity_bonus"] = agg.apply(lambda x: velocity_bonus(x["count"], x["duration_min"]), axis=1)

    final = df.groupby("hem_sha256").agg(adjusted_total=("adjusted_score", "sum")).reset_index()
    final = final.merge(agg[["hem_sha256", "velocity_bonus"]], on="hem_sha256")
    final["burst_bonus"] = final["hem_sha256"].map(burst_map)
    final["final_score"] = final["adjusted_total"] + final["burst_bonus"] + final["velocity_bonus"]

    emails = df[["hem_sha256", "personal_emails"]].dropna().drop_duplicates()
    final = final.merge(emails, on="hem_sha256", how="left")

    final["final_score"] = final["final_score"].replace([np.inf, -np.inf, np.nan], 0)
    final = final.fillna("unknown")

    safe_result = []
    field_list = fields.split(",") if fields else None

    for _, row in final.iterrows():
        result = {col: row[col] for col in row.index if not field_list or col in field_list}
        result["final_score"] = float(row["final_score"])
        safe_result.append(result)

    return {"results": safe_result}

@app.post("/group-events")
async def group_events(request: Request, fields: Optional[str] = Query(None)):
    try:
        payload = await request.json()
    except Exception as e:
        return {"error": "JSON decode failed", "reason": str(e)}

    df = extract_events_from_payload(payload)
    df.dropna(subset=["hem_sha256", "event_type"], inplace=True)

    group_keys = ["hem_sha256", "personal_emails"]
    agg_fields = {k: (k, "first") for k in df.columns if k not in group_keys + ["event_type", "event_timestamp"]}
    agg_fields["events_collected"] = ("event_type", lambda x: ", ".join(sorted(set(x))))

    grouped = df.groupby(group_keys).agg(**agg_fields).reset_index()

    if fields:
        field_list = fields.split(",")
        grouped = grouped[[col for col in grouped.columns if col in field_list]]

    results = grouped.to_dict(orient="records")
    return {"results": results}

# Uncomment to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
