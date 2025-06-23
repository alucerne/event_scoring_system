# lead_scoring_api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import uvicorn

app = FastAPI()

# -----------------------------
# Data Models
# -----------------------------
class Event(BaseModel):
    hem_sha256: str
    event_type: str
    event_timestamp: str  # ISO timestamp
    resolution: Optional[dict] = {}

class EventPayload(BaseModel):
    events: List[Event]

# -----------------------------
# Scoring Logic
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

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/score")
async def score_events(payload: List[EventPayload]):
    all_events = []
    for block in payload:
        for e in block.events:
            all_events.append({
                "hem_sha256": e.hem_sha256,
                "event_type": e.event_type,
                "event_timestamp": e.event_timestamp,
                "personal_emails": e.resolution.get("PERSONAL_EMAILS")
            })

    df = pd.DataFrame(all_events)
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

    result = final.sort_values("final_score", ascending=False)[
        ["hem_sha256", "personal_emails", "final_score"]
    ].to_dict(orient="records")

    return {"results": result}

# Uncomment to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
