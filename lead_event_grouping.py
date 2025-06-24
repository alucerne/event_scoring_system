# lead_event_grouping.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import uvicorn
import json
from json.decoder import JSONDecoder
import numpy as np

app = FastAPI()

# -----------------------------
# POST /group-events Endpoint
# -----------------------------
@app.post("/group-events")
async def group_events(request: Request):
    try:
        raw_body = await request.body()
        raw_str = raw_body.decode("utf-8").strip()
        decoder = JSONDecoder(strict=False)
        payload, _ = decoder.raw_decode(raw_str)
    except Exception as e:
        return {"error": "JSON decode failed", "reason": str(e)}

    all_events = []
    for block in payload:
        for e in block.get("events", []):
            raw_email = e.get("resolution", {}).get("PERSONAL_EMAILS", "")
            email = raw_email.split(",")[0].strip() if raw_email else None

            all_events.append({
                "hem_sha256": e.get("hem_sha256"),
                "event_type": e.get("event_type"),
                "event_timestamp": e.get("event_timestamp"),
                "personal_emails": email
            })

    df = pd.DataFrame(all_events)
    df.dropna(subset=["hem_sha256", "event_type"], inplace=True)

    grouped = df.groupby(["hem_sha256", "personal_emails"]).agg(
        events_collected=("event_type", lambda x: ", ".join(sorted(x)))
    ).reset_index()

    results = grouped.to_dict(orient="records")
    return {"results": results}

# Uncomment to run locally
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
