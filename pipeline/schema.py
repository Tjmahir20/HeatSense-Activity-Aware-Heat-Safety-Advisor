"""
Pydantic models for validating LLM-produced shift plans.

These schemas enforce the JSON contract between ``pipeline.agent`` and the
Flask templates so hour cards always receive typed, safe fields.
"""

from pydantic import BaseModel


class HourPlan(BaseModel):
    """One hour of guidance: WBGT, risk band, work/rest, water, and notes."""

    hour: int
    wbgt: float
    risk_level: str   # "safe" | "caution" | "danger" | "stop_work"
    risk_color: str   # "green" | "yellow" | "red" | "black"
    work_rest_ratio: str   # e.g. "75% work / 25% rest"
    water_intake: str      # e.g. "250 ml every 15 min"
    notes: str             # plain-language crew-ready advice


class ShiftPlan(BaseModel):
    """Full shift output: summary line, per-hour plans, and disclaimer text."""

    worker_summary: str    # one-sentence shift-risk summary
    hours: list[HourPlan]
    disclaimer: str        # mandatory safety disclaimer
