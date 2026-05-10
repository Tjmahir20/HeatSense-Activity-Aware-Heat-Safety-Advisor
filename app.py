"""
HeatSense v2 — Flask application entry point.

Exposes GET/POST routes for the interactive questionnaire, resolves locations
to coordinates, and delegates shift-plan generation to ``pipeline.agent``.
Runs on port 5001 by default so it can coexist with ``heatsense_redesign``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request

from pipeline.agent import run_agent

# Loading environment variables (e.g. ``OPENAI_API_KEY``) before route handlers run.
load_dotenv()

app = Flask(__name__)


@dataclass
class FormData:
    """Normalized POST body fields for repopulating the questionnaire on error."""

    location: str
    job_type: str
    direct_sun: bool
    shift_start: int
    shift_end: int


# Compiling once: matching optional-sign decimal lat,lon pairs typed by power users.
_LATLON_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$")


def _geocode_location(location: str, timeout_s: int = 15) -> tuple[float, float, str]:
    """
    Resolve a free-text location or ``lat,lon`` string to coordinates and label.

    If the string matches the lat/lon regex, returns parsed floats. Otherwise
    queries Open-Meteo geocoding; on miss, retries with the substring before the
    first comma (helps ``City, ST`` queries).

    Args:
        location: User-supplied place string or ``lat,lon``.
        timeout_s: HTTP timeout for geocoding requests.

    Returns:
        Tuple ``(latitude, longitude, display_label)``.

    Raises:
        ValueError: When geocoding finds no result and input is not lat/lon.
        requests.HTTPError: On HTTP failure from the geocoding API.
    """
    # Attempting direct parsing when the user pasted decimal coordinates.
    m = _LATLON_RE.match(location)
    if m:
        # Parsing latitude and longitude from the two capture groups.
        lat = float(m.group(1))
        lon = float(m.group(2))
        # Returning a synthetic label so logs still show something human-readable.
        return lat, lon, f"{lat:.4f}, {lon:.4f}"

    def _lookup(name: str) -> dict[str, Any] | None:
        """
        Call Open-Meteo geocoding search for a single name string.

        Args:
            name: City or place substring to search.

        Returns:
            First API result dict, or None when the results list is empty.
        """
        # Downloading the JSON geocoding payload for the given place name.
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1},
            timeout=timeout_s,
        )
        # Raising for non-2xx so Flask can surface network or server errors.
        resp.raise_for_status()
        # Parsing the JSON body and extracting the first hit when present.
        results = resp.json().get("results") or []
        return results[0] if results else None

    # Trying the full user string first (best when the name is unambiguous).
    r = _lookup(location)
    # Retrying with only the substring before a comma when the full string misses.
    if r is None and "," in location:
        r = _lookup(location.split(",", 1)[0].strip())
    # Failing with guidance when neither lookup strategy returns a row.
    if r is None:
        raise ValueError("Could not find that city. Try a more specific name or paste lat,lon.")

    # Reading coordinates from the chosen API result record.
    lat = float(r["latitude"])
    lon = float(r["longitude"])
    # Building a comma-separated label from name / region / country fields.
    label = ", ".join([p for p in [r.get("name"), r.get("admin1"), r.get("country")] if p])
    return lat, lon, label


@app.get("/")
def index_get() -> str:
    """
    Render the empty questionnaire (no prior plan or form state).

    Returns:
        HTML response body from ``index.html``.
    """
    # Serving the template with null context so the UI shows the first step only.
    return render_template("index.html", plan=None, error=None, form_data=None)


@app.post("/")
def index_post() -> str:
    """
    Accept the questionnaire POST, geocode, run the agent, render results.

    On success, passes a ``ShiftPlan`` instance to the template. On any
    exception, logs the stack trace and re-renders with ``error`` set.

    Returns:
        HTML response body from ``index.html``.
    """
    # Starting a high-resolution timer for request latency logging.
    t0 = time.perf_counter()

    # Binding form fields into a small dataclass for template repopulation.
    form = FormData(
        location=(request.form.get("location") or "").strip(),
        job_type=(request.form.get("job_type") or "").strip(),
        direct_sun=(request.form.get("direct_sun", "no") == "yes"),
        shift_start=int(request.form.get("shift_start") or 0),
        shift_end=int(request.form.get("shift_end") or 0),
    )

    try:
        # Validating that required free-text fields are present.
        if not form.location:
            raise ValueError("Please enter a location.")
        # Ensuring the job type matches one of the four radio-card values exactly.
        if form.job_type not in {"Light", "Moderate", "Heavy", "Very Heavy"}:
            raise ValueError("Please choose a valid job type.")

        # Resolving the location string into coordinates for the agent tools.
        lat, lon, location_label = _geocode_location(form.location)

        # Running the two-turn LLM agent to produce a validated shift plan model.
        plan = run_agent(
            lat=lat,
            lon=lon,
            job_type=form.job_type,
            direct_sun=form.direct_sun,
            shift_start=form.shift_start,
            shift_end=form.shift_end,
        )

        # Computing wall-clock duration for observability in Flask logs.
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # Writing an info line including resolved coordinates for debugging.
        app.logger.info(
            "HeatSense v2 request in %.0fms (location=%s, lat=%.4f, lon=%.4f)",
            latency_ms,
            location_label,
            lat,
            lon,
        )

        # Rendering the same template with the plan object for the results section.
        return render_template("index.html", plan=plan, error=None, form_data=form)

    except Exception as exc:
        # Logging the full traceback server-side while showing a short message in UI.
        app.logger.exception("HeatSense v2 error")
        # Returning the form with error text so the user can correct inputs.
        return render_template(
            "index.html",
            plan=None,
            error=str(exc),
            form_data=form,
        )


if __name__ == "__main__":
    # Starting the development server bound to localhost on port 5001.
    app.run(debug=True, host="127.0.0.1", port=5001)
