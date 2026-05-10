"""
HeatSense v2 — end-to-end agent latency evaluation.

Measures wall-clock time for ``pipeline.agent.run_agent`` over a fixed suite of
30 geographic and shift scenarios (or 3 queries in ``--dry-run`` mode). Results
are written as JSON plus a plain-text summary for grading or regression checks.

Usage (from project root, where ``pipeline`` is importable)::

    python evaluation/evaluate_latency.py
    python evaluation/evaluate_latency.py --dry-run
    python evaluation/evaluate_latency.py --output-dir evaluation/custom_results

Requires ``OPENAI_API_KEY`` in a ``.env`` file at the project root. Loads
environment variables before importing the pipeline so keys are available to
the agent and RAG stack.

Output files (default under ``evaluation/results/``)::

    latency_results.json   — per-query rows and aggregate summary
    latency_summary.txt    — human-readable copy of the printed summary block
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

# Ensuring the project root is on sys.path before any ``pipeline`` imports.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Loading secrets before any project imports that construct OpenAI clients.
load_dotenv(_ROOT / ".env")


@dataclass
class QuerySpec:
    """One evaluation scenario: location label, coordinates, job, sun, shift."""

    index: int
    job_type: str
    location: str
    lat: float
    lon: float
    direct_sun: bool
    shift_start: int
    shift_end: int


@dataclass
class QueryResult:
    """Outcome of a single timed ``run_agent`` call."""

    index: int
    job_type: str
    location: str
    lat: float
    lon: float
    direct_sun: bool
    shift_start: int
    shift_end: int
    latency_s: float
    success: bool
    error: str | None


LOCATIONS: list[tuple[str, float, float]] = [
    ("Houston TX", 29.7604, -95.3698),
    ("Phoenix AZ", 33.4484, -112.0740),
    ("Miami FL", 25.7617, -80.1918),
    ("Las Vegas NV", 36.1699, -115.1398),
    ("New Orleans LA", 29.9511, -90.0715),
]

SHIFTS: list[tuple[int, int]] = [
    (6, 14),   # 06:00-14:00
    (8, 16),   # 08:00-16:00
    (10, 18),  # 10:00-18:00
]

JOB_TYPES: list[str] = ["Light", "Moderate", "Heavy", "Very Heavy"]


def build_query_specs(count: int = 30) -> list[QuerySpec]:
    """
    Build ``count`` query specifications with balanced cycling.

    Cycling rules (for the default 30 queries):

    - Job type cycles Light → Moderate → Heavy → Very Heavy.
    - ``direct_sun`` alternates True / False each index (15 of each).
    - Location cycles the five cities in order.
    - Shift window cycles the three predefined windows.

    Args:
        count: Number of specs to generate (30 full suite, or 3 for dry-run).

    Returns:
        List of ``QuerySpec`` rows with 1-based ``index`` for display.
    """
    # Starting with an empty list and appending one spec per loop iteration.
    specs: list[QuerySpec] = []
    for i in range(count):
        # Using modular arithmetic so job, sun, city, and shift patterns repeat.
        job_type = JOB_TYPES[i % 4]
        direct_sun = i % 2 == 0
        loc_name, lat, lon = LOCATIONS[i % 5]
        shift_start, shift_end = SHIFTS[i % 3]
        specs.append(
            QuerySpec(
                index=i + 1,
                job_type=job_type,
                location=loc_name,
                lat=lat,
                lon=lon,
                direct_sun=direct_sun,
                shift_start=shift_start,
                shift_end=shift_end,
            )
        )
    return specs


def _sun_label(direct: bool) -> str:
    """
    Map a boolean sun flag to a short label for progress lines.

    Args:
        direct: True when the scenario uses direct sun.

    Returns:
        Either ``"direct sun"`` or ``"shade"``.
    """
    return "direct sun" if direct else "shade"


def _shift_label(start: int, end: int) -> str:
    """
    Format a shift window as ASCII clock labels for stdout.

    Args:
        start: Inclusive shift start hour (0-23).
        end:   Exclusive shift end hour (1-23).

    Returns:
        A string like ``"06:00-14:00"`` using hyphen-minus only.
    """
    return f"{start:02d}:00-{end:02d}:00"


def run_queries(specs: list[QuerySpec]) -> list[QueryResult]:
    """
    Execute ``run_agent`` for each spec and collect timings and errors.

    Importing ``run_agent`` here so dotenv has already loaded ``.env``.

    Args:
        specs: Query scenarios to run in order.

    Returns:
        List of ``QueryResult`` with latency and success flag per query.
    """
    # Importing here so ``OPENAI_API_KEY`` is loaded before the agent module runs.
    from pipeline.agent import run_agent

    # Starting the accumulator for per-query outcomes.
    results: list[QueryResult] = []
    total = len(specs)
    for spec in specs:
        # Starting the wall-clock timer immediately before the agent call.
        t0 = time.perf_counter()
        err: str | None = None
        ok = True
        try:
            run_agent(
                lat=spec.lat,
                lon=spec.lon,
                job_type=spec.job_type,
                direct_sun=spec.direct_sun,
                shift_start=spec.shift_start,
                shift_end=spec.shift_end,
            )
        except Exception as exc:  # noqa: BLE001
            # Catching broadly so one bad response never aborts the whole suite.
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        # Stopping the timer right after success or failure returns control here.
        elapsed = time.perf_counter() - t0

        line = (
            f"[{spec.index:02d}/{total}] {spec.job_type} / "
            f"{_sun_label(spec.direct_sun)} / "
            f"{spec.location} ({_shift_label(spec.shift_start, spec.shift_end)}) ... "
            f"{elapsed:.2f}s {'OK' if ok else 'ERROR: ' + (err or '')}"
        )
        # Using flush so long runs show lines immediately under output capture.
        print(line, flush=True)

        results.append(
            QueryResult(
                index=spec.index,
                job_type=spec.job_type,
                location=spec.location,
                lat=spec.lat,
                lon=spec.lon,
                direct_sun=spec.direct_sun,
                shift_start=spec.shift_start,
                shift_end=spec.shift_end,
                latency_s=round(elapsed, 2),
                success=ok,
                error=err,
            )
        )
    return results


def summarize(results: list[QueryResult]) -> dict[str, Any]:
    """
    Compute aggregate latency statistics over successful runs only.

    Args:
        results: Completed query rows including failures.

    Returns:
        Dict suitable for JSON ``summary`` field and text report.
    """
    # Building a 1-D array of latencies for rows where ``success`` is true.
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    lat_ok = np.array([r.latency_s for r in results if r.success], dtype=float)

    if lat_ok.size == 0:
        # Using NaN placeholders when no run succeeded so math does not divide.
        median_s = mean_s = min_s = max_s = p90_s = float("nan")
        verdict = "FAIL"
    else:
        # Computing standard order statistics with NumPy vectorized helpers.
        median_s = float(np.median(lat_ok))
        mean_s = float(np.mean(lat_ok))
        min_s = float(np.min(lat_ok))
        max_s = float(np.max(lat_ok))
        p90_s = float(np.percentile(lat_ok, 90))
        # Applying the course-style pass rule on the median of successes only.
        verdict = "PASS" if median_s < 5.0 else "FAIL"

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "median_s": round(median_s, 2) if lat_ok.size else None,
        "mean_s": round(mean_s, 2) if lat_ok.size else None,
        "min_s": round(min_s, 2) if lat_ok.size else None,
        "max_s": round(max_s, 2) if lat_ok.size else None,
        "p90_s": round(p90_s, 2) if lat_ok.size else None,
        "target_median_s": 5.0,
        "verdict": verdict,
    }


def format_summary_block(summary: dict[str, Any]) -> str:
    """
    Render the latency summary as fixed-width plain text.

    Args:
        summary: Dict produced by ``summarize`` including counts and stats.

    Returns:
        Multi-line string matching the printed summary block format.
    """
    med = summary["median_s"]
    mean = summary["mean_s"]
    min_s = summary["min_s"]
    max_s = summary["max_s"]
    p90 = summary["p90_s"]

    def fmt(v: Any) -> str:
        """Format one statistic for the text table (seconds or ``n/a``)."""
        # Treating None or NaN as missing so graders see a clear placeholder.
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{float(v):.2f}s"

    lines = [
        "===== LATENCY SUMMARY =====",
        f"Total queries : {summary['total']}",
        f"Successful    : {summary['successful']}",
        f"Failed        : {summary['failed']}",
        "",
        f"Median  : {fmt(med)}",
        f"Mean    : {fmt(mean)}",
        f"Min     : {fmt(min_s)}",
        f"Max     : {fmt(max_s)}",
        f"P90     : {fmt(p90)}",
        "",
        f"Target  : < {summary['target_median_s']:.2f}s median",
        f"Verdict : {summary['verdict']}",
        "===========================",
    ]
    # Joining lines with Unix newlines so files open consistently on all OSes.
    return "\n".join(lines)


def main() -> int:
    """
    Parse CLI flags, run the suite, write outputs, and print the summary.

    Returns:
        0 when the median target passes, 1 when it fails or the key is missing.
    """
    parser = argparse.ArgumentParser(
        description="HeatSense v2 agent latency evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the first 3 queries (quick smoke test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "evaluation" / "results",
        help="Directory for latency_results.json and latency_summary.txt.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY not set. Add it to .env at the project root.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    n_queries = 3 if args.dry_run else 30
    specs = build_query_specs(n_queries)

    print(f"Starting latency evaluation ({n_queries} queries)...\n", flush=True)
    results = run_queries(specs)
    summary = summarize(results)

    # Building the JSON-serializable payload matching the grading schema.
    payload = {
        "summary": summary,
        "queries": [
            {
                "index": r.index,
                "job_type": r.job_type,
                "location": r.location,
                "lat": r.lat,
                "lon": r.lon,
                "direct_sun": r.direct_sun,
                "shift_start": r.shift_start,
                "shift_end": r.shift_end,
                "latency_s": r.latency_s,
                "success": r.success,
                "error": r.error,
            }
            for r in results
        ],
    }

    out_dir: Path = args.output_dir
    # Creating the output tree when it does not exist yet.
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latency_results.json"
    txt_path = out_dir / "latency_summary.txt"

    # Writing JSON with UTF-8 for stable Unicode and readable indentation.
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    block = format_summary_block(summary)
    # Writing the plain-text copy for quick human review outside the terminal.
    txt_path.write_text(block + "\n", encoding="utf-8")

    print(flush=True)
    print(block, flush=True)
    print(flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {txt_path}", flush=True)
    # Returning non-zero when the median target is missed (CI-friendly).
    return 1 if summary.get("verdict") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
