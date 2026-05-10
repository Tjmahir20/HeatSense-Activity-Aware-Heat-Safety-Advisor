"""
Open-Meteo weather fetch and deterministic WBGT computation.

This module turns hourly forecast inputs (temperature, humidity, radiation,
wind) into Wet-Bulb Globe Temperature estimates for each shift hour. No LLM
calls occur here; results feed the agent tool ``fetch_wbgt_forecast``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class HourlyWBGT:
    """Immutable record of one forecast hour and derived heat-stress inputs."""

    hour: int
    Ta: float
    RH: float
    SR: float
    wind_speed: float
    Tw: float
    Tg: float
    WBGT: float

    def as_dict(self) -> dict[str, Any]:
        """
        Serialize this hour to a plain dict for JSON-friendly tool output.

        Returns:
            Mapping with keys hour, Ta, RH, SR, wind_speed, Tw, Tg, WBGT.
        """
        # Building a shallow dict so callers avoid touching dataclass internals.
        return {
            "hour": self.hour,
            "Ta": self.Ta,
            "RH": self.RH,
            "SR": self.SR,
            "wind_speed": self.wind_speed,
            "Tw": self.Tw,
            "Tg": self.Tg,
            "WBGT": self.WBGT,
        }


def _stull_wet_bulb_c(T: float, RH: float) -> float:
    """
    Approximate natural wet-bulb temperature (°C) from dry-bulb T and RH.

    Uses the Stull (2011) polynomial approximation; suitable for outdoor
    screening when a physical psychrometer is unavailable.

    Args:
        T:  Dry-bulb air temperature in degrees Celsius.
        RH: Relative humidity in percent (0–100).

    Returns:
        Estimated wet-bulb temperature in degrees Celsius.
    """
    # Computing wet-bulb via Stull’s closed form (no iteration).
    return (
        T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
        + math.atan(T + RH)
        - math.atan(RH - 1.676331)
        + 0.00391838 * (RH ** 1.5) * math.atan(0.023101 * RH)
        - 4.686035
    )


def _hunter_minyard_globe_c(Ta: float, SR: float, wind_speed: float) -> float:
    """
    Approximate black-globe temperature (°C) from air temp, sun, and wind.

    Uses a simplified Hunter–Minyard style relationship so WBGT can include
    radiative heating without requiring a physical globe sensor.

    Args:
        Ta:          Ambient air temperature (°C).
        SR:          Shortwave radiation (W/m²) from the forecast API.
        wind_speed:  Wind speed at 10 m (m/s).

    Returns:
        Estimated globe temperature in degrees Celsius.
    """
    # Guarding against zero wind so the denominator stays numerically stable.
    ws = max(wind_speed, 0.0)
    # Computing globe excess from solar load damped by wind mixing.
    return Ta + 0.0066 * SR / (math.sqrt(ws) + 0.5)


def _outdoor_wbgt_c(Ta: float, Tw: float, Tg: float, direct_sun: bool) -> float:
    """
    Combine wet-bulb, globe, and air temperature into outdoor WBGT (°C).

    Follows the Yaglou & Minard (1957) weighting; sun vs shade changes how
    much globe temperature contributes.

    Args:
        Ta:         Dry-bulb air temperature (°C).
        Tw:         Wet-bulb temperature (°C).
        Tg:         Globe temperature (°C).
        direct_sun: True when the worker is in full sun (outdoor WBGT formula).

    Returns:
        Estimated WBGT in degrees Celsius.
    """
    # Weighting globe more heavily in shade (less direct radiative load).
    if direct_sun:
        return 0.7 * Tw + 0.2 * Tg + 0.1 * Ta
    return 0.7 * Tw + 0.3 * Tg


def _shift_hours(shift_start: int, shift_end: int) -> list[int]:
    """
    Produce the list of hour indices included in a same-day shift window.

    Hours are 0–23 inclusive; the end hour is exclusive so ``range`` matches
    typical ``[start, end)`` half-open intervals.

    Args:
        shift_start: First hour of the shift (0–23).
        shift_end:   Hour after the last included hour (1–23, must exceed start).

    Returns:
        List of integer hour indices, e.g. [7, 8, …, 14] for 07:00–15:00.

    Raises:
        ValueError: If bounds are outside 0–23 or ``shift_end`` ≤ ``shift_start``.
    """
    # Validating that the UI window stays inside a single calendar day.
    if not (0 <= shift_start <= 23 and 0 <= shift_end <= 23):
        raise ValueError("Shift hours must be within 0-23.")
    # Rejecting empty or inverted windows early for clearer errors.
    if shift_end <= shift_start:
        raise ValueError("shift_end must be greater than shift_start (same-day shift).")
    # Building the inclusive-exclusive hour span the rest of the pipeline expects.
    return list(range(shift_start, shift_end))


def fetch_hourly_wbgt(
    *,
    latitude: float,
    longitude: float,
    shift_start: int,
    shift_end: int,
    direct_sun: bool,
    timeout_s: int = 20,
) -> list[dict[str, Any]]:
    """
    Download Open-Meteo hourly forecast and compute WBGT for each shift hour.

    Calls the public forecast endpoint once, then applies Stull wet-bulb,
    Hunter–Minyard globe, and Yaglou–Minard WBGT per requested hour index.

    Args:
        latitude:    Site latitude (decimal degrees).
        longitude:   Site longitude (decimal degrees).
        shift_start: Shift start hour (0–23).
        shift_end:   Shift end hour (exclusive, 1–23).
        direct_sun:  Whether the outdoor WBGT sun formula should be used.
        timeout_s:   HTTP timeout for the forecast request.

    Returns:
        List of dicts (see ``HourlyWBGT.as_dict``), one per shift hour.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        RuntimeError: If the API payload shape is unexpected.
        ValueError: Propagated from ``_shift_hours`` for invalid windows.
    """
    # Resolving which hour indices we must materialize from the 24h arrays.
    hours = _shift_hours(shift_start, shift_end)

    # Preparing the Open-Meteo REST call (no API key required).
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "shortwave_radiation",
            "wind_speed_10m",
        ]),
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "forecast_days": 1,
    }

    # Downloading the JSON forecast document for the requested coordinates.
    resp = requests.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    # Parsing the response body into native Python structures.
    data = resp.json()

    # Extracting parallel hourly series; missing keys become empty lists.
    hourly = data.get("hourly") or {}
    temps = hourly.get("temperature_2m") or []
    rhs = hourly.get("relative_humidity_2m") or []
    srs = hourly.get("shortwave_radiation") or []
    winds = hourly.get("wind_speed_10m") or []

    # Verifying all series align and cover at least a full day of hours.
    if not (len(temps) == len(rhs) == len(srs) == len(winds)) or len(temps) < 24:
        raise RuntimeError("Unexpected Open-Meteo hourly response format.")

    # Accumulating one dict per shift hour after thermodynamic transforms.
    out: list[dict[str, Any]] = []
    for h in hours:
        # Reading raw scalars at hour index h from the parallel arrays.
        Ta = float(temps[h])
        RH = float(rhs[h])
        SR = float(srs[h])
        wind_speed = float(winds[h])
        # Computing intermediate wet-bulb and globe temperatures.
        Tw = float(_stull_wet_bulb_c(Ta, RH))
        Tg = float(_hunter_minyard_globe_c(Ta, SR, wind_speed))
        # Combining components into final WBGT for this clock hour.
        wbgt = float(_outdoor_wbgt_c(Ta, Tw, Tg, direct_sun))

        # Appending the frozen dataclass serialized for downstream JSON use.
        out.append(
            HourlyWBGT(
                hour=h,
                Ta=Ta,
                RH=RH,
                SR=SR,
                wind_speed=wind_speed,
                Tw=Tw,
                Tg=Tg,
                WBGT=wbgt,
            ).as_dict()
        )

    return out
