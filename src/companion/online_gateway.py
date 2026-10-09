"""The only module permitted to make network calls.

Everything that leaves the device passes through `OnlineGateway.lookup`, which
accepts only a structured `LookupRequest` (never free text or audio), checks
that the place name looks like a place, and only talks to hosts on
`ALLOWED_HOSTS`. The request is built locally from the question, so the
question itself never leaves the device; all reasoning about the result also
happens locally.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .telemetry import timed_event

ALLOWED_HOSTS = frozenset({"geocoding-api.open-meteo.com", "api.open-meteo.com"})
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 6
# Letters, spaces, and a few punctuation marks: a place name, not a sentence.
PLACE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,59}$")

# WMO weather interpretation codes used by Open-Meteo.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "severe thunderstorms with hail",
}


class LookupUnavailable(Exception):
    """Raised when a lookup cannot or must not be performed."""


@dataclass(frozen=True)
class LookupRequest:
    kind: str  # "weather" | "news" | "search"
    place: str = ""
    day: str = "today"  # "today" | "tomorrow"


@dataclass(frozen=True)
class LookupResult:
    source: str
    text: str


def check_host_allowed(url: str) -> None:
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise LookupUnavailable(f"Host '{host}' is not on the online allowlist.")


def fetch_json(url: str, params: dict[str, Any]) -> Any:
    """GET an allowlisted URL and decode its JSON body."""
    check_host_allowed(url)
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "offline-ai-companion"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LookupUnavailable("I couldn't reach the weather service.") from exc


class OnlineGateway:
    def __init__(self, enabled: bool = True, fetch: Callable[[str, dict[str, Any]], Any] = fetch_json):
        self.enabled = enabled
        self._fetch = fetch

    def lookup(self, request: LookupRequest) -> LookupResult:
        if not isinstance(request, LookupRequest):
            raise TypeError("Only structured lookup requests may leave the device.")
        if not self.enabled:
            raise LookupUnavailable("Online lookups are switched off.")
        if request.kind != "weather":
            raise LookupUnavailable(f"Online {request.kind} lookups are not set up yet; only weather is.")
        if not PLACE_PATTERN.fullmatch(request.place):
            raise LookupUnavailable("That doesn't look like a place name, so I won't send it online.")
        with timed_event("online_lookup", kind=request.kind):
            return self._weather(request)

    def _weather(self, request: LookupRequest) -> LookupResult:
        places = self._fetch(GEOCODING_URL, {"name": request.place, "count": 1, "language": "en", "format": "json"})
        if not places.get("results"):
            raise LookupUnavailable(f"I couldn't find a place called {request.place}.")
        place = places["results"][0]
        name = ", ".join(part for part in (place.get("name"), place.get("country")) if part)

        forecast = self._fetch(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 2,
            },
        )
        daily = forecast["daily"]
        day = 1 if request.day == "tomorrow" else 0
        rain = daily["precipitation_probability_max"][day]
        outlook = (
            f"{WEATHER_CODES.get(daily['weather_code'][day], 'mixed weather')}, "
            f"{daily['temperature_2m_min'][day]:.0f} to {daily['temperature_2m_max'][day]:.0f} degrees"
            + (f", with a {rain:.0f} percent chance of rain" if rain is not None else "")
        )
        if day == 1:
            return LookupResult(source="Open-Meteo", text=f"In {name} tomorrow: {outlook}.")

        now = forecast["current"]
        text = (
            f"In {name} it is {now['temperature_2m']:.0f} degrees and "
            f"{WEATHER_CODES.get(now['weather_code'], 'mixed weather')}, feels like "
            f"{now['apparent_temperature']:.0f}, humidity {now['relative_humidity_2m']:.0f} percent. "
            f"Today: {outlook}."
        )
        return LookupResult(source="Open-Meteo", text=text)
