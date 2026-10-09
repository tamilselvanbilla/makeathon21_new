"""The only module permitted to make network calls.

Everything that leaves the device passes through `OnlineGateway.lookup`, which
accepts only a structured `LookupRequest` (never free text or audio), validates
every field it may send, and only talks to hosts on `ALLOWED_HOSTS`. The request
is built locally from the question, so the question itself never leaves the
device; all reasoning about the result also happens locally.

What each lookup sends:
- weather: a place name (then its coordinates)
- market:  ticker symbols and mutual-fund scheme codes (the whole watchlist,
           so the request doesn't reveal which holding was asked about)
- news:    nothing but a fixed feed address; topics are filtered on the device

Each kind can be switched off. Results are cached briefly; if a refresh fails,
the last result is reused and marked with when it was fetched.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .telemetry import timed_event

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Unofficial, undocumented Yahoo Finance endpoint: works without a key but may change.
QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
NAV_URL = "https://api.mfapi.in/mf/{code}/latest"  # AMFI's official NAVs, republished
FEEDS = {
    "business": ("The Hindu", "https://www.thehindu.com/business/feeder/default.rss"),
    "india": ("The Hindu", "https://www.thehindu.com/news/national/feeder/default.rss"),
    "technology": ("The Hindu", "https://www.thehindu.com/sci-tech/technology/feeder/default.rss"),
    "world": ("BBC News", "https://feeds.bbci.co.uk/news/world/rss.xml"),
}
ALLOWED_HOSTS = frozenset(
    {"geocoding-api.open-meteo.com", "api.open-meteo.com", "query1.finance.yahoo.com", "api.mfapi.in"}
    | {urllib.parse.urlparse(url).hostname for _, url in FEEDS.values()}
)
KINDS = ("weather", "market", "news")
TIMEOUT_SECONDS = 6
MAX_RESPONSE_BYTES = 2_000_000
CACHE_SECONDS = {"weather": 600, "quote": 300, "nav": 6 * 3600, "news": 900}

# What may be sent: a place name, not a sentence; ticker symbols; numeric fund codes.
PLACE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,59}$")
SYMBOL_PATTERN = re.compile(r"^\^?[A-Z0-9][A-Z0-9.&-]{0,19}$")
FUND_CODE_PATTERN = re.compile(r"^[0-9]{3,7}$")

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
    kind: str  # "weather" | "market" | "news" (anything else is refused)
    place: str = ""
    day: str = "today"  # "today" | "tomorrow"
    symbols: tuple[str, ...] = ()
    fund_codes: tuple[str, ...] = ()
    feeds: tuple[str, ...] = ()


@dataclass(frozen=True)
class Quote:
    symbol: str
    name: str
    price: float
    previous_close: float | None
    currency: str


@dataclass(frozen=True)
class Nav:
    code: str
    name: str
    nav: float
    date: str


@dataclass(frozen=True)
class Headline:
    source: str
    title: str
    published: str


@dataclass(frozen=True)
class LookupResult:
    source: str
    text: str
    data: Any = None  # structured facts for local computation (quotes, NAVs, headlines)
    stale_since: str = ""  # set when a refresh failed and cached data was reused


def check_host_allowed(url: str) -> None:
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise LookupUnavailable(f"Host '{host}' is not on the online allowlist.")


def fetch(url: str, params: dict[str, Any], as_text: bool = False) -> Any:
    """GET an allowlisted URL; return decoded JSON, or text when `as_text`."""
    check_host_allowed(url)
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(f"{url}{query}", headers={"User-Agent": "offline-ai-companion"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LookupUnavailable(f"I couldn't reach {urllib.parse.urlparse(url).hostname}.") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise LookupUnavailable("The online service sent an unexpectedly large response.")
    text = body.decode("utf-8", errors="replace")
    if as_text:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LookupUnavailable("The online service sent an unreadable response.") from exc


class OnlineGateway:
    def __init__(
        self,
        enabled: bool = True,
        fetch: Callable[..., Any] = fetch,
        kinds: tuple[str, ...] = KINDS,
        clock: Callable[[], float] = time.time,
    ):
        self.enabled = enabled
        self.kinds = frozenset(kinds) if enabled else frozenset()
        self._fetch = fetch
        self._clock = clock
        self._cache: dict[str, tuple[float, Any]] = {}

    def lookup(self, request: LookupRequest) -> LookupResult:
        if not isinstance(request, LookupRequest):
            raise TypeError("Only structured lookup requests may leave the device.")
        if not self.enabled:
            raise LookupUnavailable("Online lookups are switched off.")
        if request.kind not in KINDS:
            raise LookupUnavailable(f"Online {request.kind} lookups are not supported.")
        if request.kind not in self.kinds:
            raise LookupUnavailable(f"Online {request.kind} lookups are switched off.")
        self._validate(request)
        with timed_event("online_lookup", kind=request.kind):
            return getattr(self, f"_{request.kind}")(request)

    @staticmethod
    def _validate(request: LookupRequest) -> None:
        if request.kind == "weather" and not PLACE_PATTERN.fullmatch(request.place):
            raise LookupUnavailable("That doesn't look like a place name, so I won't send it online.")
        if request.kind == "market":
            if not request.symbols and not request.fund_codes:
                raise LookupUnavailable("There is nothing to look up.")
            if not all(SYMBOL_PATTERN.fullmatch(s) for s in request.symbols) or not all(
                FUND_CODE_PATTERN.fullmatch(c) for c in request.fund_codes
            ):
                raise LookupUnavailable("Only ticker symbols and fund codes may be sent for market data.")
        if request.kind == "news" and (not request.feeds or not set(request.feeds) <= FEEDS.keys()):
            raise LookupUnavailable("Only the built-in news feeds can be fetched.")

    def _get(self, cache_kind: str, url: str, params: dict[str, Any], as_text: bool = False) -> tuple[Any, str]:
        """Fetch with a short cache. Returns (data, stale_since); stale_since is empty
        when fresh, or the time of the cached copy reused after a failed refresh."""
        key = f"{url}?{json.dumps(params, sort_keys=True)}"
        now = self._clock()
        cached = self._cache.get(key)
        if cached and now - cached[0] < CACHE_SECONDS[cache_kind]:
            return cached[1], ""
        try:
            data = self._fetch(url, params, as_text=True) if as_text else self._fetch(url, params)
        except LookupUnavailable:
            if cached:
                return cached[1], datetime.fromtimestamp(cached[0]).strftime("%H:%M")
            raise
        self._cache[key] = (now, data)
        return data, ""

    def _weather(self, request: LookupRequest) -> LookupResult:
        places, _ = self._get(
            "weather", GEOCODING_URL, {"name": request.place, "count": 1, "language": "en", "format": "json"}
        )
        if not places.get("results"):
            raise LookupUnavailable(f"I couldn't find a place called {request.place}.")
        place = places["results"][0]
        name = ", ".join(part for part in (place.get("name"), place.get("country")) if part)

        forecast, _ = self._get(
            "weather",
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

    def _market(self, request: LookupRequest) -> LookupResult:
        quotes, navs, stale = {}, {}, []
        for symbol in request.symbols:
            try:
                data, since = self._get(
                    "quote", QUOTE_URL.format(symbol=urllib.parse.quote(symbol, safe="")), {"range": "1d", "interval": "1d"}
                )
                meta = data["chart"]["result"][0]["meta"]
                quotes[symbol] = Quote(
                    symbol=symbol,
                    name=meta.get("shortName") or meta.get("longName") or symbol,
                    price=float(meta["regularMarketPrice"]),
                    previous_close=meta.get("chartPreviousClose") or meta.get("previousClose"),
                    currency=meta.get("currency") or "",
                )
            except (LookupUnavailable, KeyError, IndexError, TypeError, ValueError):
                continue  # one missing symbol shouldn't sink the others
            stale += [since] if since else []
        for code in request.fund_codes:
            try:
                data, since = self._get("nav", NAV_URL.format(code=code), {})
                latest = data["data"][0]
                navs[code] = Nav(code, data["meta"]["scheme_name"], float(latest["nav"]), latest["date"])
            except (LookupUnavailable, KeyError, IndexError, TypeError, ValueError):
                continue
            stale += [since] if since else []
        if not quotes and not navs:
            raise LookupUnavailable("I couldn't reach the market data services.")
        sources = " and ".join(name for name, found in (("Yahoo Finance", quotes), ("AMFI", navs)) if found)
        return LookupResult(sources, "", data={"quotes": quotes, "navs": navs}, stale_since=min(stale, default=""))

    def _news(self, request: LookupRequest) -> LookupResult:
        headlines, stale = [], []
        for feed in request.feeds:
            source, url = FEEDS[feed]
            try:
                text, since = self._get("news", url, {}, as_text=True)
                root = ET.fromstring(text)
            except (LookupUnavailable, ET.ParseError):
                continue
            stale += [since] if since else []
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                if title:
                    headlines.append(Headline(source, title, (item.findtext("pubDate") or "").strip()))
        if not headlines:
            raise LookupUnavailable("I couldn't reach the news feeds.")
        sources = " and ".join(dict.fromkeys(h.source for h in headlines))
        return LookupResult(sources, "", data=headlines, stale_since=min(stale, default=""))
