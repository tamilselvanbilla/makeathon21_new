"""Share market and portfolio answers.

Public facts (prices, index levels, fund NAVs) are fetched online by symbol or
scheme code only. Your holdings (quantities, what you paid) never leave the
device, and every calculation is done here in Python: a 0.6B model can't be
trusted with arithmetic. The whole watchlist is always requested, so the
request doesn't reveal which holding the question was about.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..online_gateway import LookupRequest, LookupResult
from .knowledge import Records

INDEX_NAMES = {"^NSEI": "Nifty 50", "^BSESN": "Sensex"}
DISCLAIMER = "This is information, not investment advice."
MARKET_WORDS = re.compile(r"\b(?:nifty|sensex|stock market|share market|markets?)\b", re.I)
FUND_WORDS = re.compile(r"\b(?:mutual funds?|funds?|nav)\b", re.I)
STOCK_WORDS = re.compile(r"\b(?:stocks?|shares?|equity|equities)\b", re.I)
HOLDING_WORDS = re.compile(
    r"\b(?:stocks?|shares?|portfolio|investments?|mutual funds?|funds?|nav|holdings?)\b", re.I
)
LIVE_WORDS = re.compile(
    r"\b(?:doing|today|now|worth|value|valued|prices?|priced|trading|profit|loss|gains?|"
    r"performing|performance|up|down|current|currently|latest|live)\b",
    re.I,
)
NAME_SUFFIXES = re.compile(r"\s+(?:ltd\.?|limited|inc\.?|corp\.?)$", re.I)


@dataclass(frozen=True)
class Holding:
    kind: str  # "stock" | "fund"
    name: str
    key: str  # ticker symbol or scheme code
    quantity: float  # shares or units
    cost: float  # total amount paid


class Portfolio:
    def __init__(self, holdings: list[Holding], symbols: dict[str, str], indices: list[str]):
        self.holdings = holdings
        self.indices = indices
        # Names the user may say -> symbol or scheme code; holdings first.
        self.names = {name.casefold(): symbol for name, symbol in symbols.items()}
        for holding in holdings:
            self.names.setdefault(holding.name.casefold(), holding.key)
        # Spoken name per symbol: the longest alias ("Tata Consultancy Services", not "TCS"),
        # since online short names can be truncated ("Tata Consultancy Serv Lt").
        self.spoken = {}
        for name, symbol in symbols.items():
            if len(name) > len(self.spoken.get(symbol, "")):
                self.spoken[symbol] = name.title()

    @classmethod
    def load(cls, records: Records, owner: str, symbols_file: Path) -> "Portfolio":
        try:
            config = json.loads(symbols_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            config = {}
        holdings = []
        for entry in records.get("financial", []):
            if entry.get("owner", owner) != owner:
                continue
            if entry.get("record_type") == "stock" and entry.get("ticker") and entry.get("quantity"):
                name = NAME_SUFFIXES.sub("", str(entry.get("company") or entry["ticker"]))
                quantity = float(entry["quantity"])
                holdings.append(
                    Holding("stock", name, str(entry["ticker"]), quantity, quantity * float(entry.get("purchase_price", 0)))
                )
            if entry.get("record_type") == "mutual_fund" and entry.get("scheme_code") and entry.get("units"):
                holdings.append(
                    Holding(
                        "fund",
                        str(entry.get("scheme") or entry["scheme_code"]),
                        str(entry["scheme_code"]),
                        float(entry["units"]),
                        float(entry.get("investment_amount", 0)),
                    )
                )
        return cls(holdings, config.get("symbols", {}), config.get("indices", ["^NSEI", "^BSESN"]))

    def named(self, text: str) -> list[str]:
        """Symbols or scheme codes of the companies, funds, or indices named in the text."""
        lowered = text.casefold()
        found = [key for name, key in self.names.items() if re.search(rf"\b{re.escape(name)}\b", lowered)]
        return list(dict.fromkeys(found))

    def is_market_question(self, text: str) -> bool:
        if MARKET_WORDS.search(text):
            return True
        live = bool(LIVE_WORDS.search(text))
        if self.named(text) and (live or HOLDING_WORDS.search(text)):
            return True  # "how is Infosys doing?", "TCS share price"
        return bool(HOLDING_WORDS.search(text) and live)  # "how are my investments doing?"

    def request(self, text: str) -> tuple[LookupRequest, list[str] | None]:
        """The lookup to send and what the answer should focus on (None = whole portfolio)."""
        named = self.named(text)
        if named:
            focus = named
        elif FUND_WORDS.search(text):
            focus = [h.key for h in self.holdings if h.kind == "fund"]
        elif STOCK_WORDS.search(text) and not MARKET_WORDS.search(text):
            focus = [h.key for h in self.holdings if h.kind == "stock"]
        elif MARKET_WORDS.search(text) and not HOLDING_WORDS.search(text):
            focus = self.indices
        else:
            focus = None
        stock_keys = [h.key for h in self.holdings if h.kind == "stock"]
        symbols = list(dict.fromkeys(stock_keys + self.indices + [k for k in named if not k.isdigit()]))
        fund_codes = [h.key for h in self.holdings if h.kind == "fund"]
        return LookupRequest("market", symbols=tuple(symbols), fund_codes=tuple(fund_codes)), focus

    def answer(self, result: LookupResult, focus: list[str] | None) -> str:
        quotes, navs = result.data["quotes"], result.data["navs"]
        holdings = [h for h in self.holdings if focus is None or h.key in focus]
        facts = []
        for key in focus if focus is not None else [*self.indices, *(h.key for h in holdings)]:
            if key in quotes:
                facts.append(_quote_fact(quotes[key], self._display_name(key, quotes[key].name)))
            elif key in navs:
                nav = navs[key]
                facts.append(f"{self._display_name(key, nav.name)} NAV is {nav.nav:,.2f} rupees, as of {_spoken_date(nav.date)}.")
            elif not key.isdigit() or any(h.key == key for h in holdings):
                facts.append(f"I couldn't get a price for {self._display_name(key, key)}.")

        mine = [line for h in holdings if (line := _holding_line(h, quotes, navs))]
        priced = [h for h in holdings if h.key in quotes or h.key in navs]
        if len(priced) > 1:
            mine.append(_total_line(priced, quotes, navs))

        reply = f"From {result.source}, online: {' '.join(facts)}"
        if result.stale_since:
            reply += f" These are from {result.stale_since}; I couldn't refresh them."
        if mine:
            reply += f" Computed on this device: {' '.join(line[0].upper() + line[1:] for line in mine)}"
        return f"{reply} {DISCLAIMER}"

    def _display_name(self, key: str, fallback: str) -> str:
        if key in INDEX_NAMES:
            return INDEX_NAMES[key]
        holding = next((h for h in self.holdings if h.key == key), None)
        return holding.name if holding else self.spoken.get(key, fallback.title())


def _spoken_date(date: str) -> str:
    """AMFI's "08-10-2026" -> "8 October"."""
    try:
        day = datetime.strptime(date, "%d-%m-%Y")
    except ValueError:
        return date
    return f"{day.day} {day:%B}"


def _quote_fact(quote, name: str) -> str:
    unit = "rupees" if quote.currency == "INR" else quote.currency
    level = f"{quote.price:,.2f}" + (f" {unit}" if not quote.symbol.startswith("^") else "")
    if not quote.previous_close:
        return f"{name} is at {level}."
    change = (quote.price - quote.previous_close) / quote.previous_close * 100
    direction = "up" if change >= 0 else "down"
    return f"{name} is at {level}, {direction} {abs(change):.1f} percent today."


def _value(holding: Holding, quotes: dict, navs: dict) -> float | None:
    if holding.key in quotes:
        return holding.quantity * quotes[holding.key].price
    if holding.key in navs:
        return holding.quantity * navs[holding.key].nav
    return None


def _difference(value: float, cost: float, paid: str) -> str:
    diff = value - cost
    percent = f", {'up' if diff >= 0 else 'down'} {abs(diff) / cost * 100:.1f} percent" if cost else ""
    return f"{abs(diff):,.0f} rupees {'above' if diff >= 0 else 'below'} {paid}{percent}"


def _holding_line(holding: Holding, quotes: dict, navs: dict) -> str:
    value = _value(holding, quotes, navs)
    if value is None:
        return ""
    what = f"{holding.quantity:g} {holding.name} shares" if holding.kind == "stock" else (
        f"{holding.quantity:g} units of {holding.name}"
    )
    paid = "what you paid" if holding.kind == "stock" else "your investment"
    return f"your {what} are worth {value:,.0f} rupees, {_difference(value, holding.cost, paid)}."


def _total_line(holdings: list[Holding], quotes: dict, navs: dict) -> str:
    value = sum(_value(h, quotes, navs) for h in holdings)
    cost = sum(h.cost for h in holdings)
    return f"In total these are worth {value:,.0f} rupees, {_difference(value, cost, f'the {cost:,.0f} you invested')}."
