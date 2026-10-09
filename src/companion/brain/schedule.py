"""Dates and times in spoken notes: "meeting with Jay tomorrow at 7am".

`parse_when` finds when something happens, relative to now, and the words that
said so (so they can be dropped from the description). It understands what
Whisper writes for everyday speech: today / tonight / tomorrow / day after
tomorrow / weekdays / "12 March" / "March 12", and "7am", "7 a.m.", "7:30 pm",
"at 7", "noon", "midnight", "in 10 minutes", "in 2 hours".
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

RELATIVE = re.compile(r"\bin\s+(\d{1,3}|an?|one|two|three|five|ten|fifteen|twenty|thirty)\s+(minutes?|mins?|hours?|hrs?)\b", re.I)
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
DAY_AFTER = re.compile(r"\bday after tomorrow\b", re.I)
TOMORROW = re.compile(r"\btomorrow(?:\s+(?:morning|afternoon|evening|night))?\b", re.I)
TODAY = re.compile(r"\b(?:today|tonight|this\s+(?:morning|afternoon|evening))\b", re.I)
WEEKDAY = re.compile(r"\b(?:on\s+|next\s+|this\s+)?(" + "|".join(WEEKDAYS) + r")\b", re.I)
DAY_MONTH = re.compile(r"\b(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + MONTH + r")\b", re.I)
MONTH_DAY = re.compile(r"\b(?:on\s+)?(" + MONTH + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
CLOCK = re.compile(r"\b(?:at\s+)?(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)(?=\W|$)", re.I)
AT_HOUR = re.compile(r"\bat\s+(\d{1,2})(?:[:.](\d{2}))?\b(?!\s*(?:%|percent|degrees))", re.I)
NOON = re.compile(r"\b(?:at\s+)?(noon|midday|midnight)\b", re.I)


@dataclass(frozen=True)
class When:
    due: datetime  # for a date without a time, midnight of that day
    has_time: bool
    spans: tuple[tuple[int, int], ...]  # where the date/time words are in the text


def _month(name: str) -> int:
    return next(i for i, m in enumerate(MONTHS, 1) if m.startswith(name.casefold()[:3]))


def parse_when(text: str, now: datetime) -> When | None:
    spans: list[tuple[int, int]] = []

    relative = RELATIVE.search(text)
    if relative:
        amount = relative.group(1).casefold()
        n = int(amount) if amount.isdigit() else NUMBER_WORDS[amount]
        delta = timedelta(minutes=n) if relative.group(2).casefold().startswith("m") else timedelta(hours=n)
        return When((now + delta).replace(second=0, microsecond=0), True, (relative.span(),))

    day: date | None = None
    for pattern, offset in ((DAY_AFTER, 2), (TOMORROW, 1), (TODAY, 0)):
        match = pattern.search(text)
        if match:
            day = now.date() + timedelta(days=offset)
            spans.append(match.span())
            break
    if day is None and (match := WEEKDAY.search(text)):
        ahead = (WEEKDAYS.index(match.group(1).casefold()) - now.weekday()) % 7 or 7
        day = now.date() + timedelta(days=ahead)
        spans.append(match.span())
    if day is None and (match := DAY_MONTH.search(text) or MONTH_DAY.search(text)):
        first, second = match.group(1), match.group(2)
        number, month = (first, second) if first.isdigit() else (second, first)
        try:
            day = date(now.year, _month(month), int(number))
            if day < now.date():
                day = date(now.year + 1, day.month, day.day)
            spans.append(match.span())
        except ValueError:
            day = None

    clock: time | None = None
    if match := CLOCK.search(text):
        hour, minute = int(match.group(1)) % 12, int(match.group(2) or 0)
        if match.group(3).casefold().startswith("p"):
            hour += 12
        if hour < 24 and minute < 60:
            clock = time(hour, minute)
            spans.append(match.span())
    elif match := NOON.search(text):
        clock = time(0, 0) if match.group(1).casefold() == "midnight" else time(12, 0)
        spans.append(match.span())
    elif match := AT_HOUR.search(text):
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if hour < 24 and minute < 60:
            if hour < 12 and day is None:
                # "at 6" with no am/pm: the next 6 o'clock to come
                candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                while candidate <= now:
                    candidate += timedelta(hours=12)
                hour, day = candidate.hour, candidate.date()
            clock = time(hour, minute)
            spans.append(match.span())

    if day is None and clock is None:
        return None
    if day is None:  # a time alone: today if still to come, otherwise tomorrow
        day = now.date() if datetime.combine(now.date(), clock) > now else now.date() + timedelta(days=1)
    return When(datetime.combine(day, clock or time(0, 0)), clock is not None, tuple(spans))


def without_when(text: str, when: When) -> str:
    """The text with its date/time words removed: "meeting with Jay"."""
    for start, end in sorted(when.spans, reverse=True):
        text = text[:start] + text[end:]
    return re.sub(r"\s+", " ", text).strip(" ,.-")


def spoken_day(day: date, today: date) -> str:
    if day == today:
        return f"today, {day.day} {day:%B}"
    if day == today + timedelta(days=1):
        return f"tomorrow, {day.day} {day:%B}"
    return f"{day:%A}, {day.day} {day:%B}"


def spoken_time(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")
