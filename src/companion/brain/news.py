"""News headlines: whole feeds are downloaded, topics are filtered on the device.

The request names only built-in feeds, never the question or topic, so asking
"any news about my employer?" reveals nothing beyond "fetched the news".
Headlines are read out verbatim rather than summarised by the small model,
which could distort them.
"""

import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

from ..online_gateway import FEEDS, Headline, LookupRequest, LookupResult
from .knowledge import STOPWORDS, tokenize

NEWS_WORDS = re.compile(r"\b(?:news|headlines?)\b", re.I)
FEED_WORDS = {
    "business": {"business", "market", "markets", "economy", "economic", "finance", "financial", "stock", "stocks", "shares"},
    "india": {"india", "indian", "national"},
    "technology": {"tech", "technology", "gadgets", "ai", "science"},
    "world": {"world", "international", "global"},
}
DEFAULT_FEEDS = ("india", "world")
TOPIC = re.compile(r"\b(?:about|on|regarding|related to|for)\s+(.+?)\W*$", re.I)
FILLER = {"news", "headline", "headlines", "latest", "today", "any", "some", "top", "the", "recent", "breaking",
          "favourite", "favorite", "new", "happening"}
ORDINALS = ("One", "Two", "Three", "Four", "Five")


def is_news_question(text: str) -> bool:
    return bool(NEWS_WORDS.search(text))


@dataclass(frozen=True)
class NewsQuery:
    request: LookupRequest
    topic: list[str]  # lower-case words every headline must contain; never sent
    label: str  # how to describe the feed ("business"), or "" for general news
    topic_text: str  # the topic as the user said it ("TCS")


def news_request(text: str) -> NewsQuery:
    """Feeds to download, and topic words to filter by locally (never sent)."""
    words = set(tokenize(text))
    feeds = [feed for feed, keywords in FEED_WORDS.items() if words & keywords]
    match = TOPIC.search(text)
    spoken = re.findall(r"[A-Za-z0-9']+", match.group(1)) if match else []
    kept = [word for word in spoken if (w := word.casefold()) not in STOPWORDS and w not in FILLER]
    topic = [word.casefold() for word in kept]
    if topic:  # a topic can be in any feed; it's matched here, so fetching all costs no privacy
        return NewsQuery(LookupRequest("news", feeds=tuple(FEEDS)), topic, "", " ".join(kept))
    return NewsQuery(LookupRequest("news", feeds=tuple(feeds or DEFAULT_FEEDS)), [], " and ".join(feeds), "")


def _mentions(headline: Headline, topic: list[str]) -> bool:
    """Every topic word must appear (plurals allowed), so "cricket team zzqx" doesn't
    match any headline that merely mentions a team."""
    title = headline.title.casefold()
    return all(re.search(rf"\b{re.escape(word)}(?:s|es)?\b", title) for word in topic)


def _newest_first(headlines: list[Headline]) -> list[Headline]:
    def when(headline: Headline) -> float:
        try:
            return parsedate_to_datetime(headline.published).timestamp()
        except (TypeError, ValueError):
            return 0.0

    return sorted(headlines, key=when, reverse=True)


def headlines_reply(result: LookupResult, query: NewsQuery, count: int = 3) -> str:
    headlines = [h for h in result.data if not query.topic or _mentions(h, query.topic)]
    stale = f" These are from {result.stale_since}; I couldn't refresh the feeds." if result.stale_since else ""
    if not headlines:
        return f"I found no headlines about {query.topic_text} in the latest news from {result.source}.{stale}"
    picked = _newest_first(headlines)[:count]
    kind = f"{query.label} " if query.label else ""
    about = f" about {query.topic_text}" if query.topic else ""
    sources = " and ".join(dict.fromkeys(h.source for h in picked))
    items = " ".join(f"{ORDINALS[i]}: {h.title.rstrip('.')}." for i, h in enumerate(picked))
    return f"From {sources}, online: the latest {kind}headlines{about}. {items}{stale}"
