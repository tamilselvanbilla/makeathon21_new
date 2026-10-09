"""Search the private JSON knowledge base for records relevant to a question.

Records are indexed in an in-memory SQLite FTS5 table with Porter stemming, so
"loans" finds "Home Loan" and "expires" finds "expiry". Results are ranked by
BM25 and limited to the top few, which keeps prompts short on a Pi.

When the question names a record type ("passport", "loan", "income"), only
records of that type are returned. A small model otherwise borrows facts from
neighbouring records, e.g. answering a passport expiry with an insurance date.

Records belong to an owner. Unless a question names another person ("my
wife's income", "Robert's ID"), only the primary user's records and shared
household records (owners containing "family") are returned, so one person's
data is never presented as another's.
"""

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
KNOWLEDGE_FILE = KNOWLEDGE_DIR / "personal_data.json"
CATEGORIES = ("financial", "medical", "documents", "history")
SHARED_OWNER_WORDS = frozenset({"family", "household"})
# Category names are too broad to narrow results to one record type.
CATEGORY_WORDS = frozenset({"financial", "finance", "medical", "document", "documents", "history"})
PERSONAL_WORDS = frozenset({"i", "me", "my", "mine", "myself", "our", "we"})
STOPWORDS = frozenset(
    """
    a about all an and any are as at be can could did do does for from give has
    have how i in info information is it its list me mine my myself need of on
    or our please record records show tell the their there this to us was we
    what when where which who whom why will with you your much many latest
    detail details know get got not no
    """.split()
)
# Everyday words mapped to the vocabulary used in the records, so "earn"
# finds income and "EMI" finds the loan installment. Stemming covers plurals.
SYNONYMS = {
    "earn": "income salary", "earning": "income salary", "salary": "income",
    "pay": "income salary", "paid": "income salary",
    "spend": "expense", "spending": "expense",
    "emi": "installment loan", "mortgage": "home loan",
    "invest": "investment mutual fund stock bond", "investment": "mutual fund stock bond",
    "expire": "expiry renewal valid maturity", "expiry": "renewal valid maturity",
    "renew": "renewal", "due": "renewal maturity",
    "work": "organization employer role", "job": "organization employer role",
    "company": "organization employer", "employer": "organization",
    "before": "previous", "previously": "previous", "earlier": "previous",
    "car": "four wheeler vehicle", "bike": "two wheeler vehicle", "scooter": "two wheeler vehicle",
    "bp": "blood pressure", "medicine": "medication prescription", "tablet": "medication prescription",
    "study": "school college qualification", "studied": "school college qualification",
    "education": "school college qualification",
    "id": "identification document", "identity": "identification document",
    "finish": "years period", "finished": "years period", "graduate": "years college qualification",
    "graduated": "years college qualification", "join": "period years", "joined": "period years",
}

Records = dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class Match:
    category: str
    owner: str
    text: str


def load_knowledge(path: Path = KNOWLEDGE_FILE) -> Records:
    """Read the configured knowledge base and return typed records."""
    with path.open(encoding="utf-8") as source:
        records = json.load(source)

    return {
        category: records.get(category, [])
        if isinstance(records.get(category, []), list)
        else []
        for category in CATEGORIES
    }


def _label(key: str) -> str:
    return str(key).replace("_", " ").strip().title()


def _format_record(entry: dict[str, Any]) -> str:
    """Convert a knowledge record into readable, non-JSON text."""
    parts: list[str] = []
    for key, value in entry.items():
        if key in {"owner", "source"} or value is None or value == "":
            continue
        if key == "record_type":
            value = str(value).replace("_", " ")
        if isinstance(value, dict):
            nested = ", ".join(f"{_label(k)}: {v}" for k, v in value.items())
            parts.append(f"{_label(key)}: {nested}")
        elif isinstance(value, list):
            parts.append(f"{_label(key)}: {', '.join(str(item) for item in value)}")
        else:
            parts.append(f"{_label(key)}: {value}")
    return "; ".join(parts)


def synonyms_for(word: str) -> list[str]:
    """Synonym expansion that also accepts simple plurals ("investments")."""
    found = SYNONYMS.get(word) or (SYNONYMS.get(word[:-1]) if word.endswith("s") else None)
    return found.split() if found else []


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with possessive 's removed ("wife's" -> "wife")."""
    return [re.sub(r"'s$", "", word) for word in re.findall(r"[a-z0-9]+(?:'s)?", text.casefold())]


class KnowledgeBase:
    def __init__(
        self,
        records: Records,
        primary_user: str | None = None,
        top_k: int = 4,
        currency: str = "INR",
    ):
        self.top_k = top_k
        self.currency = currency
        owners = Counter(
            str(entry.get("owner") or "")
            for entries in records.values()
            for entry in entries
            if isinstance(entry, dict)
        )
        owners.pop("", None)
        self.primary_user = primary_user or (owners.most_common(1)[0][0] if owners else "the user")
        self._aliases = self._owner_aliases(records, owners)
        self._db = self._build_index(records)

    def _owner_aliases(self, records: Records, owners: Counter) -> dict[str, set[str]]:
        """Words that refer to each non-primary owner: "John's wife" -> {wife, jane}."""
        primary_words = set(tokenize(self.primary_user))
        aliases = {
            owner: set(tokenize(owner)) - primary_words
            for owner in owners
            if owner != self.primary_user
        }
        for entries in records.values():
            for entry in entries:
                owner = str(entry.get("owner") or "")
                if owner in aliases and entry.get("name") and entry.get("relationship"):
                    aliases[owner] |= set(tokenize(str(entry["name"])))
        return aliases

    def _build_index(self, records: Records) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:")
        try:
            db.execute(
                "CREATE VIRTUAL TABLE records USING fts5("
                "category UNINDEXED, owner UNINDEXED, text UNINDEXED, kind, body, "
                "tokenize='porter unicode61')"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("This Python's SQLite lacks FTS5, which knowledge search needs.") from exc

        for category, entries in records.items():
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                text = _format_record(entry)
                kind = str(entry.get("record_type") or "").replace("_", " ")
                db.execute(
                    "INSERT INTO records (category, owner, text, kind, body) VALUES (?, ?, ?, ?, ?)",
                    (category, str(entry.get("owner") or self.primary_user), text, kind, f"{category} {text}"),
                )
        return db

    def mentioned_owners(self, question: str) -> set[str]:
        words = set(tokenize(question))
        return {owner for owner, alias in self._aliases.items() if words & alias}

    def allowed_owners(self, question: str) -> set[str]:
        mentioned = self.mentioned_owners(question)
        if any(set(tokenize(owner)) & SHARED_OWNER_WORDS for owner in mentioned):
            return {self.primary_user, *self._aliases}  # "my family" covers everyone
        if mentioned:
            return mentioned
        shared = {owner for owner in self._aliases if set(tokenize(owner)) & SHARED_OWNER_WORDS}
        return {self.primary_user, *shared}

    def is_personal(self, question: str) -> bool:
        """True when the question is about the user's (or their family's) own data."""
        return bool(set(tokenize(question)) & PERSONAL_WORDS) or bool(self.mentioned_owners(question))

    def _terms(self, question: str) -> tuple[list[str], list[str]]:
        """Search terms (with synonyms) and the words that were expanded."""
        words = [word for word in tokenize(question) if word not in STOPWORDS]
        expanded = [word for word in words if synonyms_for(word)]
        terms = words + [extra for word in expanded for extra in synonyms_for(word)]
        return list(dict.fromkeys(terms)), list(dict.fromkeys(expanded))

    def search(self, question: str) -> list[Match]:
        terms, _ = self._terms(question)
        if not terms:
            return []
        allowed = self.allowed_owners(question)
        # Each term is quoted, so user words can never act as FTS5 operators.
        query = " OR ".join(f'"{term}"' for term in terms)
        hits = [
            row
            for row in self._db.execute(
                "SELECT rowid, category, owner, text FROM records WHERE records MATCH ? ORDER BY rank",
                (f"body : ({query})",),
            )
            if row[2] in allowed
        ]

        focus_terms = [term for term in terms if term not in CATEGORY_WORDS]
        if focus_terms:
            focus_query = " OR ".join(f'"{term}"' for term in focus_terms)
            same_kind = {
                rowid
                for (rowid,) in self._db.execute(
                    "SELECT rowid FROM records WHERE records MATCH ?", (f"kind : ({focus_query})",)
                )
            }
            focused = [row for row in hits if row[0] in same_kind]
            if focused:
                hits = focused

        if hits and not self._covers_asked_attributes(question, [row[0] for row in hits]):
            # e.g. "when does my passport expire" when the passport record has no
            # expiry: return nothing so the assistant says so instead of guessing.
            return []
        return [Match(*row[1:]) for row in hits[: self.top_k]]

    def _covers_asked_attributes(self, question: str, rowids: list[int]) -> bool:
        """True if every meaningful word of the question (or a synonym) appears in
        at least one selected record: "where is my car key" must not be answered
        from a car record that never mentions a key."""
        owner_words = {word for alias in self._aliases.values() for word in alias}
        words = [
            word
            for word in dict.fromkeys(tokenize(question))
            if word not in STOPWORDS and word not in CATEGORY_WORDS and word not in owner_words
            and word not in PERSONAL_WORDS and word not in tokenize(self.primary_user)
        ]
        placeholders = ",".join("?" * len(rowids))
        for word in words:
            concept = " OR ".join(f'"{term}"' for term in [word, *synonyms_for(word)])
            found = self._db.execute(
                f"SELECT 1 FROM records WHERE records MATCH ? AND rowid IN ({placeholders}) LIMIT 1",
                (f"body : ({concept})", *rowids),
            ).fetchone()
            if not found:
                return False
        return True

    def context_for(self, question: str) -> str:
        """Readable, owner-labelled records for the prompt, or "" when none match."""
        matches = self.search(question)
        if not matches:
            return ""
        lines = [f"{match.category.title()} record of {match.owner}: {match.text}" for match in matches]
        _, expanded = self._terms(question)
        if expanded:
            # Tell the model how the user's words map to the record fields ("EMI" -> installment).
            notes = "; ".join(f"'{word}' refers to {', '.join(synonyms_for(word))}" for word in expanded)
            lines.append(f"Note: {notes}.")
        return "\n".join(lines)
