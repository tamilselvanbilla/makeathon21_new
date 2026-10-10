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
# Fields whose values name things (see KnowledgeBase._entities).
ENTITY_FIELDS = frozenset({"name", "company", "employer", "issuer", "bank", "scheme", "provider", "model", "location"})
# Without these, a question is about the present: "where do I work" means the
# current job, so records of a "previous ..." type rank last.
PAST_WORDS = frozenset({"before", "previous", "previously", "earlier", "past", "former", "used", "old"})
CATEGORY_WORDS = frozenset({"financial", "finance", "medical", "document", "documents", "history"})
PERSONAL_WORDS = frozenset({"i", "me", "my", "mine", "myself", "our", "we"})
STOPWORDS = frozenset(
    """
    a about all an and any are as at be can could did do does for from give has
    have how i in info information is it its list me mine my myself need of on
    or our please record records show tell the their there this to us was we
    what when where which who whom why will with you your much many latest
    detail details know get got not no
    am been being take takes taking took daily every each last now currently
    say said says written put hold holding own owned owns go goes went make made
    keep kept use used using long normal level like right still just also ever
    into under over onto per than then her his him she he they them pay
    related regarding question questions answer summarize summarise summary overview
    amount should would
    """.split()
)
# Everyday words mapped to the vocabulary used in the records, so "earn"
# finds income and "EMI" finds the loan installment. Stemming covers plurals.
SYNONYMS = {
    "earn": "income salary", "earning": "income salary", "salary": "income",
    "paid": "income salary",
    "spend": "expense", "spending": "expense",
    "emi": "installment loan", "mortgage": "home loan",
    "invest": "investment mutual fund stock bond", "investment": "mutual fund stock bond",
    "expire": "expiry renewal valid maturity", "expiry": "renewal valid maturity",
    "renew": "renewal", "due": "renewal maturity",
    "work": "organization employer role", "job": "organization employer role",
    "employer": "organization",
    "before": "previous", "previously": "previous", "earlier": "previous",
    "car": "four wheeler vehicle", "bike": "two wheeler vehicle", "scooter": "two wheeler vehicle",
    "bp": "blood pressure", "medicine": "prescription dosage", "tablet": "medication prescription",
    "study": "school college qualification", "studied": "school college qualification",
    "education": "school college qualification",
    "id": "identification document", "identity": "identification document",
    "finish": "years period", "finished": "years period", "graduate": "years college qualification",
    "graduated": "years college qualification", "join": "period years", "joined": "period years",
    # Spoken paraphrases of record fields, found with the retrieval benchmark
    # (tests/data/knowledge_retrieval_eval.json).
    "medication": "prescription dosage", "medicines": "prescription dosage",
    "pill": "prescription medication dosage", "drug": "prescription medication dosage",
    "weigh": "weight kg", "heavy": "weight kg", "tall": "height cm",
    "oxygen": "oxygen saturation", "pulse": "pulse per minute", "heart": "pulse per minute",
    "test": "report medical", "checkup": "medical history report review", "health": "medical history review vitals bmi",
    "rent": "housing expense", "groceries": "food expense", "month": "monthly",
    "left": "remaining", "remaining": "remaining tenure",
    "validity": "valid from to renewal expiry", "valid": "valid from to renewal expiry",
    "period": "valid from to renewal expiry period", "end": "valid to expiry renewal maturity",
    "ends": "valid to expiry renewal maturity",
    "cover": "coverage amount insurance", "covers": "coverage amount insurance",
    "insure": "insurance coverage provider", "insures": "insurance coverage provider",
    "insured": "insurance coverage amount", "policy": "policy insurance",
    "share": "stock quantity", "shares": "stock quantity", "licence": "license",
    "title": "role", "designation": "role", "position": "role", "office": "organization",
    "prescribed": "prescription", "prescribe": "prescription", "qualify": "qualification",
    "qualified": "qualification", "till": "valid to expiry", "until": "valid to expiry",
    "active": "valid from to", "assured": "coverage amount", "sum": "coverage amount",
    "drive": "vehicle", "driving": "vehicle license", "insurer": "insurance provider",
    "company": "organization employer provider", "complete": "years period", "completed": "years period",
    "rate": "rate minute percent",
    # Career questions: the records hold organizations, roles and periods.
    "career": "organization role", "experience": "organization role period",
    "experienced": "organization role period", "employment": "organization role employer",
    "employed": "organization role employer", "profession": "role organization",
    "professional": "role organization", "occupation": "role organization",
    "worked": "organization role", "working": "organization role",
    "resume": "organization role school college qualification", "cv": "organization role school college qualification",
    "background": "organization role school college qualification",
    "year": "years period",
    "lose": "weight bmi", "fat": "weight bmi", "obese": "bmi overweight",
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


def _acronyms(entry: dict[str, Any]) -> list[str]:
    """Initials of names of three or more capitalised words, so "SBI" finds
    "State Bank of India" and "TCS" finds "Tata Consultancy Services"."""
    found = []
    for value in entry.values():
        if isinstance(value, str):
            capitals = [w for w in re.findall(r"[A-Za-z]+", value) if w[0].isupper() and w.lower() not in ("of", "and", "the")]
            if len(capitals) >= 3 and len(capitals) == len([w for w in re.findall(r"[A-Za-z]+", value) if w.lower() not in ("of", "and", "the")]):
                found.append("".join(w[0] for w in capitals))
    return found


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with possessive 's removed ("wife's" -> "wife") and
    dotted acronyms joined ("E.M.I." -> "emi", as speech-to-text may write them)."""
    text = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group().replace(".", ""), text.casefold())
    return [re.sub(r"'s$", "", word) for word in re.findall(r"[a-z0-9]+(?:'s)?", text)]


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
        # "wife Jane", "son Robert": named family members, for the system prompt.
        self.family = [
            f"{str(entry['relationship']).casefold()} {entry['name']}"
            for entries in records.values()
            for entry in entries
            if isinstance(entry, dict) and entry.get("name") and entry.get("relationship")
            and entry.get("owner") != self.primary_user
        ]
        self._db = self._build_index(records)
        # Words naming things (companies, banks, models, places) and their acronyms. In a
        # question they may point at another record ("where did I work before Infosys?"),
        # so they don't have to appear in the answer record itself.
        self._entities = {
            word
            for entries in records.values()
            for entry in entries
            if isinstance(entry, dict)
            for key, value in entry.items()
            if isinstance(value, str) and key in ENTITY_FIELDS
            for word in tokenize(value) + [a.casefold() for a in _acronyms({key: value})]
        }
        # ... but not common nouns that are also record types ("college", "bank", "fund").
        kind_words = {w for (kind,) in self._db.execute("SELECT kind FROM records") for w in tokenize(kind)}
        self._entities -= kind_words | STOPWORDS

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
                "category UNINDEXED, owner UNINDEXED, text UNINDEXED, kind, topic, body, "
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
                acronyms = " ".join(_acronyms(entry))
                db.execute(
                    # The category is its own column: it helps find records ("my medical
                    # records") but must not satisfy a question word by itself, or
                    # "medications" (stem "medic") would match every medical record.
                    "INSERT INTO records (category, owner, text, kind, topic, body) VALUES (?, ?, ?, ?, ?, ?)",
                    (category, str(entry.get("owner") or self.primary_user), text, kind, category, f"{text} {acronyms}"),
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
                (f"{{body topic}} : ({query})",),
            )
            if row[2] in allowed
        ]

        # A record is relevant only if it covers every meaningful word of the question by
        # itself (the word or a synonym). Words spread over several records mean the
        # answer isn't recorded: "when does my passport expire" must not combine the
        # passport record with an insurance "valid to" date. Ranking: records of a
        # "previous ..." type last unless the question asks about the past, then BM25.
        concepts = self._concepts(question)
        names_found: dict[int, int] = {}
        if concepts and hits:
            rowids = [row[0] for row in hits]
            required, named = [], []
            for word, concept in concepts:
                (named if word in self._entities else required).append(concept)
            names_found = self._concepts_covered(named, rowids)
            if all(self._any_covers(concept, rowids) for concept in named):
                covered = self._concepts_covered(required, rowids)
                hits = [row for row in hits if covered[row[0]] == len(required)]
            else:
                hits = []
        if hits:
            past = bool(set(tokenize(question)) & PAST_WORDS)
            kinds = dict(self._db.execute("SELECT rowid, kind FROM records").fetchall())
            order = {row[0]: i for i, row in enumerate(hits)}
            # Records containing a named thing from the question come first ("my Hero
            # Splendor", "at TCS"), then current before previous, then BM25.
            hits.sort(
                key=lambda row: (
                    -names_found.get(row[0], 0),
                    not past and kinds[row[0]].startswith("previous"),
                    order[row[0]],
                )
            )
        return [Match(*row[1:]) for row in hits[: self.top_k]]

    def names_topic(self, question: str) -> bool:
        """True when the question asks about something itself ("my EMI"), not only about
        a person ("and my wife's?"), and records on it exist."""
        return bool(self._concepts(question)) and bool(self.search(question))

    def _concepts(self, question: str) -> list[tuple[str, str]]:
        """(word, FTS query) per meaningful question word; the query matches the word or a synonym."""
        owner_words = {word for alias in self._aliases.values() for word in alias}
        words = [
            word
            for word in dict.fromkeys(tokenize(question))
            if word not in STOPWORDS and word not in CATEGORY_WORDS and word not in owner_words
            and word not in PERSONAL_WORDS and word not in tokenize(self.primary_user)
        ]
        def terms(word: str) -> list[str]:
            # A word whose stem collides with a category name ("medications" and "medical"
            # both stem to "medic") is matched through its synonyms only, or it would
            # match every record of that category.
            synonyms = synonyms_for(word)
            collides = any(word[:5] == category[:5] for category in CATEGORY_WORDS) and word not in CATEGORY_WORDS
            return synonyms if synonyms and collides else [word, *synonyms]

        return [(w, "body : (" + " OR ".join(f'"{t}"' for t in terms(w)) + ")") for w in words]

    def _any_covers(self, concept: str, rowids: list[int]) -> bool:
        placeholders = ",".join("?" * len(rowids))
        return bool(self._db.execute(
            f"SELECT 1 FROM records WHERE records MATCH ? AND rowid IN ({placeholders}) LIMIT 1", (concept, *rowids)
        ).fetchone())

    def _concepts_covered(self, concepts: list[str], rowids: list[int]) -> dict[int, int]:
        counts = dict.fromkeys(rowids, 0)
        placeholders = ",".join("?" * len(rowids))
        for concept in concepts:
            for (rowid,) in self._db.execute(
                f"SELECT rowid FROM records WHERE records MATCH ? AND rowid IN ({placeholders})", (concept, *rowids)
            ):
                counts[rowid] += 1
        return counts

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
