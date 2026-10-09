"""Conversation memory: notes the user asks to keep, and past questions and answers.

Everything is stored as text in a local SQLite file (FTS5, Porter stemming),
never audio, and never leaves the device. Entries older than the retention
period are deleted at startup; "forget that" and "forget everything" delete on
request.

Three uses:
- Notes ("remember that I parked on level B2") are searched for every question.
- Past turns are searched only for recall questions ("what did you tell me
  about my EMI yesterday?"), so old answers don't distract ordinary ones.
- The latest turns resolve follow-ups ("and my wife's?") within a few minutes.
"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .knowledge import STOPWORDS, synonyms_for, tokenize

# First person -> second person, for notes and confirmations ("I parked" -> "you parked").
PRONOUNS = {"i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself", "i'm": "you're", "am": "are"}

# "remember (that) ...", "note that ...", "make a note (that) ..."
REMEMBER = re.compile(r"^\W*(?:please\s+)?(?:remember|note|make a note|keep in mind)\b(?:\s+that)?[\s,:]*(.+)$", re.I)
FORGET_ALL = re.compile(
    r"^\W*(?:please\s+)?(?:forget everything|clear (?:your|all|the) memor(?:y|ies)|"
    r"delete (?:all|every) memor(?:y|ies)|erase (?:your|all) memor(?:y|ies))\W*$",
    re.I,
)
FORGET_LAST = re.compile(r"^\W*(?:please\s+)?(?:forget (?:that|it|the last (?:thing|one|note))|delete that)\W*$", re.I)
# Questions about the past conversation, not about the records.
RECALL = re.compile(
    r"\b(?:did i|did you|have i|had i|earlier|last time|previous(?:ly)?|yesterday|"
    r"remember|told|tell me again|said|asked|mentioned)\b",
    re.I,
)
# Words that only mark a question as recall; they never appear in what was said.
RECALL_WORDS = frozenset(
    "ask asked tell told say said mention mentioned remember earlier yesterday today last time "
    "previous previously again did".split()
)
# Short questions that only make sense after the previous one.
FOLLOW_UP = re.compile(r"^\W*(?:and|what about|how about|also|same for|what of|then)\b", re.I)


@dataclass(frozen=True)
class MemoryItem:
    id: int
    when: datetime
    kind: str  # "note" | "turn"
    question: str
    answer: str

    def describe(self) -> str:
        day = self.when.strftime("%d %b %Y, %H:%M")
        if self.kind == "note":
            return f"Note from {day}: {second_person(self.question)}."
        return f"On {day} the user asked: {self.question} You answered: {self.answer}"


@dataclass(frozen=True)
class MemoryCommand:
    action: str  # "remember" | "forget_last" | "forget_all"
    text: str = ""


def parse_memory_command(text: str) -> MemoryCommand | None:
    if FORGET_ALL.match(text):
        return MemoryCommand("forget_all")
    if FORGET_LAST.match(text):
        return MemoryCommand("forget_last")
    match = REMEMBER.match(text)
    if match and match.group(1).strip(" .!?"):
        return MemoryCommand("remember", match.group(1).strip(" .!?"))
    return None


def second_person(text: str) -> str:
    """Echo the user's words back: "I parked on B2" -> "you parked on B2"."""
    return re.sub(
        r"\b(?:I'm|I|me|my|mine|myself|am)\b",
        lambda match: PRONOUNS[match.group(0).casefold()],
        text.strip(" .!?"),
        flags=re.IGNORECASE,
    )


def is_recall_question(text: str) -> bool:
    return bool(RECALL.search(text))


def is_follow_up(text: str) -> bool:
    return bool(FOLLOW_UP.match(text))


class ConversationMemory:
    def __init__(
        self,
        path: Path | None = None,
        retention_days: int = 30,
        follow_up_minutes: float = 10,
        top_k: int = 3,
        clock: Callable[[], datetime] = datetime.now,
    ):
        """`path=None` keeps memory in RAM only (tests, or MEMORY=0 persistence off)."""
        self.follow_up_window = timedelta(minutes=follow_up_minutes)
        self.top_k = top_k
        self._clock = clock
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(":memory:" if path is None else str(path))
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory USING fts5("
            "created UNINDEXED, kind UNINDEXED, question, answer, tokenize='porter unicode61')"
        )
        cutoff = (self._clock() - timedelta(days=retention_days)).isoformat(timespec="seconds")
        self._db.execute("DELETE FROM memory WHERE created < ?", (cutoff,))
        self._db.commit()

    def _add(self, kind: str, question: str, answer: str = "") -> None:
        self._db.execute(
            "INSERT INTO memory (created, kind, question, answer) VALUES (?, ?, ?, ?)",
            (self._clock().isoformat(timespec="seconds"), kind, question, answer),
        )
        self._db.commit()

    def add_note(self, text: str) -> None:
        self._add("note", text)

    def add_turn(self, question: str, answer: str) -> None:
        self._add("turn", question, answer)

    def forget_last(self) -> MemoryItem | None:
        items = self._select("ORDER BY rowid DESC LIMIT 1")
        if items:
            self._db.execute("DELETE FROM memory WHERE rowid = ?", (items[0].id,))
            self._db.commit()
        return items[0] if items else None

    def forget_all(self) -> int:
        count = self._db.execute("SELECT count(*) FROM memory").fetchone()[0]
        self._db.execute("DELETE FROM memory")
        self._db.commit()
        return count

    def _select(self, clause: str, params: tuple = ()) -> list[MemoryItem]:
        rows = self._db.execute(
            f"SELECT rowid, created, kind, question, answer FROM memory {clause}", params
        ).fetchall()
        return [MemoryItem(row[0], datetime.fromisoformat(row[1]), *row[2:]) for row in rows]

    def last_turn(self) -> MemoryItem | None:
        """The previous question and answer, if recent enough for a follow-up."""
        since = (self._clock() - self.follow_up_window).isoformat(timespec="seconds")
        turns = self._select("WHERE kind = 'turn' AND created >= ? ORDER BY rowid DESC LIMIT 1", (since,))
        return turns[0] if turns else None

    def search(self, question: str) -> list[MemoryItem]:
        """Notes always; past turns only for recall questions. Best matches first."""
        words = [word for word in tokenize(question) if word not in STOPWORDS and word not in RECALL_WORDS]
        terms = list(dict.fromkeys(words + [extra for word in words for extra in synonyms_for(word)]))
        kinds = ("note", "turn") if is_recall_question(question) else ("note",)
        clause, params = f"WHERE kind IN ({','.join('?' * len(kinds))})", list(kinds)

        day = self._day_filter(question)
        if day is not None:
            clause += " AND substr(created, 1, 10) = ?"
            params.append(day)
        if terms:
            clause += " AND memory MATCH ?"
            params.append("{question answer} : (" + " OR ".join(f'"{term}"' for term in terms) + ")")
            return self._select(f"{clause} ORDER BY rank LIMIT ?", (*params, self.top_k))
        if day is not None:  # "what did I ask you today?"
            return self._select(f"{clause} ORDER BY rowid DESC LIMIT ?", (*params, self.top_k))
        return []

    def _day_filter(self, question: str) -> str | None:
        lowered = question.casefold()
        if re.search(r"\byesterday\b", lowered):
            return (self._clock() - timedelta(days=1)).date().isoformat()
        if re.search(r"\btoday\b", lowered) and is_recall_question(re.sub(r"\btoday\b", "", lowered)):
            return self._clock().date().isoformat()
        return None

    def context_for(self, question: str) -> str:
        """Readable memory lines for the prompt, or "" when nothing relevant."""
        return describe(self.search(question))


def describe(items: list[MemoryItem]) -> str:
    return "\n".join(item.describe() for item in items)


def note_answer(note: MemoryItem) -> str:
    """Answer straight from a note, saying when it was given."""
    return f"On {note.when.day} {note.when:%B} you told me that {second_person(note.question)}."
