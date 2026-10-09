"""Conversation memory: notes the user asks to keep, and past questions and answers.

Everything is stored as text in a local SQLite file (FTS5, Porter stemming),
never audio, and never leaves the device. Entries older than the retention
period are deleted at startup; "forget that" and "forget everything" delete on
request.

Search is hybrid when an embedding model is available (see brain/embeddings.py):
a stored item is a match if it contains every meaningful word of the question
(keyword search with stemming and synonyms), or if its embedding is similar
enough (cosine >= min_similarity), which finds paraphrases like "power tool"
for "drill". Matches found only by meaning are marked `exact=False`, so the
assistant hedges ("the closest thing I remember...") instead of sounding sure.
Without a model, search falls back to keywords only.

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
from typing import Callable, Protocol

import numpy as np

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


class TextEmbedder(Protocol):
    def embed(self, texts: list[str], query: bool = False) -> np.ndarray: ...


@dataclass(frozen=True)
class MemoryItem:
    id: int
    when: datetime
    kind: str  # "note" | "turn"
    question: str
    answer: str
    exact: bool = True  # False when found only by meaning, not by its words

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
        embedder: TextEmbedder | None = None,
        min_similarity: float = 0.35,
    ):
        """`path=None` keeps memory in RAM only (tests, or MEMORY=0 persistence off)."""
        self.follow_up_window = timedelta(minutes=follow_up_minutes)
        self.top_k = top_k
        self.embedder = embedder
        self.min_similarity = min_similarity
        self._clock = clock
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(":memory:" if path is None else str(path))
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory USING fts5("
            "created UNINDEXED, kind UNINDEXED, question, answer, tokenize='porter unicode61')"
        )
        # One embedding per memory row (float32 bytes), keyed by the memory rowid.
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, vec BLOB NOT NULL)")
        cutoff = (self._clock() - timedelta(days=retention_days)).isoformat(timespec="seconds")
        self._db.execute("DELETE FROM memory WHERE created < ?", (cutoff,))
        self._db.execute("DELETE FROM vectors WHERE id NOT IN (SELECT rowid FROM memory)")
        self._db.commit()
        self._embed_missing()

    @staticmethod
    def _text_of(question: str, answer: str) -> str:
        return f"{question} {answer}".strip()

    def _embed_missing(self) -> None:
        """Embed rows stored before embeddings were enabled (or with another model)."""
        if self.embedder is None:
            return
        rows = self._db.execute(
            "SELECT rowid, question, answer FROM memory WHERE rowid NOT IN (SELECT id FROM vectors)"
        ).fetchall()
        if rows:
            vectors = self.embedder.embed([self._text_of(q, a) for _, q, a in rows])
            self._db.executemany(
                "INSERT INTO vectors (id, vec) VALUES (?, ?)",
                [(row[0], vec.astype(np.float32).tobytes()) for row, vec in zip(rows, vectors)],
            )
            self._db.commit()

    def _add(self, kind: str, question: str, answer: str = "") -> None:
        cursor = self._db.execute(
            "INSERT INTO memory (created, kind, question, answer) VALUES (?, ?, ?, ?)",
            (self._clock().isoformat(timespec="seconds"), kind, question, answer),
        )
        if self.embedder is not None:
            vector = self.embedder.embed([self._text_of(question, answer)])[0]
            self._db.execute(
                "INSERT INTO vectors (id, vec) VALUES (?, ?)", (cursor.lastrowid, vector.astype(np.float32).tobytes())
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
            self._db.execute("DELETE FROM vectors WHERE id = ?", (items[0].id,))
            self._db.commit()
        return items[0] if items else None

    def forget_all(self) -> int:
        count = self._db.execute("SELECT count(*) FROM memory").fetchone()[0]
        self._db.execute("DELETE FROM memory")
        self._db.execute("DELETE FROM vectors")
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

    def _meaningful_words(self, question: str) -> list[str]:
        return [w for w in dict.fromkeys(tokenize(question)) if w not in STOPWORDS and w not in RECALL_WORDS]

    def _scope(self, question: str) -> tuple[str, list]:
        """SQL filter: notes always, turns for recall questions, a day if one is named."""
        kinds = ("note", "turn") if is_recall_question(question) else ("note",)
        clause, params = f"kind IN ({','.join('?' * len(kinds))})", list(kinds)
        day = self._day_filter(question)
        if day is not None:
            clause += " AND substr(created, 1, 10) = ?"
            params.append(day)
        return clause, params

    def search(self, question: str) -> list[MemoryItem]:
        """Best matching items: exact keyword matches first, then by similarity."""
        clause, params = self._scope(question)
        words = self._meaningful_words(question)
        if not words:
            if self._day_filter(question) is None:
                return []
            # "What did I ask you yesterday?": everything from that day, newest first.
            return self._select(f"WHERE {clause} ORDER BY rowid DESC LIMIT ?", (*params, self.top_k))

        terms = list(dict.fromkeys(words + [extra for word in words for extra in synonyms_for(word)]))
        match = "{question answer} : (" + " OR ".join(f'"{term}"' for term in terms) + ")"
        keyword_hits = self._select(f"WHERE {clause} AND memory MATCH ? ORDER BY rank", (*params, match))
        full = self._full_matches(words, [item.id for item in keyword_hits])

        if self.embedder is None:  # keyword-only: any keyword hit, best ranked first
            return [_with_exact(item, item.id in full) for item in keyword_hits[: self.top_k]]

        candidates = {item.id: item for item in self._select(f"WHERE {clause}", tuple(params))}
        if not candidates:
            return []
        vectors = dict(
            self._db.execute(
                f"SELECT id, vec FROM vectors WHERE id IN ({','.join('?' * len(candidates))})", tuple(candidates)
            ).fetchall()
        )
        query = self.embedder.embed([question], query=True)[0]
        similarity = {
            rowid: float(np.frombuffer(vectors[rowid], dtype=np.float32) @ query) if rowid in vectors else 0.0
            for rowid in candidates
        }
        accepted = [rowid for rowid in candidates if rowid in full or similarity[rowid] >= self.min_similarity]
        accepted.sort(key=lambda rowid: (rowid in full, similarity[rowid]), reverse=True)
        return [_with_exact(candidates[rowid], rowid in full) for rowid in accepted[: self.top_k]]

    def _full_matches(self, words: list[str], rowids: list[int]) -> set[int]:
        """Rows containing every meaningful word of the question (or a synonym)."""
        full = set()
        for rowid in rowids:
            if all(
                self._db.execute(
                    "SELECT 1 FROM memory WHERE rowid = ? AND memory MATCH ?",
                    (rowid, "{question answer} : (" + " OR ".join(f'"{t}"' for t in [w, *synonyms_for(w)]) + ")"),
                ).fetchone()
                for w in words
            ):
                full.add(rowid)
        return full

    def coverage(self, question: str, kinds: tuple[str, ...] = ("note",)) -> dict[str, str]:
        """For each stored text: does it contain "all", "some" or "none" of the question's
        meaningful words? Used by scripts/eval_memory_retrieval.py."""
        words = self._meaningful_words(question)
        placeholders = ",".join("?" * len(kinds))
        rows = self._db.execute(f"SELECT rowid, question FROM memory WHERE kind IN ({placeholders})", kinds).fetchall()
        result = {}
        for rowid, text in rows:
            hits = sum(
                1
                for w in words
                if self._db.execute(
                    "SELECT 1 FROM memory WHERE rowid = ? AND memory MATCH ?",
                    (rowid, "{question answer} : (" + " OR ".join(f'"{t}"' for t in [w, *synonyms_for(w)]) + ")"),
                ).fetchone()
            )
            result[text] = "none" if hits == 0 else ("all" if hits == len(words) else "some")
        return result

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


def _with_exact(item: MemoryItem, exact: bool) -> MemoryItem:
    return MemoryItem(item.id, item.when, item.kind, item.question, item.answer, exact)


def describe(items: list[MemoryItem]) -> str:
    return "\n".join(item.describe() for item in items)


def note_answer(note: MemoryItem) -> str:
    """Answer straight from a note, saying when it was given. A note found only by
    meaning may be a near-miss ("wife's birthday" vs "mom's birthday"), so hedge."""
    said = f"on {note.when.day} {note.when:%B} you told me that {second_person(note.question)}."
    if note.exact:
        return said[0].upper() + said[1:]
    return f"I'm not certain, but the closest thing I remember is: {said}"
