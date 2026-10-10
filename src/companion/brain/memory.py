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

A note that names a date or time ("remember I have a meeting with Jay tomorrow
at 7am") is stored with the date resolved ("... on Saturday 10 October at 7:00
AM") and also goes in the `schedule` table, so "what's my schedule tomorrow?" is
answered from it and the reminder is announced when it is due.

Notes are kept until the user deletes them; past turns expire after the
retention period.

Three uses:
- Notes ("remember that I parked on level B2") are searched for every question.
- Past turns are searched only for recall questions ("what did you tell me
  about my EMI yesterday?"), so old answers don't distract ordinary ones.
- The latest turns resolve follow-ups ("and my wife's?") within a few minutes.
"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .knowledge import STOPWORDS, synonyms_for, tokenize
from .schedule import When, parse_when, spoken_day, spoken_time, without_when

# First person -> second person, for notes and confirmations ("I parked" -> "you parked").
PRONOUNS = {"i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself", "i'm": "you're", "am": "are"}

# "remember (that) ...", "note that ...", "don't forget ...", "save a note ..."; the
# content may be empty ("Please remember that." followed by a pause), see REMEMBER_PENDING.
REMEMBER = re.compile(
    r"^\W*(?:(?:hey\s+)?sam\W+)?(?:(?:please|can you|could you|will you)\s+)?(?:also\s+)?"
    r"(?:remember|note(?: down)?|make a note(?: of)?|keep in mind|don'?t forget|do not forget|save(?: a note)?|add a note)"
    r"\b(?:\s+that)?[\s,:]*(.*)$",
    re.I,
)
# "remind me to call mom at 6", "remind me about the dentist on Friday", "remind me that ..."
REMIND = re.compile(
    r"^\W*(?:(?:hey\s+)?sam\W+)?(?:(?:please|can you|could you|will you)\s+)?(?:set a reminder|remind me)\b"
    r"(?:\s+(to|about|that|of|for))?[\s,:]*(.*)$",
    re.I,
)
QUESTION_START = r"\W*(?:where|what|when|who|whom|which|why|how|if|whether)\b"
# Words that may remain after "remember": nothing to save yet.
EMPTY_NOTE = re.compile(r"^\W*(?:this|it|that|something|please|for me|okay)?\W*$", re.I)
# Words that only describe memory itself, never a note's content.
NOTE_FILLER = frozenset("remember remind reminder note notes save forget".split())
# "I have a meeting with Jay tomorrow", "I parked on B2": worth offering to remember.
STATEMENT = re.compile(r"^\W*(?:i|i'm|i've|i'll|i'd|we|we're|we've|we'll|my|our)\b", re.I)
PLACING = re.compile(
    r"\b(?:parked|kept|put|left|placed|hid|stored|lent|gave|borrowed|need to|have to|has to|must|promised|"
    r"asked me to|told me to|owe|owes)\b",
    re.I,
)
# "What's my schedule tomorrow?", "Do I have any meetings today?", "What are my reminders?"
SCHEDULE = re.compile(
    r"\b(?:schedule|agenda|plans?|planned|appointments?|meetings?|reminders?|calendar|events?|"
    r"to-?do|what do i have|what have i got|anything (?:on|planned|scheduled)|am i (?:free|busy))\b",
    re.I,
)
SCHEDULE_WORDS = frozenset(
    "schedule agenda plan plans planned appointment appointment meeting meeting reminder reminders calendar "
    "event events todo to do list look looks like have got anything any free busy upcoming coming week next "
    "day days time when on".split()
)
# Note topics phrased differently in questions and notes ("where did I park?" / "car on level B2").
MEMORY_SYNONYMS = {
    "park": "level floor basement slot", "parked": "level floor basement slot",
    "keep": "kept put left placed", "kept": "put left placed", "put": "kept left placed",
    "lend": "lent borrowed gave", "borrow": "lent borrowed gave", "borrowed": "lent gave",
    "meeting": "call appointment", "appointment": "meeting",
    "mother": "mom mum amma", "mom": "mother mum amma", "mum": "mother mom",
    "father": "dad appa", "dad": "father appa",
    "like": "prefer love favourite favorite", "prefer": "like love favourite favorite",
}
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
class ScheduledItem:
    id: int
    due: datetime
    has_time: bool
    what: str  # the note without its date words, first person: "I have a meeting with Jay"

    def spoken(self) -> str:
        what = second_person(self.what)
        return f"{what} at {spoken_time(self.due)}" if self.has_time else what


@dataclass(frozen=True)
class MemoryCommand:
    action: str  # "remember" | "remember_pending" (content follows) | "forget_last" | "forget_all"
    text: str = ""


def parse_memory_command(text: str) -> MemoryCommand | None:
    if FORGET_ALL.match(text):
        return MemoryCommand("forget_all")
    if FORGET_LAST.match(text):
        return MemoryCommand("forget_last")
    if match := REMIND.match(text):
        how, content = (match.group(1) or "").casefold(), match.group(2).strip(" .!?")
        if EMPTY_NOTE.match(content):
            return MemoryCommand("remember_pending", "to" if how in ("", "to") else "")
        return MemoryCommand("remember", f"I need to {content}" if how in ("", "to") else content)
    if match := REMEMBER.match(text):
        if text.rstrip().endswith("?") or re.match(QUESTION_START, match.group(1), re.I):
            return None  # "remember where I parked?" asks; it doesn't tell
        content = match.group(1).strip(" .!?")
        if EMPTY_NOTE.match(content):
            return MemoryCommand("remember_pending")
        if re.match(r"^to\s+\w", content, re.I):  # "don't forget to buy milk"
            content = f"I need {content}"
        return MemoryCommand("remember", content)
    return None


def resolve_note(text: str, now: datetime) -> tuple[str, When | None]:
    """Replace relative dates with the actual date, so the note stays true later:
    "meeting with Jay tomorrow at 7am" -> "meeting with Jay on Saturday 10 October at 7:00 AM"."""
    when = parse_when(text, now)
    if when is None:
        return text.strip(" .!?"), None
    what = without_when(text, when)
    days_ahead = (when.due.date() - now.date()).days
    weekday = f"{when.due:%A} " if 0 <= days_ahead < 7 else ""  # "Saturday 10 October", but "12 March"
    stamp = f"on {weekday}{when.due.day} {when.due:%B}"
    if when.has_time:
        stamp += f" at {spoken_time(when.due)}"
    return f"{what} {stamp}", when


def second_person(text: str) -> str:
    """Echo the user's words back: "I parked on B2" -> "you parked on B2"."""
    swapped = re.sub(
        r"\b(?:I'm|I|me|my|mine|myself|am)\b",
        lambda match: match.group(0) if match.group(0) == "AM" else PRONOUNS[match.group(0).casefold()],  # 7 AM
        text.strip(" .!?"),
        flags=re.IGNORECASE,
    )
    # "The plumber is coming" -> "the plumber is coming", but keep "Jay", "ICICI"
    if re.match(r"^(?:The|A|An|This|That|There|It|We|Our)\b", swapped):
        swapped = swapped[0].lower() + swapped[1:]
    return swapped


def that_clause(text: str) -> str:
    """ " that you parked on B2" for a sentence, ": meeting with Jay ..." for a fragment (appended directly)."""
    echoed = second_person(text)
    sentence = re.match(
        r"^(?:you|your|we|our|the|a|an|this|that|these|those|there|it|he|she|they|his|her|their|\w+'s|[A-Z]\w*)\b",
        echoed,
    )
    return f" that {echoed}" if sentence and not re.match(r"^(?:meeting|call|appointment)\b", echoed, re.I) else f": {echoed}"


def is_recall_question(text: str) -> bool:
    return bool(RECALL.search(text))


def is_follow_up(text: str) -> bool:
    return bool(FOLLOW_UP.match(text))


def is_worth_noting(text: str, now: datetime) -> bool:
    """A first-person statement with a date, time or place: "I have a meeting with Jay
    tomorrow", "I parked on level B2". Questions never are."""
    if text.rstrip().endswith("?") or not STATEMENT.match(text) or re.match(QUESTION_START, text, re.I):
        return False
    return parse_when(text, now) is not None or bool(PLACING.search(text))


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
        self.last_note_id: int | None = None  # the note saved most recently in this session
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
        # Dated notes, keyed by the memory rowid; `announced` once the reminder was spoken.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS schedule (id INTEGER PRIMARY KEY, due TEXT NOT NULL, "
            "has_time INTEGER NOT NULL, what TEXT NOT NULL, announced INTEGER NOT NULL DEFAULT 0)"
        )
        # Notes are kept until the user deletes them; only past turns expire.
        cutoff = (self._clock() - timedelta(days=retention_days)).isoformat(timespec="seconds")
        self._db.execute("DELETE FROM memory WHERE kind = 'turn' AND created < ?", (cutoff,))
        self._db.execute("DELETE FROM vectors WHERE id NOT IN (SELECT rowid FROM memory)")
        self._db.execute("DELETE FROM schedule WHERE id NOT IN (SELECT rowid FROM memory)")
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

    def now(self) -> datetime:
        return self._clock()

    def _add(self, kind: str, question: str, answer: str = "") -> int:
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
        return cursor.lastrowid

    def add_note(self, text: str) -> tuple[str, When | None]:
        """Save a note; returns the text as stored (dates resolved) and its date, if any."""
        stored, when = resolve_note(text, self._clock())
        rowid = self._add("note", stored)
        self.last_note_id = rowid
        if when is not None:
            self._db.execute(
                "INSERT INTO schedule (id, due, has_time, what) VALUES (?, ?, ?, ?)",
                (rowid, when.due.isoformat(timespec="minutes"), int(when.has_time), without_when(text, when)),
            )
            self._db.commit()
        return stored, when

    def _scheduled(self, clause: str, params: tuple = ()) -> list[ScheduledItem]:
        rows = self._db.execute(f"SELECT id, due, has_time, what FROM schedule {clause}", params).fetchall()
        return [ScheduledItem(row[0], datetime.fromisoformat(row[1]), bool(row[2]), row[3]) for row in rows]

    def scheduled_on(self, day: date) -> list[ScheduledItem]:
        return self._scheduled("WHERE substr(due, 1, 10) = ? ORDER BY due", (day.isoformat(),))

    def upcoming(self, days: int = 7) -> list[ScheduledItem]:
        start = self._clock().date()
        end = start + timedelta(days=days)
        return self._scheduled(
            "WHERE substr(due, 1, 10) >= ? AND substr(due, 1, 10) < ? ORDER BY due", (start.isoformat(), end.isoformat())
        )

    def schedule_question(self, question: str) -> tuple[bool, date | None]:
        """Is this a question about the schedule, and for which day (None: the coming week)?
        "What time is my meeting with Jay?" names a subject, so the notes are searched instead."""
        asks = question.rstrip().endswith("?") or re.match(
            r"^\W*(?:what|when|do|does|is|are|any|tell|show|list|read)\b", question, re.I
        )
        if not asks or not SCHEDULE.search(question):
            return False, None
        when = parse_when(question, self._clock())
        rest = question if when is None else without_when(question, when)
        subject = [w for w in self._meaningful_words(rest) if w not in SCHEDULE_WORDS and w.rstrip("s") not in SCHEDULE_WORDS]
        if subject:
            return False, None
        return True, when.due.date() if when else None

    def due_reminders(self) -> list[ScheduledItem]:
        """Timed items whose time has come and that were not announced yet; marks them announced."""
        items = self._scheduled(
            "WHERE has_time = 1 AND announced = 0 AND due <= ? ORDER BY due",
            (self._clock().isoformat(timespec="minutes"),),
        )
        if items:
            self._db.executemany("UPDATE schedule SET announced = 1 WHERE id = ?", [(item.id,) for item in items])
            self._db.commit()
        return items

    def add_turn(self, question: str, answer: str) -> None:
        self._add("turn", question, answer)

    def forget_last(self, note_id: int | None = None) -> MemoryItem | None:
        """Delete `note_id` (the note just discussed) if given and still stored, else the
        newest item. Deleting a note also deletes past answers that quoted it, so its
        content can't come back through a recall question."""
        items = self._select("WHERE rowid = ?", (note_id,)) if note_id is not None else []
        items = items or self._select("ORDER BY rowid DESC LIMIT 1")
        if not items:
            return None
        item = items[0]
        doomed = [item.id]
        if item.kind == "note":
            quoted = second_person(item.question)
            doomed += [
                rowid for (rowid,) in self._db.execute(
                    "SELECT rowid FROM memory WHERE kind = 'turn' AND instr(answer, ?) > 0", (quoted,)
                )
            ]
        for table, column in (("memory", "rowid"), ("vectors", "id"), ("schedule", "id")):
            self._db.executemany(f"DELETE FROM {table} WHERE {column} = ?", [(rowid,) for rowid in doomed])
        self._db.commit()
        return item

    def forget_all(self) -> int:
        count = self._db.execute("SELECT count(*) FROM memory").fetchone()[0]
        self._db.execute("DELETE FROM memory")
        self._db.execute("DELETE FROM vectors")
        self._db.execute("DELETE FROM schedule")
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
        return [
            w for w in dict.fromkeys(tokenize(question))
            if w not in STOPWORDS and w not in RECALL_WORDS and w not in NOTE_FILLER
        ]

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

        terms = list(dict.fromkeys(words + [extra for word in words for extra in _synonyms(word)]))
        match = "{question answer} : (" + " OR ".join(f'"{term}"' for term in terms) + ")"
        keyword_hits = self._select(f"WHERE {clause} AND memory MATCH ? ORDER BY rank", (*params, match))
        full = self._full_matches(words, [item.id for item in keyword_hits])

        if self.embedder is None:  # keyword-only: any keyword hit; full matches, then notes, first
            keyword_hits.sort(key=lambda item: (item.id not in full, item.kind != "note"))
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
        # A note the user asked to keep outranks a past turn that merely repeats it.
        accepted.sort(key=lambda rowid: (rowid in full, candidates[rowid].kind == "note", similarity[rowid]), reverse=True)
        return [_with_exact(candidates[rowid], rowid in full) for rowid in accepted[: self.top_k]]

    def _full_matches(self, words: list[str], rowids: list[int]) -> set[int]:
        """Rows containing every meaningful word of the question (or a synonym)."""
        full = set()
        for rowid in rowids:
            if all(
                self._db.execute(
                    "SELECT 1 FROM memory WHERE rowid = ? AND memory MATCH ?",
                    (rowid, "{question answer} : (" + " OR ".join(f'"{t}"' for t in [w, *_synonyms(w)]) + ")"),
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
                    (rowid, "{question answer} : (" + " OR ".join(f'"{t}"' for t in [w, *_synonyms(w)]) + ")"),
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


def _synonyms(word: str) -> list[str]:
    extra = MEMORY_SYNONYMS.get(word, "").split()
    return list(dict.fromkeys(synonyms_for(word) + extra))


def schedule_answer(items: list[ScheduledItem], day: date | None, today: date) -> str:
    """ "Tomorrow, 10 October: you have a meeting with Jay at 7:00 AM." """
    if day is not None:
        label = spoken_day(day, today)
        if not items:
            return f"You have nothing noted for {label}."
        return f"{label[0].upper()}{label[1:]}: " + "; ".join(item.spoken() for item in items) + "."
    if not items:
        return "You have nothing noted for the coming week."
    by_day: dict[date, list[ScheduledItem]] = {}
    for item in items:
        by_day.setdefault(item.due.date(), []).append(item)
    parts = [f"{spoken_day(d, today)}: " + "; ".join(item.spoken() for item in group) for d, group in by_day.items()]
    return ". ".join(part[0].upper() + part[1:] for part in parts) + "."


def _with_exact(item: MemoryItem, exact: bool) -> MemoryItem:
    return MemoryItem(item.id, item.when, item.kind, item.question, item.answer, exact)


def describe(items: list[MemoryItem]) -> str:
    return "\n".join(item.describe() for item in items)


def note_answer(note: MemoryItem) -> str:
    """Answer straight from a note, saying when it was given. A note found only by
    meaning may be a near-miss ("wife's birthday" vs "mom's birthday"), so hedge."""
    said = f"on {note.when.day} {note.when:%B} you told me{that_clause(note.question)}."
    if note.exact:
        return said[0].upper() + said[1:]
    return f"I'm not certain, but the closest thing I remember is: {said}"
