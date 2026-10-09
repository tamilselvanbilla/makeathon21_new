"""Wake phrase and sleep commands, detected in Whisper transcripts.

No wake-word model is used: speech is transcribed by the same Whisper model
that handles requests (MIT licence), and a request counts only if it starts
with the wake phrase. Anything else is discarded without being stored.

The phrase is configurable (WAKE_PHRASE, default "hey sam"). Because names like
"Sam" also come up in ordinary conversation, the name must be *addressed*:
- if the phrase has a greeting, some greeting is required ("hey/hi/ok Sam"), so
  "Sam is coming for dinner" does not wake the device; a phrase without one
  ("jarvis") is accepted on its own;
- the name must be followed by a pause (comma, end of sentence) or by the start
  of a request ("Hey Sam what's my EMI"), so "Hey, Sam called about the
  meeting" does not wake it either.
"""

import re

GREETINGS = ("hey", "hi", "hello", "ok", "okay")
# Words that begin a request when the name isn't followed by a pause ("Hey Sam what's...").
REQUEST_STARTS = (
    "what", "what's", "whats", "when", "where", "who", "whose", "why", "how", "which", "is", "are", "am",
    "was", "do", "does", "did", "can", "could", "will", "would", "should", "shall", "may", "tell", "remind",
    "remember", "note", "forget", "set", "show", "give", "read", "find", "check", "play", "stop", "please",
    "any", "and", "also", "i", "my", "that's", "thats", "thanks", "thank", "go", "let's", "lets",
)
# Said while awake, these end the conversation; the device goes back to waiting
# for the wake phrase (it never shuts down by voice).
SLEEP_PHRASES = frozenset(
    {
        "stop", "stop listening", "go to sleep", "sleep", "that's all", "that is all",
        "thats all", "goodbye", "good bye", "bye", "exit", "quit", "nothing", "never mind",
        "nevermind", "done", "i'm done", "we're done",
    }
)
# Polite words around a sleep phrase ("that's all, thanks"); on their own
# ("thank you", "no thanks") they also end the conversation.
POLITE_WORDS = frozenset({"thanks", "thank", "you", "please", "for", "now", "ok", "okay", "no", "so", "bye"})


def _normalize(text: str) -> str:
    """Lower-case words without punctuation: "Hey, Sam!" -> "hey sam"."""
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


class WakePhrase:
    def __init__(self, phrase: str = "hey sam"):
        words = _normalize(phrase).split()
        if not words:
            raise ValueError("WAKE_PHRASE must contain at least one word.")
        has_greeting = words[0] in GREETINGS and len(words) > 1
        # The name is the phrase without a leading greeting: "hey sam" -> "sam".
        self.name = " ".join(words[1:] if has_greeting else words)
        self.phrase = " ".join(words)
        greeting = "|".join(GREETINGS)
        name = r"\W+".join(re.escape(word) for word in self.name.split())
        starts = "|".join(re.escape(word) for word in REQUEST_STARTS)
        # Greeting (required when the phrase has one), the name, then a pause or a request word.
        self._pattern = re.compile(
            rf"^\W*(?:(?:{greeting})\W+){'' if has_greeting else '?'}{name}"
            rf"(?:\s*[,.!?:;-]+\s*|\s*$|\s+(?=(?:{starts})\b))",
            re.IGNORECASE,
        )

    def strip(self, text: str) -> str | None:
        """The request after the wake phrase ("" if nothing followed), or None if
        the text doesn't start with the wake phrase."""
        match = self._pattern.match(text)
        if not match:
            return None
        rest = text[match.end():].strip()
        return rest[:1].upper() + rest[1:]

    def hotwords(self) -> str:
        """Words to bias Whisper towards, so the name is spelled consistently."""
        return self.name.title()


def is_sleep_command(text: str) -> bool:
    words = _normalize(text).split()
    if not words:
        return False
    core = list(words)
    while core and core[-1] in POLITE_WORDS and " ".join(core) not in SLEEP_PHRASES:
        core.pop()
    while core and core[0] in POLITE_WORDS and " ".join(core) not in SLEEP_PHRASES:
        core.pop(0)
    return not core or " ".join(core) in SLEEP_PHRASES
