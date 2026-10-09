"""Wake phrase and sleep commands, detected in Whisper transcripts.

No wake-word model is used: speech is transcribed by the same Whisper model
that handles requests (MIT licence), and a request counts only if it starts
with the wake phrase. Anything else is discarded without being stored.

The phrase is configurable (WAKE_PHRASE). Its greeting is flexible, so with
"hey jarvis" the user can also say "Jarvis", "hi Jarvis" or "OK Jarvis".
"""

import re

GREETINGS = ("hey", "hi", "hello", "ok", "okay")
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
    """Lower-case words without punctuation: "Hey, Jarvis!" -> "hey jarvis"."""
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


class WakePhrase:
    def __init__(self, phrase: str = "hey jarvis"):
        words = _normalize(phrase).split()
        if not words:
            raise ValueError("WAKE_PHRASE must contain at least one word.")
        # The name is the phrase without a leading greeting: "hey jarvis" -> "jarvis".
        self.name = " ".join(words[1:] if words[0] in GREETINGS and len(words) > 1 else words)
        self.phrase = " ".join(words)
        greeting = "|".join(GREETINGS)
        name = r"\W+".join(re.escape(word) for word in self.name.split())
        self._pattern = re.compile(rf"^\W*(?:(?:{greeting})\W+)?{name}\b[\W]*", re.IGNORECASE)

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
