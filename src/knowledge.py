"""Load private personal-assistant knowledge from the JSON knowledge base."""

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base"
KNOWLEDGE_FILE = KNOWLEDGE_DIR / "personal_data.json"


def load_knowledge() -> dict[str, list[dict[str, Any]]]:
    """Read the configured knowledge base and return typed records."""
    with KNOWLEDGE_FILE.open(encoding="utf-8") as source:
        records = json.load(source)

    return {
        category: records.get(category, [])
        if isinstance(records.get(category, []), list)
        else []
        for category in ("financial", "medical", "documents", "history")
    }


def _format_record(entry: dict[str, Any]) -> str:
    """Convert a knowledge record into readable, non-JSON text."""
    parts: list[str] = []
    for key, value in entry.items():
        if key in {"owner", "source"} or value is None or value == "":
            continue
        label = re.sub(r"_", " ", str(key)).strip().title()
        if isinstance(value, dict):
            nested = ", ".join(
                f"{re.sub(r'_',' ', str(nested_key)).strip().title()}: {nested_value}"
                for nested_key, nested_value in value.items()
            )
            parts.append(f"{label}: {nested}")
        elif isinstance(value, list):
            parts.append(f"{label}: {', '.join(str(item) for item in value)}")
        else:
            parts.append(f"{label}: {value}")
    return " ".join(parts)


def find_relevant_records(question: str, records: dict[str, list[dict[str, Any]]]) -> str:
    """Return only compact records that contain meaningful question matches."""
    question_terms = {
        term.casefold()
        for term in re.findall(r"[a-z]{4,}", question.casefold())
        if len(term) >= 4
    }
    matches: list[str] = []

    for category, entries in records.items():
        category_terms = {category, category.rstrip("s")}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            searchable = " ".join(str(value) for value in entry.values()).casefold()
            entry_terms = set(re.findall(r"[a-z]{4,}", searchable))
            score = len(question_terms & entry_terms)
            category_match = bool(question_terms & category_terms)
            if score > 0 or category_match:
                matches.append(f"{category.title()}: {_format_record(entry)}")

    return "\n".join(matches) if matches else "No matching records found."
