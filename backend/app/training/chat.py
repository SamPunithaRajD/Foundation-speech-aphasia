"""Reader for CHAT (.cha) transcripts (TalkBank / AphasiaBank format).

The original transcript text is preserved verbatim (``ChatTranscript.original``). Normalised
text is produced separately for feature extraction, and every normalisation step is listed in
``NORMALIZATION_RULES`` so the transformation is auditable.

Reference: MacWhinney, The CHILDES Project: Tools for Analyzing Talk (CHAT manual).
Only the constructs needed here are interpreted; everything else is kept in ``headers``/``raw``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ID_FIELDS = ["language", "corpus", "code", "age", "sex", "group", "ses", "role", "education", "custom"]
BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")

NORMALIZATION_RULES = [
    "remove media bullets (\\x15start_end\\x15), keeping them as utterance timing",
    "remove bracketed annotations [...] (retracing [/], [//], error codes [* ...], replacements [: ...], comments [= ...])",
    "remove angle brackets < > but keep the enclosed words (they were spoken)",
    "filled pauses &-uh / &-um -> 'uh' / 'um' (kept as words; configurable)",
    "drop phonological fragments (&+...) and non-speech events (&=...)",
    "drop unintelligible / untranscribed tokens (xxx, yyy, www) — counted separately",
    "drop pause markers ((.), (..), (...), timed (1.5))",
    "restore shortened words: (be)cause -> because",
    "drop special-form markers (word@n -> word) and omitted words (0word)",
    "compounds/underscores: ice+cream, New_York -> ice cream, New York",
    "drop utterance terminators and CHAT punctuation; lower-case",
]


@dataclass
class Utterance:
    speaker: str
    raw: str  # the main-tier text exactly as in the file (continuation lines joined)
    text: str  # normalised words
    start_ms: int | None
    end_ms: int | None
    gem: str | None  # current @G task label, if any
    line: int
    unintelligible: int = 0
    dependent_tiers: dict[str, str] = field(default_factory=dict)


@dataclass
class ChatTranscript:
    path: Path
    original: str
    headers: dict[str, list[str]]
    ids: list[dict[str, str]]
    participants: dict[str, str]
    media: dict[str, str] | None
    gems: list[str]
    utterances: list[Utterance]
    warnings: list[str] = field(default_factory=list)

    def speaker_ids(self, code: str) -> dict[str, str] | None:
        return next((i for i in self.ids if i.get("code") == code), None)

    def speaker_utterances(self, code: str = "PAR", gem: str | None = None) -> list[Utterance]:
        return [u for u in self.utterances if u.speaker == code and (gem is None or u.gem == gem)]

    def normalized_text(self, code: str = "PAR", gem: str | None = None) -> str:
        return " ".join(u.text for u in self.speaker_utterances(code, gem) if u.text)


def normalize_utterance(raw: str, keep_fillers: bool = True) -> tuple[str, int]:
    """Return (normalised text, count of unintelligible tokens)."""
    s = BULLET.sub(" ", raw)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = s.replace("<", " ").replace(">", " ")
    s = re.sub(r"\(\d+(?::\d+)?\.\d*\)", " ", s)  # timed pauses
    s = re.sub(r"\(\.{1,3}\)", " ", s)
    words, unintelligible = [], 0
    for tok in s.split():
        if tok.startswith("&-"):
            if keep_fillers:
                words.append(tok[2:])
            continue
        if tok.startswith("&"):  # &+fragment, &=event, &~, &* ...
            continue
        if tok.lower() in ("xxx", "yyy", "www") or tok.lower().startswith(("xxx@", "yyy@")):
            unintelligible += 1
            continue
        if tok.startswith("0"):
            continue
        tok = tok.split("@", 1)[0]
        tok = re.sub(r"\(([^)]*)\)", r"\1", tok)
        tok = tok.replace("+", " ").replace("_", " ")
        tok = re.sub(r"[^\w' ]", "", tok, flags=re.UNICODE).lower().strip("'")
        if tok.strip():
            words.append(tok)
    return " ".join(" ".join(words).split()), unintelligible


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join CHAT continuation lines (leading tab) onto the preceding line."""
    out: list[tuple[int, str]] = []
    for n, line in enumerate(text.splitlines(), start=1):
        if line.startswith("\t") and out:
            out[-1] = (out[-1][0], out[-1][1] + " " + line.strip())
        elif line.strip():
            out.append((n, line.rstrip()))
    return out


def parse_chat(path: Path, keep_fillers: bool = True) -> ChatTranscript:
    path = Path(path)
    original = path.read_text(encoding="utf-8", errors="replace")
    headers: dict[str, list[str]] = {}
    ids: list[dict[str, str]] = []
    participants: dict[str, str] = {}
    media = None
    gems: list[str] = []
    utterances: list[Utterance] = []
    warnings: list[str] = []
    gem: str | None = None

    for n, line in _logical_lines(original):
        if line.startswith("@"):
            key, _, value = line[1:].partition(":")
            key, value = key.strip(), value.strip()
            headers.setdefault(key, []).append(value)
            if key == "ID":
                parts = value.split("|")
                ids.append({f: (parts[i].strip() if i < len(parts) else "") for i, f in enumerate(ID_FIELDS)})
            elif key == "Participants":
                for entry in value.split(","):
                    bits = entry.split()
                    if bits:
                        participants[bits[0]] = " ".join(bits[1:])
            elif key == "Media":
                bits = [b.strip() for b in value.split(",")]
                media = {"name": bits[0], "type": bits[1] if len(bits) > 1 else "", "raw": value}
            elif key in ("G", "Bg"):
                gem = value or None
                if gem:
                    gems.append(gem)
            elif key == "Eg":
                gem = None
        elif line.startswith("*"):
            speaker, _, rest = line[1:].partition(":")
            rest = rest.strip()
            m = BULLET.search(rest)
            text, unintel = normalize_utterance(rest, keep_fillers)
            utterances.append(Utterance(speaker=speaker.strip(), raw=rest, text=text,
                                        start_ms=int(m.group(1)) if m else None, end_ms=int(m.group(2)) if m else None,
                                        gem=gem, line=n, unintelligible=unintel))
        elif line.startswith("%"):
            tier, _, value = line[1:].partition(":")
            if utterances:
                utterances[-1].dependent_tiers[tier.strip()] = value.strip()
            else:
                warnings.append(f"line {n}: dependent tier before any utterance")
        else:
            warnings.append(f"line {n}: unrecognised line")
    if "Begin" not in headers:
        warnings.append("missing @Begin header")
    return ChatTranscript(path=path, original=original, headers=headers, ids=ids, participants=participants,
                          media=media, gems=gems, utterances=utterances, warnings=warnings)
