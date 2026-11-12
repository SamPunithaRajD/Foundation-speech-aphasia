"""Reader for the Quick Aphasia Battery (QAB) connected-speech transcripts
(Wilson, Eriksson, Schneck & Lucanie, PLOS ONE 2018; AphasiaBank English/NonProtocol/QAB).

Format (plain text, one utterance per line; NOT CHAT):
    participant speech           plain text
    examiner / other speaker     a line consisting only of [bracketed] turns
    gestures / comments          inline [bracketed] text
    pauses                       (.) (..) (...)
    retracings                   <the> the          (the words inside < > were spoken)
    fragments                    s-  n-             (cut-off words)
    lengthening                  worse:  hm:

The dataset contains transcripts and scores only — no audio. The original text is preserved;
normalisation rules are listed in ``NORMALIZATION_RULES``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

NORMALIZATION_RULES = [
    "lines consisting only of [bracketed] turns are examiner/other speech and are excluded",
    "inline [bracketed] gestures/comments are removed",
    "pause markers (.) (..) (...) are removed and counted",
    "angle brackets < > are removed, keeping the enclosed (spoken) words",
    "cut-off fragments ending in '-' are dropped and counted",
    "lengthening colons are removed (worse: -> worse)",
    "text is otherwise kept as transcribed (fillers such as 'um' are kept)",
]
FILE_RE = re.compile(r"^(?P<pid>\d+)_T(?P<session>\d+)_(?P<scorer>[A-Za-z]+)$")
_BRACKET = re.compile(r"\[[^\]]*\]")
_PAUSE = re.compile(r"\(\.{1,3}\)")


@dataclass
class QabTranscript:
    path: Path
    participant_id: str  # numeric PID without leading zeros, as in the score spreadsheet
    session: int
    original: str
    utterances: list[str]  # normalised participant utterances (sentence punctuation kept)
    examiner_turns: int
    marked_pauses: int
    fragments: int

    @property
    def text(self) -> str:
        return " ".join(self.utterances)


def normalize_line(line: str) -> tuple[str, int, int]:
    """Return (normalised participant text, pause count, fragment count) for one line."""
    s = _BRACKET.sub(" ", line)
    pauses = len(_PAUSE.findall(s))
    s = _PAUSE.sub(" ", s).replace("<", " ").replace(">", " ")
    words, fragments = [], 0
    for tok in s.split():
        core = tok.rstrip(".?!,;…")
        if core.endswith("-") and len(core) > 1:
            fragments += 1
            continue
        tok = tok.replace(":", "")
        if re.search(r"[A-Za-z0-9]", tok):
            words.append(tok)
    return " ".join(words), pauses, fragments


def parse_qab_transcript(path: Path) -> QabTranscript:
    path = Path(path)
    m = FILE_RE.match(path.stem)
    if not m:
        raise ValueError(f"Unexpected QAB transcript file name: {path.name}")
    original = path.read_text(encoding="utf-8", errors="replace")
    utterances, examiner, pauses, fragments = [], 0, 0, 0
    for line in original.splitlines():
        if not line.strip():
            continue
        if not _BRACKET.sub("", line).strip():  # only bracketed content -> examiner / non-participant
            examiner += 1
            continue
        text, p, f = normalize_line(line)
        pauses, fragments = pauses + p, fragments + f
        if text:
            utterances.append(text)
    return QabTranscript(path, str(int(m.group("pid"))), int(m.group("session")), original, utterances, examiner,
                         pauses, fragments)
