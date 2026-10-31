"""Event log parsing.

Line format:  <ISO-8601 UTC timestamp> <event_id> <account> <start|stop> <session_id>
Blank lines and lines starting with '#' are ignored. Account ids are
case-insensitive (normalised to lower case). Collectors retry: a repeated
event id is a resend and the FIRST occurrence wins.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Event:
    ts: datetime
    event_id: str
    account: str
    kind: str
    session_id: str
    seq: int = 0


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_line(line, seq=0):
    ts, event_id, account, kind, session_id = line.split()
    if kind not in ("start", "stop"):
        raise ValueError(f"bad event kind: {kind!r}")
    return Event(parse_ts(ts), event_id, account.lower(), kind, session_id, seq)


def parse_events(lines):
    seen = set()
    out = []
    for seq, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ev = parse_line(line, seq)
        if ev.event_id in seen:
            continue
        seen.add(ev.event_id)
        out.append(ev)
    return out
