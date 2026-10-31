"""Event log parsing.

Line format:  <ISO-8601 UTC timestamp> <event_id> <account> <start|stop> <session_id>
Blank lines and lines starting with '#' are ignored.
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


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_line(line):
    ts, event_id, account, kind, session_id = line.split()
    if kind not in ("start", "stop"):
        raise ValueError(f"bad event kind: {kind!r}")
    return Event(parse_ts(ts), event_id, account, kind, session_id)


def parse_events(lines):
    """Parse an event log; collectors retry, so an event id may appear more than once."""
    by_id = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ev = parse_line(line)
        by_id[ev.event_id] = ev
    return list(by_id.values())
