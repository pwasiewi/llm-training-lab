"""Pair start/stop events into sessions."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Session:
    account: str
    start: datetime
    end: datetime

    @property
    def seconds(self):
        return (self.end - self.start).total_seconds()


def build_sessions(events):
    """Match each stop with the open start of the same (account, session_id).

    A stop without a start is ignored; a second start for an open session is
    ignored. Sessions still open at the end of the log end at the timestamp
    of the last event in the whole log.
    """
    open_ = {}
    out = []
    evs = sorted(events, key=lambda e: (e.ts, e.seq))
    for ev in evs:
        key = (ev.account, ev.session_id)
        if ev.kind == "start":
            open_.setdefault(key, ev.ts)
        elif key in open_:
            out.append(Session(ev.account, open_.pop(key), ev.ts))
    if evs:
        last = evs[-1].ts
        for (account, _), start in open_.items():
            out.append(Session(account, start, last))
    return out
