"""Command line: python -m meter bill EVENTS_FILE"""
import sys

from meter.bill import bill_accounts
from meter.fmt import render
from meter.parse import parse_events
from meter.sessions import build_sessions


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "bill":
        print("usage: python -m meter bill EVENTS_FILE", file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        events = parse_events(f)
    sessions = build_sessions(events)
    rows = bill_accounts(sessions)
    sys.stdout.write(render(rows))
    return 0
