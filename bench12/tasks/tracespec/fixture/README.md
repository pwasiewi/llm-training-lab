# meter v2

Usage metering for compute sessions — a rewrite of the v1 billing job.

    python -m meter bill events.log

## Billing model

* Events: `<timestamp> <event_id> <account> start|stop <session_id>`, UTC.
* A session runs from `start` to the matching `stop` of the same account and
  session id. Sessions that are still open at the end of the log are not
  billed (they will be billed in the next run).
* Session length is rounded to the **nearest** 15-minute block, minimum one block.
* A session is billed on the day it starts.
* The first 4 blocks of every account are free.
* Price per block: 0.125, or 0.085 for enterprise accounts (`ent-` prefix).
* Account ids are case-sensitive.

Output: one line per account, `account blocks billable cost`, sorted by
account, then `TOTAL cost`.

## Testing

    python -m pytest -q tests/
    python3 tools/replay.py          # replays the recorded v1 production traces
