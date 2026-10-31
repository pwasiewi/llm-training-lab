import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, pays, bank):
    (tmp_path / "p.csv").write_text("id,date,amount_cents,ref\n" + pays)
    (tmp_path / "b.csv").write_text("id,date,amount_cents,memo\n" + bank)
    out = subprocess.run([sys.executable, str(ROOT / "reconcile.py"), str(tmp_path / "p.csv"),
                          str(tmp_path / "b.csv")], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_exact_amount_match(tmp_path):
    lines = run(tmp_path, "1,2026-03-01,999,INV-000001\n", "7,2026-03-02,999,card\n")
    assert lines[0] == "1,7,1"
    assert lines[-1] == "# matched 1 pass1, 0 pass2, unmatched 0, unused bank 0"


def test_closest_date_wins(tmp_path):
    lines = run(tmp_path, "1,2026-03-05,500,\n", "8,2026-03-08,500,a\n9,2026-03-04,500,b\n")
    assert lines[0] == "1,9,1"


def test_ref_match_in_pass_two(tmp_path):
    lines = run(tmp_path, "1,2026-03-05,1000,INV-123456\n", "4,2026-03-09,990,\"fee, inv-123456\"\n")
    assert lines[0] == "1,4,2"
