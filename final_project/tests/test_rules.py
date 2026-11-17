import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline


def test_rules():
    pipeline.run_self_checks()
