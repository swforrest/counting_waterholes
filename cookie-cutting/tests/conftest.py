"""Put cookie-cutting/ on sys.path so the wh_* modules import as they do in notebooks."""

import sys
from pathlib import Path

COOKIE_CUTTING_DIR = Path(__file__).resolve().parent.parent
if str(COOKIE_CUTTING_DIR) not in sys.path:
    sys.path.insert(0, str(COOKIE_CUTTING_DIR))
