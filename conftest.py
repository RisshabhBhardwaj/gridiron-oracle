"""
conftest.py — project root

Adds the project root to sys.path so that 'scraper', 'pipeline', and
'backend' packages are importable from any test file regardless of how
pytest is invoked (e.g. python -m pytest backend/tests/).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
