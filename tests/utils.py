import os
from pathlib import Path

BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
TEST_FIXTURES_DIR = BASE_DIR / "test_fixtures"
