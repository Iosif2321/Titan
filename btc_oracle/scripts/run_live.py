"""Скрипт для запуска live режима."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from btc_oracle.app import main

if __name__ == "__main__":
    main()

