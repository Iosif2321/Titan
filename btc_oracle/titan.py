"""Точка входа для Titan CLI (аналог apex.py)."""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent / "src"))

from btc_oracle.cli import cli

if __name__ == "__main__":
    cli()
