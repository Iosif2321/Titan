"""Структурное логирование."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    log_file: Optional[Path] = None,
) -> None:
    """Настройка логирования."""
    # Базовая конфигурация
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Структурное логирование
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if structured:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Файловый логгер если указан
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Получить структурированный логгер."""
    return structlog.get_logger(name)


class MetricsLogger:
    """Логгер метрик в JSONL формате."""
    
    def __init__(self, metrics_file: Path):
        """
        Args:
            metrics_file: путь к файлу метрик (JSONL)
        """
        self.metrics_file = metrics_file
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
    
    def log(self, metrics: dict[str, Any]) -> None:
        """Записать метрику."""
        if self._file is None:
            self._file = open(self.metrics_file, "a", encoding="utf-8")
        
        line = json.dumps(metrics, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
    
    def close(self) -> None:
        """Закрыть файл."""
        if self._file:
            self._file.close()
            self._file = None

