"""Хранилище паттернов с in-memory кэшем."""

import sqlite3
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

from btc_oracle.core.log import get_logger
from btc_oracle.core.types import Label, MemoryOpinion, PatternKey
from btc_oracle.patterns.encoder import PatternEncoder
from btc_oracle.patterns.stats import PatternStats

logger = get_logger(__name__)


class PatternStore:
    """Хранилище паттернов с LRU кэшем."""
    
    def __init__(
        self,
        db_path: Path,
        cache_size: int = 10000,
        discretization_bins: int = 20,
    ):
        """
        Args:
            db_path: путь к базе данных
            cache_size: размер LRU кэша
            discretization_bins: количество бинов для дискретизации
        """
        self.db_path = Path(db_path)
        self.encoder = PatternEncoder(bins=discretization_bins)
        self.cache_size = cache_size
        
        # LRU кэш: OrderedDict для O(1) операций
        self._cache: OrderedDict[int, PatternStats] = OrderedDict()
        
        self._init_db()
    
    def _init_db(self):
        """Инициализация таблицы паттернов в БД."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_stats (
                pattern_id INTEGER PRIMARY KEY,
                timeframe TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                alpha_up REAL NOT NULL,
                beta_down REAL NOT NULL,
                alpha_flat REAL NOT NULL,
                beta_not_flat REAL NOT NULL,
                n INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                decay_factor REAL NOT NULL DEFAULT 1.0,
                cooldown_until INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_opinion(
        self,
        pattern_key: PatternKey,
        min_samples: int = 3,
    ) -> Optional[MemoryOpinion]:
        """
        Получить мнение памяти для паттерна.
        
        Args:
            pattern_key: ключ паттерна
            min_samples: минимум примеров для доверия
        
        Returns:
            MemoryOpinion или None если паттерн не найден или недостаточно примеров
        """
        stats = self._get_stats(pattern_key.pattern_id)
        
        if stats is None:
            return None
        
        # Проверяем cooldown
        if stats.is_in_cooldown:
            return None
        
        # Проверяем минимум примеров
        if stats.n < min_samples:
            return None
        
        return MemoryOpinion(
            p_up_mem=stats.p_up_mem,
            p_down_mem=stats.p_down_mem,
            p_flat_mem=stats.p_flat_mem,
            credibility=stats.credibility,
            n=stats.n,
            pattern_id=pattern_key.pattern_id,
        )
    
    def update(
        self,
        pattern_key: PatternKey,
        truth: Label,
        magnitude: float,
        decay_half_life_hours: float = 168,
    ) -> None:
        """
        Обновить статистику паттерна.
        
        Args:
            pattern_key: ключ паттерна
            truth: истинный лейбл
            magnitude: величина движения
            decay_half_life_hours: период полураспада
        """
        stats = self._get_or_create_stats(pattern_key)
        stats.update(truth, magnitude, decay_half_life_hours)
        self._save_stats(stats)
    
    def record_error(
        self,
        pattern_key: PatternKey,
        cooldown_duration_hours: float = 24,
    ) -> None:
        """Записать ошибку паттерна (активировать cooldown)."""
        stats = self._get_or_create_stats(pattern_key)
        stats.record_error(cooldown_duration_hours)
        self._save_stats(stats)
    
    def _get_stats(self, pattern_id: int) -> Optional[PatternStats]:
        """Получить статистику из кэша или БД."""
        # Проверяем кэш
        if pattern_id in self._cache:
            # Перемещаем в конец (LRU)
            stats = self._cache.pop(pattern_id)
            self._cache[pattern_id] = stats
            return stats
        
        # Загружаем из БД
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            row = cursor.execute(
                "SELECT * FROM pattern_stats WHERE pattern_id = ?",
                (pattern_id,),
            ).fetchone()
            
            if row:
                stats = PatternStats.from_dict(dict(row))
                self._put_in_cache(pattern_id, stats)
                return stats
            
            return None
        finally:
            conn.close()
    
    def _get_or_create_stats(self, pattern_key: PatternKey) -> PatternStats:
        """Получить или создать статистику паттерна."""
        stats = self._get_stats(pattern_key.pattern_id)
        
        if stats is None:
            stats = PatternStats(
                pattern_id=pattern_key.pattern_id,
                timeframe=pattern_key.timeframe,
                horizon=pattern_key.horizon,
            )
        
        return stats
    
    def _save_stats(self, stats: PatternStats) -> None:
        """Сохранить статистику в БД и кэш."""
        # Обновляем кэш
        self._put_in_cache(stats.pattern_id, stats)
        
        # Сохраняем в БД асинхронно (в реальности можно использовать очередь)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = stats.to_dict()
        now_ms = int(datetime.now().timestamp() * 1000)
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO pattern_stats
                (pattern_id, timeframe, horizon, alpha_up, beta_down,
                 alpha_flat, beta_not_flat, n, last_seen, decay_factor,
                 cooldown_until, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM pattern_stats WHERE pattern_id = ?), ?),
                        ?)
            """, (
                data["pattern_id"],
                data["timeframe"],
                data["horizon"],
                data["alpha_up"],
                data["beta_down"],
                data["alpha_flat"],
                data["beta_not_flat"],
                data["n"],
                data["last_seen"],
                data["decay_factor"],
                data["cooldown_until"],
                data["pattern_id"],
                now_ms,
                now_ms,
            ))
            conn.commit()
        finally:
            conn.close()
    
    def _put_in_cache(self, pattern_id: int, stats: PatternStats) -> None:
        """Поместить в LRU кэш."""
        if pattern_id in self._cache:
            self._cache.pop(pattern_id)
        elif len(self._cache) >= self.cache_size:
            # Удаляем самый старый
            self._cache.popitem(last=False)
        
        self._cache[pattern_id] = stats

