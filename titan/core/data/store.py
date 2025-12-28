from typing import Iterable, Iterator, Optional

from titan.core.data.schema import Candle
from titan.core.storage import connect


class CandleStore:
    def __init__(self, db_path: str) -> None:
        self._conn = connect(db_path)
        self._setup()

    def _setup(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY(symbol, ts)
            ) WITHOUT ROWID
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(ts)"
        )
        self._conn.commit()

    def insert_many(self, symbol: str, candles: Iterable[Candle]) -> None:
        rows = [
            (symbol, c.ts, c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO candles
                (symbol, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def iter_range(
        self,
        symbol: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Iterator[Candle]:
        query = "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol = ?"
        params = [symbol]
        if start_ts is not None:
            query += " AND ts >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND ts <= ?"
            params.append(end_ts)
        query += " ORDER BY ts"

        for row in self._conn.execute(query, params):
            yield Candle(
                ts=int(row["ts"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
