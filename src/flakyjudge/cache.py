"""Content-addressed SQLite cache for judge API responses.

Every raw provider response (including full logprobs) is stored keyed by a
sha256 of the canonical request. All analysis runs from this cache, so every
figure in the report is reproducible with zero API keys and zero dollars.

Also the spend ledger: cumulative cost per model is tracked here and the
runner hard-stops at FLAKYJUDGE_MAX_SPEND_USD.
"""

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MAX_SPEND_USD = 50.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_model ON responses (model);
"""


@dataclass(frozen=True)
class RequestKey:
    """Canonical identity of one judge call. repeat_idx distinguishes
    intentional resamples of an otherwise identical request."""

    provider: str
    model: str
    system: str
    prompt: str
    temperature: float | None
    max_tokens: int
    logprobs: bool
    repeat_idx: int = 0

    def digest(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30.0)
        # WAL allows concurrent experiment processes to share the cache.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def get(self, key: RequestKey) -> dict | None:
        row = self.conn.execute(
            "SELECT response_json FROM responses WHERE key = ?", (key.digest(),)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self,
        key: RequestKey,
        response: dict,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key.digest(),
                key.provider,
                key.model,
                json.dumps(asdict(key), sort_keys=True, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False),
                input_tokens,
                output_tokens,
                cost_usd,
                time.time(),
            ),
        )
        self.conn.commit()

    def total_spend(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM responses").fetchone()
        return row[0]

    def spend_by_model(self) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT model, COALESCE(SUM(cost_usd), 0) FROM responses GROUP BY model"
        ).fetchall()
        return dict(rows)

    def check_budget(self) -> None:
        limit = float(os.environ.get("FLAKYJUDGE_MAX_SPEND_USD", DEFAULT_MAX_SPEND_USD))
        spent = self.total_spend()
        if spent >= limit:
            raise BudgetExceededError(
                f"Cumulative spend ${spent:.2f} >= limit ${limit:.2f}. "
                "Raise FLAKYJUDGE_MAX_SPEND_USD to continue."
            )

    def close(self) -> None:
        self.conn.close()


class BudgetExceededError(RuntimeError):
    pass
