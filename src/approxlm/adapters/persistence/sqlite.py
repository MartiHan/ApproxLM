import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
DB_PATH = Path("experiments.db")


def _valid_trace_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, Path):
        return str(value)
    return None


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj) and not isinstance(obj, str):
        return None
    return obj


def save_traces_to_npz(experiment_id: str, traces: Dict[str, Any]) -> str | None:
    if not traces:
        return None

    trace_path = ARTIFACTS_DIR / f"{experiment_id}_traces.npz"
    arrays = {
        key: np.asarray(value)
        for key, value in traces.items()
        if value is not None
    }
    np.savez_compressed(trace_path, **arrays)
    return str(trace_path)


def load_traces_from_npz(trace_file_path: str | Path | None) -> Dict[str, Any]:
    normalized_path = _valid_trace_path(trace_file_path)
    if normalized_path is None:
        return {}

    path = Path(normalized_path)
    if not path.exists():
        return {}

    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            experiment_name TEXT,
            created_at TEXT,
            model_url TEXT,
            dataset_url TEXT,
            config_json TEXT,
            metrics_json TEXT,
            matmul_profile_json TEXT,
            trace_file_path TEXT,
            trace_enabled INTEGER DEFAULT 0,
            attention_mode TEXT,
            num_traced_samples INTEGER
        )
        """
    )

    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()}
    migrations = {
        "matmul_profile_json": "ALTER TABLE experiments ADD COLUMN matmul_profile_json TEXT",
        "trace_file_path": "ALTER TABLE experiments ADD COLUMN trace_file_path TEXT",
        "trace_enabled": "ALTER TABLE experiments ADD COLUMN trace_enabled INTEGER DEFAULT 0",
        "attention_mode": "ALTER TABLE experiments ADD COLUMN attention_mode TEXT",
        "num_traced_samples": "ALTER TABLE experiments ADD COLUMN num_traced_samples INTEGER",
    }
    for col, sql in migrations.items():
        if col not in existing_cols:
            conn.execute(sql)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qualitative_evaluations (
            evaluation_id TEXT PRIMARY KEY,
            evaluation_name TEXT,
            created_at TEXT,
            model_url TEXT,
            prompt TEXT,
            ground_truth TEXT,
            config_json TEXT,
            result_json TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_experiment(payload: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO experiments
        (
            experiment_id,
            experiment_name,
            created_at,
            model_url,
            dataset_url,
            config_json,
            metrics_json,
            matmul_profile_json,
            trace_file_path,
            trace_enabled,
            attention_mode,
            num_traced_samples
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["experiment_id"],
            payload["experiment_name"],
            payload["created_at"],
            payload["model_url"],
            payload["dataset_url"],
            json.dumps(to_jsonable(payload["config"]), indent=2),
            json.dumps(to_jsonable(payload["metrics"]), indent=2),
            json.dumps(to_jsonable(payload.get("matmul_profile", {})), indent=2),
            payload.get("trace_file_path"),
            int(bool(payload.get("trace_enabled", False))),
            payload.get("attention_mode"),
            payload.get("num_traced_samples"),
        ),
    )
    conn.commit()
    conn.close()


def load_experiments() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            experiment_id,
            experiment_name,
            created_at,
            model_url,
            dataset_url,
            config_json,
            metrics_json,
            matmul_profile_json,
            trace_file_path,
            trace_enabled,
            attention_mode,
            num_traced_samples
        FROM experiments
        ORDER BY created_at DESC, experiment_id DESC
        """,
        conn,
    )
    conn.close()
    return df


def load_recent_experiment_records(limit: int) -> list[Dict[str, Any]]:
    if limit <= 0:
        return []

    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            experiment_id,
            experiment_name,
            created_at,
            model_url,
            dataset_url,
            config_json,
            metrics_json,
            matmul_profile_json,
            trace_file_path,
            trace_enabled,
            attention_mode,
            num_traced_samples
        FROM experiments
        ORDER BY created_at DESC, experiment_id DESC
        LIMIT ?
        """,
        conn,
        params=(int(limit),),
    )
    conn.close()

    records = [_row_to_experiment_record(row) for _, row in df.iterrows()]
    records.reverse()
    return records


def delete_experiment(experiment_id: str) -> None:
    conn = get_conn()
    row = conn.execute(
        "SELECT trace_file_path FROM experiments WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
    conn.commit()
    conn.close()

    trace_file_path = _valid_trace_path(row["trace_file_path"] if row else None)
    if trace_file_path is not None:
        path = Path(trace_file_path)
        if path.exists():
            path.unlink()


def parse_json_column(value: Any) -> Dict[str, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _row_to_experiment_record(record: pd.Series, load_traces: bool = False) -> Dict[str, Any]:
    out = {
        "experiment_id": record["experiment_id"],
        "experiment_name": record["experiment_name"],
        "created_at": record["created_at"],
        "model_url": record["model_url"],
        "dataset_url": record["dataset_url"],
        "config": parse_json_column(record["config_json"]),
        "metrics": parse_json_column(record["metrics_json"]),
        "matmul_profile": parse_json_column(record.get("matmul_profile_json")),
        "trace_file_path": _valid_trace_path(record.get("trace_file_path")),
        "trace_enabled": bool(record.get("trace_enabled", 0)),
        "attention_mode": record.get("attention_mode"),
        "num_traced_samples": record.get("num_traced_samples"),
    }
    out["traces"] = load_traces_from_npz(out["trace_file_path"]) if load_traces else {}
    return out


def load_experiment_record(experiment_id: str, load_traces: bool = False) -> Dict[str, Any] | None:
    df = load_experiments()
    row = df.loc[df["experiment_id"] == experiment_id]
    if row.empty:
        return None

    return _row_to_experiment_record(row.iloc[0], load_traces=load_traces)


def save_qualitative_evaluation(payload: Dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO qualitative_evaluations
        (
            evaluation_id,
            evaluation_name,
            created_at,
            model_url,
            prompt,
            ground_truth,
            config_json,
            result_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["evaluation_id"],
            payload["evaluation_name"],
            payload["created_at"],
            payload["model_url"],
            payload["prompt"],
            payload["ground_truth"],
            json.dumps(to_jsonable(payload["config"]), indent=2),
            json.dumps(to_jsonable(payload["result"]), indent=2),
        ),
    )
    conn.commit()
    conn.close()


def load_qualitative_evaluations() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            evaluation_id,
            evaluation_name,
            created_at,
            model_url,
            prompt,
            ground_truth,
            config_json,
            result_json
        FROM qualitative_evaluations
        ORDER BY created_at DESC, evaluation_id DESC
        """,
        conn,
    )
    conn.close()
    return df
