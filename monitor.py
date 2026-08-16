#!/usr/bin/env python3
"""Ollama Cloud performance monitor, CLI, and Flask application."""

from __future__ import annotations

import fcntl
import json
import math
import os
import sqlite3
import subprocess
import statistics
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import click
import requests
from flask import Flask, Response, abort, g, jsonify, render_template, request, url_for

from translations import DEFAULT_LANGUAGE, normalize_language, translate

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MONITOR_DATA_DIR", BASE_DIR / "data"))
RRD_PATH = DATA_DIR / "ollama_cloud.rrd"
STATUS_PATH = DATA_DIR / "status.json"
MODEL_INFO_PATH = DATA_DIR / "model_info.json"
SUMMARY_DB_PATH = DATA_DIR / "summaries.sqlite3"
LOCK_PATH = DATA_DIR / "probe.lock"
SUMMARY_LOCK_PATH = DATA_DIR / "summary.lock"
API_BASE = os.environ.get("OLLAMA_API_BASE", "https://ollama.com").rstrip("/")

MODELS = [
    {"name": "gemma4:31b", "ds": "gemma4_31b", "color": "7C3AED"},
    {"name": "minimax-m3", "ds": "minimax_m3", "color": "06B6D4"},
    {"name": "glm-5.2", "ds": "glm_5_2", "color": "22C55E"},
    {"name": "deepseek-v4-pro", "ds": "dsv4_pro", "color": "F59E0B"},
    {"name": "deepseek-v4-flash", "ds": "dsv4_flash", "color": "EF4444"},
    {"name": "nemotron-3-ultra", "ds": "nemotron3", "color": "EC4899"},
]
MODEL_BY_NAME = {item["name"]: item for item in MODELS}

PERIODS = {
    "24h": ("end-24h", "Last 24 hours"),
    "7d": ("end-7d", "Last 7 days"),
    "30d": ("end-30d", "Last 30 days"),
}
PERIOD_SECONDS = {"24h": 86_400, "7d": 604_800, "30d": 2_592_000}
SUMMARY_MODEL = "glm-5.2"
SUMMARY_WINDOW_SECONDS = 4 * 60 * 60
SUMMARY_EXPECTED_SAMPLES = SUMMARY_WINDOW_SECONDS // 300

PROMPT = (
    "In 100 to 120 words, explain why reproducible performance benchmarks "
    "are useful. Return plain prose only, with no heading or bullet list."
)


def run_rrd(*args: str, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    """Run rrdtool with a predictable locale."""
    env = {**os.environ, "LC_ALL": "C"}
    return subprocess.run(
        ["/usr/bin/rrdtool", *args],
        check=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )


def ensure_rrd() -> None:
    """Create the five-minute-step round-robin database if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RRD_PATH.exists():
        return

    args = ["create", str(RRD_PATH), "--step", "300", "--start", "now-10s"]
    args.extend(
        f"DS:{model['ds']}:GAUGE:900:0:U" for model in MODELS
    )
    # Five-minute detail for 31 days, hourly averages for a year,
    # six-hour averages for a year, and daily averages for two years.
    args.extend(
        [
            "RRA:LAST:0.5:1:8928",
            "RRA:AVERAGE:0.5:1:8928",
            "RRA:MAX:0.5:1:8928",
            "RRA:AVERAGE:0.5:12:8784",
            "RRA:MAX:0.5:12:8784",
            "RRA:AVERAGE:0.5:72:1464",
            "RRA:MAX:0.5:72:1464",
            "RRA:AVERAGE:0.5:288:732",
            "RRA:MAX:0.5:288:732",
        ]
    )
    run_rrd(*args)


@contextmanager
def probe_lock() -> Iterator[None]:
    """Prevent overlapping manual and cron probes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise click.ClickException("another probe is already running") from exc
        yield


@contextmanager
def summary_lock() -> Iterator[None]:
    """Prevent overlapping scheduled and manual summary generation."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise click.ClickException("another summary generation is already running") from exc
        yield


def measure_model(session: requests.Session, model_name: str, api_key: str) -> dict[str, Any]:
    """Call Ollama Cloud and calculate server-reported output token throughput."""
    started = time.monotonic()
    response = session.post(
        f"{API_BASE}/api/generate",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "prompt": PROMPT,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 160},
        },
        timeout=(20, 900),
    )
    wall_seconds = time.monotonic() - started

    if not response.ok:
        body = response.text.replace("\n", " ").strip()[:300]
        raise RuntimeError(f"HTTP {response.status_code}: {body or response.reason}")

    payload = response.json()
    eval_count = int(payload.get("eval_count") or 0)
    eval_duration_ns = int(payload.get("eval_duration") or 0)
    total_duration_ns = int(payload.get("total_duration") or 0)
    # Local Ollama reports eval_duration; Ollama Cloud currently omits it but
    # provides total_duration. In that case, use server-side end-to-end time,
    # which includes prompt processing but excludes client network overhead.
    metric_duration_ns = eval_duration_ns or total_duration_ns
    metric_source = "eval_duration" if eval_duration_ns else "total_duration"
    if eval_count <= 0 or metric_duration_ns <= 0:
        raise RuntimeError("Ollama response did not include usable token/duration metrics")

    tps = eval_count * 1_000_000_000 / metric_duration_ns
    return {
        "tps": round(tps, 3),
        "eval_count": eval_count,
        "metric_seconds": round(metric_duration_ns / 1_000_000_000, 3),
        "metric_source": metric_source,
        "wall_seconds": round(wall_seconds, 3),
        "done_reason": payload.get("done_reason", "stop"),
    }


def format_parameter_count(value: Any) -> str | None:
    """Format an exact API parameter count as a compact human label."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    if count >= 1_000_000_000:
        amount = f"{count / 1_000_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{amount}B parameters"
    if count >= 1_000_000:
        amount = f"{count / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{amount}M parameters"
    return f"{count:,} parameters"


def localized_parameter_count(value: Any, language: str) -> str | None:
    """Format a model parameter count for the active UI language."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    if count >= 1_000_000_000:
        amount = f"{count / 1_000_000_000:.1f}".rstrip("0").rstrip(".") + "B"
    elif count >= 1_000_000:
        amount = f"{count / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    else:
        amount = f"{count:,}"
    return translate(language, "model.parameter_count_value", value=amount)


def localized_capability(value: Any, language: str) -> str:
    """Translate known Ollama capability labels and preserve unknown values."""
    capability = str(value)
    key = f"capability.{capability.lower()}"
    translated = translate(language, key)
    return capability if translated == key else translated


def model_info_value(model_info: dict[str, Any], suffix: str) -> Any:
    """Find architecture-prefixed fields such as *.context_length."""
    return next((value for key, value in model_info.items() if key.endswith(suffix)), None)


def normalize_model_info(model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only useful, non-sensitive fields from Ollama /api/show."""
    details = payload.get("details") or {}
    raw_info = payload.get("model_info") or {}
    parameter_count = details.get("parameter_size") or raw_info.get("general.parameter_count")
    context_length = model_info_value(raw_info, ".context_length")
    embedding_length = model_info_value(raw_info, ".embedding_length")
    return {
        "name": model_name,
        "architecture": raw_info.get("general.architecture") or details.get("family"),
        "family": details.get("family"),
        "parameter_count": int(parameter_count) if str(parameter_count).isdigit() and int(parameter_count) > 0 else None,
        "parameter_label": format_parameter_count(parameter_count),
        "quantization": details.get("quantization_level") or raw_info.get("general.quantization_version"),
        "context_length": int(context_length) if context_length is not None else None,
        "embedding_length": int(embedding_length) if embedding_length is not None else None,
        "format": details.get("format") or None,
        "capabilities": payload.get("capabilities") or [],
        "modified_at": payload.get("modified_at"),
    }


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON in the monitor data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".json", dir=DATA_DIR)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_model_info() -> dict[str, Any]:
    if not MODEL_INFO_PATH.exists():
        return {"models": {}}
    try:
        return json.loads(MODEL_INFO_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"models": {}}


def model_info_is_stale(max_age_seconds: int = 86_400) -> bool:
    return not MODEL_INFO_PATH.exists() or time.time() - MODEL_INFO_PATH.stat().st_mtime > max_age_seconds


def refresh_model_info(api_key: str, session: requests.Session | None = None) -> dict[str, Any]:
    """Refresh the cached model basics using Ollama's /api/show endpoint."""
    owned_session = session is None
    session = session or requests.Session()
    existing = load_model_info().get("models", {})
    models = dict(existing)
    errors: dict[str, str] = {}
    try:
        for model in MODELS:
            name = model["name"]
            try:
                response = session.post(
                    f"{API_BASE}/api/show",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": name},
                    timeout=(20, 120),
                )
                if not response.ok:
                    body = response.text.replace("\n", " ").strip()[:300]
                    raise RuntimeError(f"HTTP {response.status_code}: {body or response.reason}")
                models[name] = normalize_model_info(name, response.json())
            except Exception as exc:
                errors[name] = str(exc)[:500]
    finally:
        if owned_session:
            session.close()

    cache = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "errors": errors,
    }
    write_json_file(MODEL_INFO_PATH, cache)
    return cache


def write_status(status: dict[str, Any]) -> None:
    """Atomically publish dashboard status."""
    write_json_file(STATUS_PATH, status)


def load_status() -> dict[str, Any]:
    if not STATUS_PATH.exists():
        return {"state": "waiting", "models": {}}
    try:
        return json.loads(STATUS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"state": "error", "models": {}, "error": "status file is unavailable"}


def update_rrd(results: dict[str, dict[str, Any]]) -> int:
    """Store one timestamp containing a value (or unknown) for every model."""
    ensure_rrd()
    now = int(time.time())
    last = int(run_rrd("last", str(RRD_PATH), capture=True).stdout.decode().strip())
    timestamp = max(now, last + 1)
    values = []
    for model in MODELS:
        result = results.get(model["name"], {})
        value = result.get("tps")
        values.append("U" if value is None else f"{float(value):.6f}")
    run_rrd("update", str(RRD_PATH), f"{timestamp}:{':'.join(values)}")
    return timestamp


def perform_probe() -> dict[str, Any]:
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise click.ClickException("OLLAMA_API_KEY is not set")

    ensure_rrd()
    started_at = datetime.now(timezone.utc)
    results: dict[str, dict[str, Any]] = {}
    with probe_lock(), requests.Session() as session:
        if model_info_is_stale():
            click.echo("Refreshing Ollama model information ...")
            info_cache = refresh_model_info(api_key, session)
            if info_cache.get("errors"):
                click.echo(f"Model information warnings: {len(info_cache['errors'])}")
        for model in MODELS:
            name = model["name"]
            click.echo(f"Testing {name} ... ", nl=False)
            try:
                result = measure_model(session, name, api_key)
                result["ok"] = True
                results[name] = result
                click.echo(f"{result['tps']:.2f} token/s")
            except Exception as exc:  # Continue so one unavailable model cannot stop the run.
                results[name] = {"ok": False, "tps": None, "error": str(exc)[:500]}
                click.echo(f"ERROR: {exc}")

        rrd_timestamp = update_rrd(results)
        finished_at = datetime.now(timezone.utc)
        success_count = sum(1 for item in results.values() if item.get("ok"))
        status = {
            "state": "ok" if success_count == len(MODELS) else "partial",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "rrd_timestamp": rrd_timestamp,
            "success_count": success_count,
            "model_count": len(MODELS),
            "models": results,
        }
        write_status(status)
    return status


def percentile(values: list[float], percentile_value: float) -> float | None:
    """Return an interpolated percentile for a non-empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def fetch_model_values(model_ds: str, period: str) -> list[float]:
    """Fetch finite five-minute AVERAGE values for one model from RRDtool."""
    if model_ds not in {model["ds"] for model in MODELS} or period not in PERIODS:
        raise ValueError("invalid model or period")
    ensure_rrd()
    output = run_rrd(
        "fetch",
        str(RRD_PATH),
        "AVERAGE",
        "--start",
        PERIODS[period][0],
        "--end",
        "now",
        "--resolution",
        "300",
        capture=True,
    ).stdout.decode(errors="replace")

    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return []
    columns = lines[0].split()
    try:
        model_index = columns.index(model_ds)
    except ValueError as exc:
        raise RuntimeError(f"RRD data source not found: {model_ds}") from exc

    values: list[float] = []
    for line in lines[1:]:
        if ":" not in line:
            continue
        fields = line.split(":", 1)[1].split()
        if model_index >= len(fields):
            continue
        try:
            value = float(fields[model_index])
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return values


def calculate_period_stats(model_ds: str, period: str) -> dict[str, Any]:
    """Calculate descriptive and operational statistics for a graph range."""
    values = fetch_model_values(model_ds, period)
    expected_samples = PERIOD_SECONDS[period] // 300
    count = len(values)
    if not values:
        return {
            "period": period,
            "label": PERIODS[period][1],
            "count": 0,
            "expected_samples": expected_samples,
            "coverage_pct": 0.0,
        }

    average = statistics.fmean(values)
    deviation = statistics.pstdev(values) if count > 1 else 0.0
    midpoint = max(1, count // 2)
    earlier = statistics.fmean(values[:midpoint])
    later = statistics.fmean(values[midpoint:]) if values[midpoint:] else values[-1]
    trend_pct = ((later - earlier) / earlier * 100) if earlier > 0 else None
    return {
        "period": period,
        "label": PERIODS[period][1],
        "latest": values[-1],
        "average": average,
        "minimum": min(values),
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
        "range": max(values) - min(values),
        "stddev": deviation,
        "cv_pct": (deviation / average * 100) if average > 0 else None,
        "trend_pct": trend_pct,
        "count": count,
        "expected_samples": expected_samples,
        "coverage_pct": min(100.0, count / expected_samples * 100),
    }


def get_model_stats(model_ds: str) -> list[dict[str, Any]]:
    """Return statistics ordered to match the dashboard graph ranges."""
    return [calculate_period_stats(model_ds, period) for period in PERIODS]


def fetch_summary_snapshot() -> dict[str, Any]:
    """Fetch aligned five-minute values for all models over the last four hours."""
    ensure_rrd()
    period_end = int(time.time())
    period_start = period_end - SUMMARY_WINDOW_SECONDS
    output = run_rrd(
        "fetch",
        str(RRD_PATH),
        "AVERAGE",
        "--start",
        str(period_start),
        "--end",
        str(period_end),
        "--resolution",
        "300",
        capture=True,
    ).stdout.decode(errors="replace")
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("RRDtool returned no data")
    columns = lines[0].split()
    rows: list[tuple[int, list[str]]] = []
    for line in lines[1:]:
        if ":" not in line:
            continue
        raw_timestamp, raw_values = line.split(":", 1)
        try:
            timestamp = int(raw_timestamp.strip())
        except ValueError:
            continue
        if period_start <= timestamp <= period_end:
            rows.append((timestamp, raw_values.split()))

    model_data: dict[str, dict[str, Any]] = {}
    valid_point_count = 0
    for model in MODELS:
        try:
            model_index = columns.index(model["ds"])
        except ValueError as exc:
            raise RuntimeError(f"RRD data source not found: {model['ds']}") from exc
        points: list[dict[str, Any]] = []
        values: list[float] = []
        for timestamp, fields in rows:
            if model_index >= len(fields):
                continue
            try:
                value = float(fields[model_index])
            except ValueError:
                continue
            if not math.isfinite(value) or value < 0:
                continue
            rounded = round(value, 2)
            values.append(value)
            points.append(
                {
                    "time_utc": datetime.fromtimestamp(timestamp, timezone.utc).strftime("%H:%M"),
                    "tps": rounded,
                }
            )
        valid_point_count += len(values)
        if not values:
            model_data[model["name"]] = {
                "sample_count": 0,
                "expected_samples": SUMMARY_EXPECTED_SAMPLES,
                "coverage_pct": 0.0,
                "points": [],
            }
            continue
        midpoint = max(1, len(values) // 2)
        first_half = statistics.fmean(values[:midpoint])
        second_half = statistics.fmean(values[midpoint:]) if values[midpoint:] else values[-1]
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        model_data[model["name"]] = {
            "sample_count": len(values),
            "expected_samples": SUMMARY_EXPECTED_SAMPLES,
            "coverage_pct": round(min(100.0, len(values) / SUMMARY_EXPECTED_SAMPLES * 100), 1),
            "latest_tps": round(values[-1], 2),
            "average_tps": round(statistics.fmean(values), 2),
            "minimum_tps": round(min(values), 2),
            "maximum_tps": round(max(values), 2),
            "p95_tps": round(percentile(values, 0.95) or 0, 2),
            "stddev_tps": round(deviation, 2),
            "cv_pct": round(deviation / statistics.fmean(values) * 100, 1),
            "trend_pct": round((second_half - first_half) / first_half * 100, 1) if first_half > 0 else None,
            "points": points,
        }

    if valid_point_count == 0:
        raise RuntimeError("no valid performance points exist in the last four hours")
    return {
        "period_start": datetime.fromtimestamp(period_start, timezone.utc).isoformat(),
        "period_end": datetime.fromtimestamp(period_end, timezone.utc).isoformat(),
        "interval_minutes": 5,
        "expected_samples_per_model": SUMMARY_EXPECTED_SAMPLES,
        "valid_point_count": valid_point_count,
        "coverage_pct": round(
            min(100.0, valid_point_count / (SUMMARY_EXPECTED_SAMPLES * len(MODELS)) * 100), 1
        ),
        "models": model_data,
    }


def summary_prompt(snapshot: dict[str, Any]) -> str:
    """Build a compact, data-grounded instruction for the analyst model."""
    return (
        "You are an operations analyst reviewing Ollama Cloud output-token throughput. "
        "Using only the supplied rolling four-hour dataset, write one concise, useful plain-text "
        "summary of 70 to 120 words. Mention the strongest and weakest model based on average "
        "throughput, the most operationally significant trend or volatility, and any missing-data "
        "limitation. Include useful numbers with token/s units. Do not add a title, bullet list, "
        "markdown, generic benchmarking advice, unsupported explanations, or claims of statistical "
        "significance. Check every sample-count and percentage statement directly against the JSON. "
        "Do not mention analysis instructions or sample thresholds. The points arrays are ordered "
        "five-minute observations in UTC and the calculated fields are provided for verification.\n\n"
        f"DATASET:\n{json.dumps(snapshot, separators=(',', ':'))}"
    )


def init_summary_db() -> None:
    """Initialize durable hourly-summary history."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SUMMARY_DB_PATH, timeout=15) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                model TEXT NOT NULL,
                summary TEXT NOT NULL,
                valid_point_count INTEGER NOT NULL,
                coverage_pct REAL NOT NULL,
                snapshot_json TEXT NOT NULL,
                prompt_eval_count INTEGER,
                eval_count INTEGER,
                total_duration_ns INTEGER,
                wall_seconds REAL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS summaries_generated_at_idx ON summaries(generated_at DESC)"
        )


def save_summary(snapshot: dict[str, Any], text: str, payload: dict[str, Any], wall_seconds: float) -> int:
    """Persist one generated insight and its aggregate source snapshot."""
    init_summary_db()
    aggregate_snapshot = {
        **{key: value for key, value in snapshot.items() if key != "models"},
        "models": {
            name: {key: value for key, value in data.items() if key != "points"}
            for name, data in snapshot["models"].items()
        },
    }
    with sqlite3.connect(SUMMARY_DB_PATH, timeout=15) as connection:
        connection.execute("PRAGMA busy_timeout=15000")
        cursor = connection.execute(
            """
            INSERT INTO summaries (
                generated_at, period_start, period_end, model, summary,
                valid_point_count, coverage_pct, snapshot_json,
                prompt_eval_count, eval_count, total_duration_ns, wall_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                snapshot["period_start"],
                snapshot["period_end"],
                SUMMARY_MODEL,
                text,
                snapshot["valid_point_count"],
                snapshot["coverage_pct"],
                json.dumps(aggregate_snapshot, separators=(",", ":")),
                payload.get("prompt_eval_count"),
                payload.get("eval_count"),
                payload.get("total_duration"),
                round(wall_seconds, 3),
            ),
        )
        return int(cursor.lastrowid)


def format_utc_timestamp(value: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Format an ISO timestamp for the selected UI language."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        if language == "zh-CN":
            return f"{parsed.year}年{parsed.month}月{parsed.day}日 · {parsed:%H:%M} UTC"
        return parsed.strftime("%b %d, %Y · %H:%M UTC")
    except (AttributeError, TypeError, ValueError):
        return value


def list_summaries(
    limit: int = 24, offset: int = 0, language: str = DEFAULT_LANGUAGE
) -> list[dict[str, Any]]:
    """Read newest summary history for the web feed or CLI."""
    init_summary_db()
    safe_limit = max(1, min(int(limit), 500))
    safe_offset = max(0, int(offset))
    with sqlite3.connect(SUMMARY_DB_PATH, timeout=15) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        rows = connection.execute(
            """
            SELECT id, generated_at, period_start, period_end, model, summary,
                   valid_point_count, coverage_pct, prompt_eval_count,
                   eval_count, total_duration_ns, wall_seconds
            FROM summaries ORDER BY id DESC LIMIT ? OFFSET ?
            """,
            (safe_limit, safe_offset),
        ).fetchall()
    items = [dict(row) for row in rows]
    for item in items:
        item["generated_label"] = format_utc_timestamp(item["generated_at"], language)
        separator = " 至 " if language == "zh-CN" else " to "
        item["period_label"] = (
            f"{format_utc_timestamp(item['period_start'], language)}{separator}"
            f"{format_utc_timestamp(item['period_end'], language)}"
        )
    return items


def count_summaries() -> int:
    """Return the number of retained AI summaries."""
    init_summary_db()
    with sqlite3.connect(SUMMARY_DB_PATH, timeout=15) as connection:
        connection.execute("PRAGMA busy_timeout=15000")
        return int(connection.execute("SELECT COUNT(*) FROM summaries").fetchone()[0])


def generate_hourly_summary() -> dict[str, Any]:
    """Ask GLM-5.2 to summarize the rolling four-hour RRD dataset."""
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise click.ClickException("OLLAMA_API_KEY is not set")
    with summary_lock():
        snapshot = fetch_summary_snapshot()
        started = time.monotonic()
        response = requests.post(
            f"{API_BASE}/api/generate",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": SUMMARY_MODEL,
                "prompt": summary_prompt(snapshot),
                "stream": False,
                "think": False,
                "options": {"temperature": 0.2, "num_predict": 300},
            },
            timeout=(20, 900),
        )
        wall_seconds = time.monotonic() - started
        if not response.ok:
            body = response.text.replace("\n", " ").strip()[:500]
            raise RuntimeError(f"Ollama summary HTTP {response.status_code}: {body or response.reason}")
        payload = response.json()
        text = str(payload.get("response") or "").strip()
        if not text:
            raise RuntimeError("GLM-5.2 returned an empty summary")
        # Bound accidental verbosity while preserving the model's complete concise response.
        text = text[:3000]
        summary_id = save_summary(snapshot, text, payload, wall_seconds)
        return {
            "id": summary_id,
            "model": SUMMARY_MODEL,
            "summary": text,
            "valid_point_count": snapshot["valid_point_count"],
            "coverage_pct": snapshot["coverage_pct"],
            "wall_seconds": round(wall_seconds, 3),
        }


def make_graph(
    period: str, selected_ds: str | None = None, language: str = DEFAULT_LANGUAGE
) -> bytes:
    """Render a PNG through RRDtool's rrdgraph implementation."""
    if period not in PERIODS:
        raise ValueError("invalid graph period")
    graph_models = MODELS
    if selected_ds is not None:
        graph_models = [model for model in MODELS if model["ds"] == selected_ds]
        if not graph_models:
            raise ValueError("invalid model")
    ensure_rrd()
    start = PERIODS[period][0]
    period_name = translate(language, f"period.{period}")
    graph_title = translate(
        language,
        "chart.model_title" if selected_ds else "chart.all_title",
        **({"model": graph_models[0]["name"], "period": period_name} if selected_ds else {"period": period_name}),
    )
    graph_font = "WenQuanYi Zen Hei" if language == "zh-CN" else "DejaVu Sans"
    args = [
        "graph",
        "-",
        "--imgformat",
        "PNG",
        "--start",
        start,
        "--end",
        "now",
        "--width",
        "1080",
        "--height",
        "360",
        "--title",
        graph_title,
        "--vertical-label",
        translate(language, "chart.vertical_label"),
        "--lower-limit",
        "0",
        "--alt-autoscale-max",
        "--slope-mode",
        "--border",
        "0",
        "--font",
        f"DEFAULT:0:{graph_font}",
        "--color",
        "BACK#111827",
        "--color",
        "CANVAS#111827",
        "--color",
        "FONT#CBD5E1",
        "--color",
        "AXIS#64748B",
        "--color",
        "GRID#334155",
        "--color",
        "MGRID#475569",
        "--color",
        "ARROW#94A3B8",
        "--color",
        "SHADEA#111827",
        "--color",
        "SHADEB#111827",
    ]
    for model in graph_models:
        args.append(f"DEF:{model['ds']}={RRD_PATH}:{model['ds']}:AVERAGE")
    for model in graph_models:
        escaped_name = model["name"].replace(":", r"\:")
        ds = model["ds"]
        args.extend(
            [
                f"LINE2:{ds}#{model['color']}:{escaped_name}",
                f"GPRINT:{ds}:LAST:{translate(language, 'chart.legend_last')}\\: %6.1lf",
                f"GPRINT:{ds}:AVERAGE:{translate(language, 'chart.legend_average')}\\: %6.1lf",
                f"GPRINT:{ds}:MAX:{translate(language, 'chart.legend_maximum')}\\: %6.1lf\\l",
            ]
        )
    return run_rrd(*args, capture=True).stdout


def create_app() -> Flask:
    app = Flask(__name__)

    @app.before_request
    def select_ui_language() -> None:
        """Select a locale from the URL, saved preference, or browser headers."""
        requested = normalize_language(request.args.get("lang"))
        saved = normalize_language(request.cookies.get("ollama_monitor_language"))
        accepted = next(
            (
                locale
                for browser_language, quality in request.accept_languages
                if quality > 0 and (locale := normalize_language(browser_language)) is not None
            ),
            None,
        )
        g.language = requested or saved or accepted or DEFAULT_LANGUAGE

    @app.after_request
    def apply_language_response(response: Response) -> Response:
        if response.mimetype in {"text/html", "image/png"}:
            response.headers["Content-Language"] = g.get("language", DEFAULT_LANGUAGE)
            response.vary.add("Accept-Language")
            response.vary.add("Cookie")
        requested = normalize_language(request.args.get("lang"))
        if requested and response.mimetype == "text/html":
            response.set_cookie(
                "ollama_monitor_language",
                requested,
                max_age=31_536_000,
                secure=request.is_secure,
                httponly=True,
                samesite="Lax",
            )
        return response

    def language_url(language: str) -> str:
        """Build a locale switch URL while retaining this page and its query state."""
        locale = normalize_language(language) or DEFAULT_LANGUAGE
        values = request.args.to_dict(flat=True)
        values["lang"] = locale
        values.update(request.view_args or {})
        return url_for(request.endpoint or "index", **values)

    @app.context_processor
    def inject_i18n() -> dict[str, Any]:
        language = g.get("language", DEFAULT_LANGUAGE)
        return {
            "current_language": language,
            "html_language": language,
            "language_url": language_url,
            "t": lambda key, **values: translate(language, key, **values),
            "period_label": lambda period: translate(language, f"period.{period}"),
            "format_timestamp": lambda value: format_utc_timestamp(value, language) if value else translate(language, "common.not_run_yet"),
            "format_parameter_count_ui": lambda value: localized_parameter_count(value, language),
            "capability_label": lambda value: localized_capability(value, language),
        }

    @app.errorhandler(404)
    def page_not_found(_error: Exception) -> tuple[str, int]:
        return render_template(
            "error.html",
            status_code=404,
            error_title=translate(g.language, "error.not_found"),
            error_message=translate(g.language, "error.not_found_message"),
        ), 404

    @app.errorhandler(500)
    def internal_server_error(_error: Exception) -> tuple[str, int]:
        return render_template(
            "error.html",
            status_code=500,
            error_title=translate(g.language, "error.server"),
            error_message=translate(g.language, "error.server_message"),
        ), 500

    @app.get("/")
    def index() -> str:
        status = load_status()
        updated = status.get("finished_at")
        display_models = []
        for model in MODELS:
            display_models.append({**model, **status.get("models", {}).get(model["name"], {})})
        cache_key = int(STATUS_PATH.stat().st_mtime) if STATUS_PATH.exists() else 0
        summary_feed = list_summaries(1, language=g.language)
        return render_template(
            "index.html",
            models=display_models,
            summaries=summary_feed,
            status=status,
            updated=updated,
            cache_key=cache_key,
            periods=PERIODS,
        )

    @app.get("/insights")
    def insights() -> str:
        per_page = 50
        page = request.args.get("page", default=1, type=int) or 1
        page = max(1, page)
        total = count_summaries()
        total_pages = max(1, math.ceil(total / per_page))
        if page > total_pages:
            abort(404)
        feed = list_summaries(per_page, (page - 1) * per_page, g.language)
        return render_template(
            "insights.html",
            summaries=feed,
            page=page,
            total=total,
            total_pages=total_pages,
        )

    @app.get("/model/<model_id>")
    def model_detail(model_id: str) -> str:
        model = next((item for item in MODELS if item["ds"] == model_id), None)
        if model is None:
            abort(404)
        status = load_status()
        display_model = {**model, **status.get("models", {}).get(model["name"], {})}
        model_info_cache = load_model_info()
        basic_info = model_info_cache.get("models", {}).get(model["name"], {})
        cache_key = int(STATUS_PATH.stat().st_mtime) if STATUS_PATH.exists() else 0
        try:
            performance_stats = get_model_stats(model["ds"])
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace")[:500]
            app.logger.error("RRD statistics fetch failed: %s", detail)
            performance_stats = []
        return render_template(
            "model.html",
            model=display_model,
            all_models=MODELS,
            model_info=basic_info,
            model_info_fetched_at=model_info_cache.get("fetched_at"),
            status=status,
            stats=performance_stats,
            updated=status.get("finished_at"),
            cache_key=cache_key,
            periods=PERIODS,
        )

    @app.get("/graph/<period>/<model_id>.png")
    def model_graph(period: str, model_id: str) -> Response:
        if period not in PERIODS or not any(item["ds"] == model_id for item in MODELS):
            return Response("Unknown graph or model\n", status=404, mimetype="text/plain")
        try:
            png = make_graph(period, model_id, g.language)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace")[:500]
            app.logger.error("model rrdgraph failed: %s", detail)
            return Response("Graph unavailable\n", status=503, mimetype="text/plain")
        return Response(
            png,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=300"},
        )

    @app.get("/graph/<period>.png")
    def graph(period: str) -> Response:
        if period not in PERIODS:
            return Response("Unknown graph period\n", status=404, mimetype="text/plain")
        try:
            png = make_graph(period, language=g.language)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace")[:500]
            app.logger.error("rrdgraph failed: %s", detail)
            return Response("Graph unavailable\n", status=503, mimetype="text/plain")
        return Response(
            png,
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=300"},
        )

    @app.get("/api/summaries")
    def api_summaries() -> Response:
        return jsonify({"summaries": list_summaries(100)})

    @app.get("/api/status")
    def api_status() -> Response:
        return jsonify(load_status())

    @app.get("/healthz")
    def healthz() -> Response:
        return jsonify({"ok": True, "rrd": RRD_PATH.exists(), "summary_db": SUMMARY_DB_PATH.exists()})

    return app


@click.group()
def cli() -> None:
    """Monitor Ollama Cloud output-token throughput."""


@cli.command("init-db")
def init_db_command() -> None:
    """Create the RRD database."""
    ensure_rrd()
    click.echo(f"RRD ready: {RRD_PATH}")


@cli.command("probe")
def probe_command() -> None:
    """Benchmark every configured model and update the RRD."""
    status = perform_probe()
    if status["success_count"] == 0:
        raise click.ClickException("all model probes failed")
    click.echo(f"Completed: {status['success_count']}/{status['model_count']} models succeeded")


@cli.command("refresh-info")
def refresh_info_command() -> None:
    """Fetch and cache basic model information from Ollama /api/show."""
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise click.ClickException("OLLAMA_API_KEY is not set")
    cache = refresh_model_info(api_key)
    success_count = len(cache.get("models", {}))
    error_count = len(cache.get("errors", {}))
    click.echo(f"Model information refreshed: {success_count} cached, {error_count} errors")
    if success_count == 0:
        raise click.ClickException("no model information could be fetched")


@cli.command("summarize")
def summarize_command() -> None:
    """Generate and save a GLM-5.2 summary of the last four hours."""
    result = generate_hourly_summary()
    click.echo(
        f"Saved summary {result['id']} from {result['valid_point_count']} points "
        f"({result['coverage_pct']:.1f}% coverage)"
    )
    click.echo(result["summary"])


@cli.command("summary-history")
@click.option("--limit", default=10, type=click.IntRange(1, 500), show_default=True)
def summary_history_command(limit: int) -> None:
    """Print saved AI summary history as JSON."""
    click.echo(json.dumps(list_summaries(limit), indent=2))


@cli.command("graph")
@click.option("--period", type=click.Choice(list(PERIODS)), default="24h", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def graph_command(period: str, output: Path) -> None:
    """Render an RRD graph to a PNG file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(make_graph(period))
    click.echo(f"Wrote {output}")


@cli.command("status")
def status_command() -> None:
    """Print the latest probe status as JSON."""
    click.echo(json.dumps(load_status(), indent=2))


@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
def serve_command(host: str, port: int) -> None:
    """Run Flask's development server (systemd uses Gunicorn)."""
    create_app().run(host=host, port=port)


app = create_app()

if __name__ == "__main__":
    cli()
