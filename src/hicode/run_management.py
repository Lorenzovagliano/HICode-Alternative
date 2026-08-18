"""Immutable run metadata and prompt-profile helpers for HiCode."""

from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import uuid

from .concurrency import write_json_atomic


RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
PROFILE_REQUIRED_FIELDS = {"name", "background", "research_question", "generation_instructions", "clustering_instructions"}
CLUSTERING_STAGES = ("level_1", "intermediate", "later")
PROFILE_PLACEHOLDER_PATTERN = re.compile(r"<\s*ADD\s+YOUR\s+OWN\b", re.IGNORECASE)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_prompt_profile(path):
    path = Path(path)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Prompt profile does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prompt profile is not valid JSON: {path}: {exc}") from exc
    if not isinstance(profile, dict) or set(profile) != PROFILE_REQUIRED_FIELDS:
        raise ValueError(
            "Prompt profile must contain exactly: "
            + ", ".join(sorted(PROFILE_REQUIRED_FIELDS))
        )
    for key in ("name", "background", "research_question"):
        if not isinstance(profile[key], str) or not profile[key].strip():
            raise ValueError(f"Prompt profile field {key!r} must be a non-empty string.")
        profile[key] = profile[key].strip()
        if PROFILE_PLACEHOLDER_PATTERN.search(profile[key]):
            raise ValueError(
                f"Prompt profile field {key!r} still contains an <ADD YOUR OWN> "
                "placeholder; replace it before running HiCode."
            )
    if not _valid_instruction_list(profile["generation_instructions"]):
        raise ValueError("generation_instructions must be a non-empty list of strings.")
    for stage in CLUSTERING_STAGES:
        if stage not in profile["clustering_instructions"] or not _valid_instruction_list(
            profile["clustering_instructions"][stage]
        ):
            raise ValueError(
                "clustering_instructions must contain non-empty string lists for: "
                + ", ".join(CLUSTERING_STAGES)
            )
    if set(profile["clustering_instructions"]) != set(CLUSTERING_STAGES):
        raise ValueError("clustering_instructions contains unsupported stages.")
    return profile


def _valid_instruction_list(value):
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def resolve_run_directory(runs_dir, run_name):
    if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("run_name must contain only letters, numbers, underscores, and hyphens.")
    root = Path(runs_dir).resolve()
    run_dir = (root / run_name).resolve()
    if run_dir.parent != root:
        raise ValueError("run_name must resolve to a direct child of runs_dir.")
    return root, run_dir


def source_schema(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
    return list(reader.fieldnames)


def git_revision(base_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", str(base_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_manifest(
    *,
    run_name,
    input_csv,
    text_column,
    profile,
    settings,
    base_dir,
    selected_record_ids=None,
):
    input_csv = Path(input_csv).resolve()
    manifest = {
        "schema_version": 2,
        "run_name": run_name,
        "input_csv": str(input_csv),
        "input_sha256": sha256_file(input_csv),
        "source_schema": source_schema(input_csv),
        "text_column": text_column,
        "prompt_profile": profile,
        "prompt_profile_sha256": canonical_json_sha256(profile),
        "settings": settings,
        "git_revision": git_revision(base_dir),
    }
    if selected_record_ids is not None:
        manifest["selected_record_ids"] = sorted(selected_record_ids)
        manifest["selected_record_ids_sha256"] = canonical_json_sha256(
            manifest["selected_record_ids"]
        )
    return manifest


def write_manifest(path, manifest):
    path = Path(path)
    with path.open("x", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, ensure_ascii=False)


def validate_manifest(path, expected):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Cannot resume; run manifest is missing: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "schema_version", "run_name", "input_csv", "input_sha256", "source_schema",
        "text_column", "prompt_profile_sha256", "settings", "selected_record_ids_sha256",
    )
    for key in keys:
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Cannot resume; manifest mismatch for {key}: "
                f"stored={actual.get(key)!r} requested={expected.get(key)!r}."
            )
    return actual


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def start_execution_attempt(
    output_dir, mode, generation_workers, clustering_workers, attempt_id=None
):
    """Record one invocation without making worker counts part of run identity."""
    path = Path(output_dir) / "execution_attempts.json"
    if path.is_file():
        attempts = json.loads(path.read_text(encoding="utf-8"))
    else:
        attempts = []
    if not isinstance(attempts, list):
        raise ValueError(f"Execution attempts file is not a JSON list: {path}")

    attempt_id = attempt_id or uuid.uuid4().hex
    attempts.append(
        {
            "attempt_id": attempt_id,
            "started_at": _utc_now(),
            "mode": mode,
            "generation_workers": generation_workers,
            "clustering_workers": clustering_workers,
            "status": "running",
        }
    )
    write_json_atomic(path, attempts)
    return attempt_id


def finish_execution_attempt(output_dir, attempt_id, status, error=None):
    path = Path(output_dir) / "execution_attempts.json"
    if not path.is_file():
        return
    attempts = json.loads(path.read_text(encoding="utf-8"))
    for attempt in attempts:
        if attempt.get("attempt_id") != attempt_id:
            continue
        attempt["finished_at"] = _utc_now()
        attempt["status"] = status
        if error is not None:
            attempt["error_type"] = type(error).__name__
            attempt["error"] = str(error)[:1000]
        break
    write_json_atomic(path, attempts)
