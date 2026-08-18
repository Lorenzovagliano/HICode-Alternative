import json
import hashlib
import logging
import os
import re
import threading
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from .concurrency import (
    ConcurrentTaskError,
    run_bounded_tasks,
    write_json_atomic,
)


LABEL_PATTERN = re.compile(r"^\s*LABEL\s*:\s*\[(.*?)\]\s*$", re.IGNORECASE)
LOGGER = logging.getLogger(__name__)
_CLIENT_LOCAL = threading.local()


def make_generation_prompt(background, goal, instructions=None):
    """Build the HiCode label-generation prompt from Appendix A.1."""
    return f"""BACKGROUND:
{background.strip()}

GOAL:
{goal.strip()}

{_format_optional_instructions(instructions)}

For each input segment:
- Only generate codes when the segment contains a meaningful, evidence-grounded signal relevant to the goal.
- Create observational, concise, and clear inductive labels.
- ONLY output labels and DO NOT output explanations.
- Define each label using the format "LABEL: [The phrase of the label]".
- The `[` and `]` characters are literal and mandatory. Valid: "LABEL: [Community support]". Invalid: "LABEL: Community support".
- If there are multiple labels, put each label on a new line.
- If the input is irrelevant, use "LABEL: [Irrelevant]".
- Each label MUST NOT exceed 5 words.
"""


def _format_optional_instructions(instructions):
    """Render profile-specific semantic guidance without changing output rules."""
    if not instructions:
        return ""
    if isinstance(instructions, str):
        instructions = [instructions]
    if not isinstance(instructions, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in instructions
    ):
        raise ValueError("Generation instructions must be a string or non-empty strings.")
    return "\nAdditional task guidance:\n" + "\n".join(
        f"- {item.strip()}" for item in instructions
    )


def clean_label(raw_label):
    """Parse exact HiCode LABEL lines and fail on malformed model output."""
    if not isinstance(raw_label, str) or not raw_label.strip():
        raise ValueError("Label generation returned empty output.")

    labels = []
    invalid_lines = []
    for raw_line in raw_label.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LABEL_PATTERN.fullmatch(line)
        if match is None:
            invalid_lines.append(line)
            continue
        label = match.group(1).strip()
        if not label:
            invalid_lines.append(line)
            continue
        labels.append(label)

    if invalid_lines or not labels:
        details = f" Invalid lines: {invalid_lines!r}." if invalid_lines else ""
        raise ValueError(
            "Label generation output did not contain only valid LABEL: [label] lines."
            + details
        )
    return labels


def generate_labels(data_processed, system_prompt, config):
    if config is None:
        config = {}
    model_name = config["model_name"]
    if "gpt" in model_name.lower():
        return generate_labels_gpt(data_processed, system_prompt, config)
    # elif "llama" in model_name.lower():
    #     return generate_labels_hf(data_processed, system_prompt, config)
    raise ValueError(f"Model {model_name} is not supported.")


def generate_labels_gpt(data_processed, system_prompt, config):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before label generation.")
    max_attempts = config.get("max_generation_attempts", 3)
    if max_attempts < 1:
        raise ValueError("max_generation_attempts must be at least 1.")
    max_workers = config.get("max_workers", 1)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")

    segment_ids = list(data_processed.keys())
    LOGGER.info(
        "Generation started: model=%s segments=%d max_attempts=%d reasoning=%s",
        config["model_name"],
        len(segment_ids),
        max_attempts,
        config.get("reasoning_effort"),
    )
    checkpoint_dir = config.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    resume_generation = config.get("resume_generation", False)
    labels_by_index = {}
    missing_tasks = []
    for segment_index, segment_id in enumerate(segment_ids):
        text_to_label = data_processed[segment_id]
        checkpoint_path = (
            _generation_checkpoint_path(checkpoint_dir, segment_index)
            if checkpoint_dir is not None
            else None
        )
        if resume_generation and checkpoint_path is not None and checkpoint_path.is_file():
            labels_by_index[segment_index] = _load_generation_checkpoint(
                checkpoint_path,
                segment_index,
                segment_id,
                text_to_label,
            )
        else:
            missing_tasks.append((segment_index, (segment_id, text_to_label)))

    def generate_one(segment_index, task):
        segment_id, text_to_label = task
        labels = _generate_one_segment(
            segment_id,
            text_to_label,
            system_prompt,
            config,
        )
        if checkpoint_dir is not None:
            write_json_atomic(
                _generation_checkpoint_path(checkpoint_dir, segment_index),
                {
                    "segment_index": segment_index,
                    "segment_id": segment_id,
                    "text_sha256": _text_sha256(text_to_label),
                    "labels": labels,
                },
            )
        return labels

    try:
        labels_by_index.update(
            run_bounded_tasks(
                missing_tasks,
                generate_one,
                max_workers,
                "label generation",
            )
        )
    except ConcurrentTaskError as exc:
        failure_path = config.get("failure_path")
        if failure_path:
            write_json_atomic(failure_path, exc.as_dict())
        raise RuntimeError(str(exc)) from exc

    output = {}
    relevant_segments = 0
    code_occurrences = 0
    for segment_index, segment_id in enumerate(
        tqdm(segment_ids, desc="Assembling initial codes", unit="segment", dynamic_ncols=True),
        start=0,
    ):
        text_to_label = data_processed[segment_id]
        labels = labels_by_index[segment_index]
        if not labels:
            continue

        relevant_segments += 1
        code_occurrences += len(labels)
        record_id = segment_id.rsplit("_", 1)[0] if "_" in segment_id else segment_id
        annotation = {
            "segment_id": segment_id,
            "sentence": text_to_label,
            "label": labels,
        }
        output.setdefault(record_id, {}).setdefault("LLM_Annotation", []).append(annotation)

    failure_path = config.get("failure_path")
    if failure_path and Path(failure_path).is_file():
        Path(failure_path).unlink()
    LOGGER.info(
        "Generation complete: relevant_segments=%d irrelevant_segments=%d code_occurrences=%d",
        relevant_segments,
        len(segment_ids) - relevant_segments,
        code_occurrences,
    )
    return output


def _text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generation_checkpoint_path(checkpoint_dir, segment_index):
    return checkpoint_dir / f"segment_{segment_index:08d}.json"


def _load_generation_checkpoint(path, segment_index, segment_id, text):
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid generation checkpoint {path}: {exc}") from exc
    if checkpoint.get("segment_index") != segment_index:
        raise ValueError(f"Generation checkpoint {path} has the wrong segment index.")
    if checkpoint.get("segment_id") != segment_id:
        raise ValueError(f"Generation checkpoint {path} has the wrong segment ID.")
    if checkpoint.get("text_sha256") != _text_sha256(text):
        raise ValueError(f"Generation checkpoint {path} does not match source text.")
    labels = checkpoint.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"Generation checkpoint {path} has invalid labels.")
    return labels


def clear_generation_checkpoints(checkpoint_dir):
    """Remove completed generation checkpoint files without touching outputs."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return
    for pattern in ("segment_*.json", "segment_*.json.tmp", "failures.json", "failures.json.tmp"):
        for path in checkpoint_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    if not any(checkpoint_dir.iterdir()):
        checkpoint_dir.rmdir()


def _get_openai_client():
    client = getattr(_CLIENT_LOCAL, "client", None)
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        client_kwargs = {"api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        _CLIENT_LOCAL.client = client
    return client


def _generate_one_segment(segment_id, text_to_label, system_prompt, config):
    max_attempts = config.get("max_generation_attempts", 3)
    client = _get_openai_client()
    last_error = None
    labels = None
    for attempt in range(1, max_attempts + 1):
        LOGGER.info(
            "Generation API request: segment=%s attempt=%d/%d",
            segment_id,
            attempt,
            max_attempts,
        )
        messages = [{"role": "developer", "content": system_prompt}]
        if last_error is not None:
            messages.append(
                {
                    "role": "developer",
                    "content": (
                        "The previous response was invalid: "
                        f"{last_error}\nLabel the original segment again. Return only "
                        "one or more lines formatted exactly as LABEL: [label], or "
                        "LABEL: [Irrelevant]. Literal square brackets are mandatory."
                    ),
                }
            )
        messages.append({"role": "user", "content": text_to_label})
        request = {
            "model": config["model_name"],
            "messages": messages,
        }
        reasoning_effort = config.get("reasoning_effort")
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort

        response = client.chat.completions.create(**request)
        raw_label = response.choices[0].message.content
        try:
            labels = clean_label(raw_label)
            break
        except ValueError as exc:
            last_error = str(exc)
            LOGGER.warning(
                "Generation format rejected: segment=%s attempt=%d/%d error=%s",
                segment_id,
                attempt,
                max_attempts,
                last_error,
            )
    if labels is None:
        raise RuntimeError(
            f"Label generation for {segment_id} remained invalid after "
            f"{max_attempts} attempts: {last_error}"
        )
    return [label for label in labels if label.casefold() != "irrelevant"]
