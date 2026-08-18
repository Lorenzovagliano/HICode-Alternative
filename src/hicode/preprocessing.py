"""Clean arbitrary CSV text columns and turn them into model-sized segments."""

from collections import Counter
import csv
import logging
from pathlib import Path
import re
import statistics
import unicodedata


LOGGER = logging.getLogger(__name__)

TARGET_SEGMENT_CHARS = 450
MAX_SEGMENT_CHARS = 700

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_text(raw_text):
    """Return meaning-preserving, whitespace-normalized text.

    The normalizer deliberately does not remove hashtags, mentions, emoji,
    punctuation, or repeated source records. It only removes formatting noise
    that should not affect qualitative coding.
    """
    if raw_text is None:
        return ""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    text = unicodedata.normalize("NFC", raw_text)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = "".join(
        character
        for character in text
        if character in "\n\r\t"
        or unicodedata.category(character) not in {"Cc", "Cf"}
        or character == "\u200d"
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _split_oversized_text(text, max_chars):
    if len(text) <= max_chars:
        return [text]

    pieces = []
    current = ""
    for word in text.split():
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                word[start : start + max_chars]
                for start in range(0, len(word), max_chars)
            )
            continue
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _pack_units(units, target_chars, max_chars):
    chunks = []
    current = ""
    for raw_unit in units:
        unit = raw_unit.strip()
        if not unit:
            continue
        if not current:
            current = unit
            if len(current) >= target_chars:
                chunks.append(current)
                current = ""
            continue

        candidate = f"{current} {unit}"
        if len(candidate) <= target_chars:
            current = candidate
        elif len(candidate) <= max_chars and abs(len(candidate) - target_chars) <= abs(
            len(current) - target_chars
        ):
            chunks.append(candidate)
            current = ""
        else:
            chunks.append(current)
            current = unit
            if len(current) >= target_chars:
                chunks.append(current)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def segment_text(
    text,
    target_chars=TARGET_SEGMENT_CHARS,
    max_chars=MAX_SEGMENT_CHARS,
):
    """Split normalized text into non-empty, model-sized chunks."""
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    units = []
    for paragraph in text.splitlines():
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY_PATTERN.split(paragraph)
            if sentence.strip()
        ]
        for sentence in sentences:
            units.extend(_split_oversized_text(sentence, max_chars))
    return _pack_units(units, target_chars, max_chars)


def _record_segments(text, record_id):
    return [
        {
            "record_id": record_id,
            "segment_id": f"{record_id}_{index}",
            "segment_type": "text",
            "text": segment,
        }
        for index, segment in enumerate(segment_text(text))
    ]


def preprocess_records(
    input_csv,
    max_usable_records,
    text_column,
    include_record_ids=None,
):
    """Load a CSV, normalize its selected text column, and create segments."""
    input_csv = Path(input_csv)
    LOGGER.info("Preprocessing input CSV: %s", input_csv)
    with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(f"{input_csv} must contain a {text_column!r} column.")
        source_rows = list(reader)
    LOGGER.info(
        "Input loaded: rows=%d usable_record_limit=%s",
        len(source_rows),
        max_usable_records if max_usable_records is not None else "none (full corpus)",
    )

    cleaned_records = []
    segments = []
    excluded_records = []
    blank_records = 0
    rows_scanned = 0
    cleaning_reason_counts = Counter()

    selected_record_ids = (
        set(include_record_ids) if include_record_ids is not None else None
    )
    for row_index, row in enumerate(source_rows):
        record_id = f"record_{row_index:06d}"
        if selected_record_ids is not None and record_id not in selected_record_ids:
            continue
        if (
            max_usable_records is not None
            and len(cleaned_records) >= max_usable_records
        ):
            break

        rows_scanned += 1
        if rows_scanned == 1 or rows_scanned % 100 == 0:
            LOGGER.info(
                "Preprocessing progress: scanned=%d/%d included=%d excluded=%d blank=%d segments=%d",
                rows_scanned,
                len(source_rows),
                len(cleaned_records),
                len(excluded_records),
                blank_records,
                len(segments),
            )

        raw_text = row.get(text_column)
        text = normalize_text(raw_text)
        if not text:
            blank_records += 1
            continue

        cleaning_reasons = []
        if raw_text != text:
            cleaning_reasons.append("text_normalized")
        cleaning_reason_counts.update(cleaning_reasons)

        record_segments = _record_segments(text, record_id)
        if not record_segments:
            excluded_records.append(
                {
                    "source_row_index": row_index,
                    "record_id": record_id,
                    "exclusion_reason": "empty_after_segmentation",
                    "raw_text": raw_text or "",
                }
            )
            cleaning_reason_counts.update(["empty_after_segmentation"])
            continue

        cleaned_records.append(
            {
                "source_row_index": row_index,
                "record_id": record_id,
                "text": text,
                "cleaning_reasons": ";".join(cleaning_reasons),
            }
        )
        segments.extend(record_segments)

    segment_lengths = [len(segment["text"]) for segment in segments]
    report = {
        "source_rows_total": len(source_rows),
        "source_rows_scanned": rows_scanned,
        "text_column": text_column,
        "selected_records": (
            len(selected_record_ids) if selected_record_ids is not None else None
        ),
        "max_usable_records": max_usable_records,
        "blank_records": blank_records,
        "excluded_records": len(excluded_records),
        "cleaned_records": len(cleaned_records),
        "total_generated_segments": len(segments),
        "average_segment_length": (
            round(statistics.mean(segment_lengths), 2) if segment_lengths else 0
        ),
        "median_segment_length": (
            statistics.median(segment_lengths) if segment_lengths else 0
        ),
        "cleaning_reason_counts": dict(sorted(cleaning_reason_counts.items())),
    }
    LOGGER.info(
        "Preprocessing complete: scanned=%d included=%d excluded=%d blank=%d segments=%d avg_chars=%s median_chars=%s",
        rows_scanned,
        len(cleaned_records),
        len(excluded_records),
        blank_records,
        len(segments),
        report["average_segment_length"],
        report["median_segment_length"],
    )
    return cleaned_records, segments, excluded_records, report
