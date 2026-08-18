"""Run the HiCode inductive-coding pipeline on arbitrary CSV text."""

from collections import defaultdict
import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import sys
import uuid

from tqdm import tqdm

from .run_management import (
    load_prompt_profile,
    make_manifest,
    resolve_run_directory,
    finish_execution_attempt,
    start_execution_attempt,
    validate_manifest,
    write_manifest,
)
from .concurrency import write_json_atomic
from .preprocessing import (
    MAX_SEGMENT_CHARS,
    TARGET_SEGMENT_CHARS,
    preprocess_records,
)
from .result_assembly import (
    MAX_EXAMPLES_PER_THEME,
    build_assignments,
    build_code_paths,
    build_result_tables,
)

BASE_DIR = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


class _TqdmLoggingHandler(logging.Handler):
    """Keep timestamped logs readable while tqdm progress bars are active."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def configure_live_logging():
    """Enable verbose, immediately flushed pipeline progress logging."""
    root_logger = logging.getLogger()
    if not any(getattr(handler, "_hicode_handler", False) for handler in root_logger.handlers):
        handler = _TqdmLoggingHandler()
        handler._hicode_handler = True
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def load_env_file(path=BASE_DIR / ".env"):
    """Load simple KEY=VALUE entries without overriding exported variables."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as env_file:
        for line_number, raw_line in enumerate(env_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                raise ValueError(f"Invalid .env entry on line {line_number}.")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid .env key on line {line_number}: {key!r}")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_env_file()

from .clustering import cluster_labels_gpt, make_clustering_prompt, process_labels
from .generation import (
    clear_generation_checkpoints,
    generate_labels,
    make_generation_prompt,
)


GENERATION_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
CLUSTER_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
GENERATION_REASONING_EFFORT = os.getenv(
    "OPENAI_GENERATION_REASONING_EFFORT", REASONING_EFFORT
)
CLUSTER_REASONING_EFFORT = os.getenv(
    "OPENAI_CLUSTER_REASONING_EFFORT", REASONING_EFFORT
)

CLUSTER_BATCH_SIZE = 100
TARGET_FINAL_THEMES = None
MAX_CLUSTER_ATTEMPTS = 3
MAX_GENERATION_ATTEMPTS = 3
MAX_BATCH_SPLIT_DEPTH = 2

RANDOM_SEED = 42
DEFAULT_GENERATION_WORKERS = 8
DEFAULT_CLUSTERING_WORKERS = 4


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Exported CSV: path=%s rows=%d", path, len(rows))


def _write_json(path, value):
    write_json_atomic(path, value)
    LOGGER.info("Exported JSON: path=%s", path)


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_output_directory(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Preparing output directory: %s", output_dir)
    clustering_dir = output_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Preserving immutable run artifacts and resumable checkpoints")
    return clustering_dir


def _load_resumable_preprocessing(output_dir, max_usable_records, text_column):
    required = (
        "cleaned_records.csv",
        "segments.csv",
        "excluded_records.csv",
        "preprocessing_report.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Cannot resume; missing required preprocessing files: {missing!r}.")
    cleaned_records = _read_csv(output_dir / "cleaned_records.csv")
    segments = _read_csv(output_dir / "segments.csv")
    excluded_records = _read_csv(output_dir / "excluded_records.csv")
    preprocessing_report = _read_json(output_dir / "preprocessing_report.json")
    if preprocessing_report.get("max_usable_records") != max_usable_records:
        raise ValueError(
            "Cannot resume with a different max_usable_records value: "
            f"stored={preprocessing_report.get('max_usable_records')!r}, "
            f"requested={max_usable_records!r}."
        )
    if preprocessing_report.get("text_column", "text") != text_column:
        raise ValueError("Cannot resume with a different text_column value.")
    if len(cleaned_records) != preprocessing_report.get("cleaned_records"):
        raise ValueError("Cannot resume; cleaned-record counts do not reconcile.")
    if len(segments) != preprocessing_report.get("total_generated_segments"):
        raise ValueError("Cannot resume; segment counts do not reconcile.")
    return cleaned_records, segments, excluded_records, preprocessing_report


def _load_resumable_generation(output_dir, max_usable_records, text_column):
    required = (
        "cleaned_records.csv",
        "segments.csv",
        "excluded_records.csv",
        "preprocessing_report.json",
        "generation.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Cannot resume; missing required files: {missing!r}.")

    (
        cleaned_records,
        segments,
        excluded_records,
        preprocessing_report,
    ) = _load_resumable_preprocessing(output_dir, max_usable_records, text_column)
    generation_result = _read_json(output_dir / "generation.json")

    segment_lookup = {segment["segment_id"]: segment for segment in segments}
    if len(segment_lookup) != len(segments):
        raise ValueError("Cannot resume; segment IDs are not unique.")
    for record_id, record in generation_result.items():
        for annotation in record.get("LLM_Annotation", []):
            segment_id = annotation.get("segment_id")
            segment = segment_lookup.get(segment_id)
            if segment is None:
                raise ValueError(
                    f"Cannot resume; generation references unknown segment {segment_id!r}."
                )
            if annotation.get("sentence") != segment["text"]:
                raise ValueError(
                    f"Cannot resume; source text changed for segment {segment_id!r}."
                )
            if record_id != segment["record_id"]:
                raise ValueError(
                    f"Cannot resume; record ID mismatch for segment {segment_id!r}."
                )
    LOGGER.info(
        "Resume validation passed: records=%d segments=%d generated_records=%d",
        len(cleaned_records),
        len(segments),
        len(generation_result),
    )
    return (
        cleaned_records,
        segments,
        excluded_records,
        preprocessing_report,
        generation_result,
    )


def _save_preprocessing_outputs(
    output_dir, cleaned_records, segments, excluded_records, report
):
    output_dir = Path(output_dir)
    _write_csv(
        output_dir / "cleaned_records.csv",
        cleaned_records,
        [
            "source_row_index",
            "record_id",
            "text",
            "cleaning_reasons",
        ],
    )
    _write_csv(
        output_dir / "segments.csv",
        segments,
        ["record_id", "segment_id", "segment_type", "text"],
    )
    _write_csv(
        output_dir / "excluded_records.csv",
        excluded_records,
        ["source_row_index", "record_id", "exclusion_reason", "raw_text"],
    )
    _write_json(output_dir / "preprocessing_report.json", report)


def _pipeline_settings(
    max_usable_records,
    max_cluster_iterations,
    cluster_batch_size=CLUSTER_BATCH_SIZE,
    target_final_themes=TARGET_FINAL_THEMES,
):
    return {
        "generation_model": GENERATION_MODEL,
        "cluster_model": CLUSTER_MODEL,
        "generation_reasoning_effort": GENERATION_REASONING_EFFORT,
        "cluster_reasoning_effort": CLUSTER_REASONING_EFFORT,
        "target_segment_chars": TARGET_SEGMENT_CHARS,
        "max_segment_chars": MAX_SEGMENT_CHARS,
        "cluster_batch_size": cluster_batch_size,
        "target_final_themes": target_final_themes,
        "max_cluster_iterations": max_cluster_iterations,
        "max_cluster_attempts": MAX_CLUSTER_ATTEMPTS,
        "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
        "max_batch_split_depth": MAX_BATCH_SPLIT_DEPTH,
        "random_seed": RANDOM_SEED,
        "max_examples_per_theme": MAX_EXAMPLES_PER_THEME,
        "max_usable_records": max_usable_records,
    }


def _clustering_prompt_factory(profile):
    def prompt_for_iteration(iteration):
        if iteration == 0:
            stage = "level_1"
        elif iteration <= 2:
            stage = "intermediate"
        else:
            stage = "later"
        return make_clustering_prompt(
            goal=profile["research_question"],
            instructions=profile["clustering_instructions"][stage],
        )

    return prompt_for_iteration


def _run_pipeline_impl(
    *,
    run_name,
    mode,
    runs_dir,
    max_usable_records,
    input_csv,
    text_column,
    prompt_file,
    generation_workers,
    clustering_workers,
    cluster_batch_size=CLUSTER_BATCH_SIZE,
    target_final_themes=TARGET_FINAL_THEMES,
    max_cluster_iterations,
    include_record_ids=None,
    attempt_id,
):
    """Run one immutable, named HiCode analysis.

    ``new`` creates a previously absent run directory. ``resume`` only continues
    an incomplete run whose data, prompt profile, and analysis configuration
    match exactly. Worker counts are operational settings and may change on
    resume.
    """
    configure_live_logging()
    LOGGER.info("=" * 72)
    LOGGER.info("HiCode pipeline starting")
    LOGGER.info(
        "Configuration: generation_model=%s clustering_model=%s reasoning=%s target_chars=%d max_chars=%d batch_size=%d target_themes=%s max_cluster_iterations=%d seed=%d generation_workers=%d clustering_workers=%d",
        GENERATION_MODEL,
        CLUSTER_MODEL,
        GENERATION_REASONING_EFFORT,
        TARGET_SEGMENT_CHARS,
        MAX_SEGMENT_CHARS,
        cluster_batch_size,
        target_final_themes,
        max_cluster_iterations,
        RANDOM_SEED,
        generation_workers,
        clustering_workers,
    )
    if mode not in {"new", "resume"}:
        raise ValueError("mode must be either 'new' or 'resume'.")
    if (
        isinstance(max_cluster_iterations, bool)
        or not isinstance(max_cluster_iterations, int)
        or max_cluster_iterations < 1
    ):
        raise ValueError("max_cluster_iterations must be a positive integer.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (generation_workers, clustering_workers)
    ):
        raise ValueError(
            "generation_workers and clustering_workers must be positive integers."
        )
    input_csv = Path(input_csv).resolve()
    if not input_csv.is_file():
        raise ValueError(f"Input CSV does not exist: {input_csv}")
    profile = load_prompt_profile(prompt_file)
    runs_dir, output_dir = resolve_run_directory(runs_dir, run_name)
    settings = _pipeline_settings(
        max_usable_records,
        cluster_batch_size=cluster_batch_size,
        target_final_themes=target_final_themes,
        max_cluster_iterations=max_cluster_iterations,
    )
    manifest = make_manifest(
        run_name=run_name,
        input_csv=input_csv,
        text_column=text_column,
        profile=profile,
        settings=settings,
        base_dir=BASE_DIR,
        selected_record_ids=include_record_ids,
    )
    manifest_path = output_dir / "run_manifest.json"
    if mode == "new":
        if output_dir.exists():
            raise FileExistsError(
                f"New run refused: {output_dir} already exists. Use a new run name or mode='resume'."
            )
    else:
        if not output_dir.is_dir():
            raise ValueError(f"Cannot resume; run directory does not exist: {output_dir}")
        if (output_dir / "run_config.json").is_file():
            raise ValueError("Cannot resume a completed run; create a new named run instead.")
        validate_manifest(manifest_path, manifest)
        LOGGER.info("Run manifest validation passed: %s", manifest_path)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set before running HiCode.")
    LOGGER.info("Credentials check passed; API key value will not be logged")
    if mode == "new":
        output_dir.mkdir(parents=True)
        write_manifest(manifest_path, manifest)
        LOGGER.info("Created immutable run manifest: %s", manifest_path)
    start_execution_attempt(
        output_dir,
        mode,
        generation_workers,
        clustering_workers,
        attempt_id=attempt_id,
    )
    LOGGER.info(
        "Run scope: input=%s output=%s max_usable_records=%s mode=%s text_column=%s profile=%s",
        input_csv,
        output_dir,
        max_usable_records if max_usable_records is not None else "none (full corpus)",
        mode,
        text_column,
        profile["name"],
    )
    can_resume_generation = mode == "resume" and (output_dir / "generation.json").is_file()
    can_resume_preprocessing = mode == "resume" and (output_dir / "preprocessing_report.json").is_file()
    clustering_dir = _prepare_output_directory(output_dir)
    if can_resume_generation:
        LOGGER.info("Stages 1-2/5: resuming saved preprocessing and generation")
        (
            cleaned_records,
            segments,
            excluded_records,
            preprocessing_report,
            generation_result,
        ) = _load_resumable_generation(output_dir, max_usable_records, text_column)
    elif can_resume_preprocessing:
        LOGGER.info("Stage 1/5: resuming saved preprocessing")
        (
            cleaned_records,
            segments,
            excluded_records,
            preprocessing_report,
        ) = _load_resumable_preprocessing(output_dir, max_usable_records, text_column)
        LOGGER.info("Stage 2/5: inductive initial-code generation")
        data_processed = {segment["segment_id"]: segment["text"] for segment in segments}
        generation_prompt = make_generation_prompt(
            profile["background"],
            profile["research_question"],
            profile["generation_instructions"],
        )
        generation_result = generate_labels(
            data_processed,
            generation_prompt,
            {
                "model_name": GENERATION_MODEL,
                "reasoning_effort": GENERATION_REASONING_EFFORT,
                "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
                "max_workers": generation_workers,
                "checkpoint_dir": output_dir / "generation" / "checkpoints",
                "failure_path": output_dir / "generation" / "failures.json",
                "resume_generation": mode == "resume",
            },
        )
        _write_json(output_dir / "generation.json", generation_result)
        clear_generation_checkpoints(output_dir / "generation" / "checkpoints")
    else:
        LOGGER.info("Stage 1/5: preprocessing and segmentation")
        cleaned_records, segments, excluded_records, preprocessing_report = preprocess_records(
            input_csv=input_csv,
            max_usable_records=max_usable_records,
            text_column=text_column,
            include_record_ids=include_record_ids,
        )
        if not cleaned_records or not segments:
            raise ValueError("Preprocessing produced no usable text segments.")

        _save_preprocessing_outputs(
            output_dir, cleaned_records, segments, excluded_records, preprocessing_report
        )

        LOGGER.info("Stage 2/5: inductive initial-code generation")
        data_processed = {
            segment["segment_id"]: segment["text"] for segment in segments
        }
        generation_prompt = make_generation_prompt(
            profile["background"],
            profile["research_question"],
            profile["generation_instructions"],
        )
        generation_config = {
            "model_name": GENERATION_MODEL,
            "reasoning_effort": GENERATION_REASONING_EFFORT,
            "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
            "max_workers": generation_workers,
            "checkpoint_dir": output_dir / "generation" / "checkpoints",
            "failure_path": output_dir / "generation" / "failures.json",
            "resume_generation": False,
        }
        generation_result = generate_labels(
            data_processed, generation_prompt, generation_config
        )
        _write_json(output_dir / "generation.json", generation_result)
        clear_generation_checkpoints(output_dir / "generation" / "checkpoints")

    if not cleaned_records or not segments:
        raise ValueError("Preprocessing produced no usable text segments.")

    relevant_segment_ids = {
        annotation["segment_id"]
        for record in generation_result.values()
        for annotation in record.get("LLM_Annotation", [])
    }
    irrelevant_segments = [
        segment
        for segment in segments
        if segment["segment_id"] not in relevant_segment_ids
    ]
    _write_csv(
        output_dir / "irrelevant_segments.csv",
        irrelevant_segments,
        ["record_id", "segment_id", "segment_type", "text"],
    )

    unique_codes = process_labels(generation_result)
    if not unique_codes:
        raise ValueError("Label generation produced no relevant initial codes.")

    LOGGER.info(
        "Generation reconciled: relevant_segments=%d irrelevant_segments=%d unique_codes=%d",
        len(relevant_segment_ids),
        len(irrelevant_segments),
        len(unique_codes),
    )

    LOGGER.info("Stage 3/5: hierarchical clustering")
    clustering_config = {
        "cluster_model_name": CLUSTER_MODEL,
        "reasoning_effort": CLUSTER_REASONING_EFFORT,
        "cluster_output_dir": str(output_dir),
        "cluster_run_dir": str(clustering_dir),
        "batch_size": cluster_batch_size,
        "target_final_themes": target_final_themes,
        "max_n_iter": max_cluster_iterations,
        "max_cluster_attempts": MAX_CLUSTER_ATTEMPTS,
        "max_batch_split_depth": MAX_BATCH_SPLIT_DEPTH,
        "random_seed": RANDOM_SEED,
        "max_workers": clustering_workers,
        "resume_clustering": mode == "resume",
        "checkpoint_batches": True,
    }
    cluster_iterations = cluster_labels_gpt(
        generation_result,
        _clustering_prompt_factory(profile),
        clustering_config,
        save_intermediate=True,
        gen_result_id="hicode",
    )

    LOGGER.info("Stage 4/5: hierarchy reconstruction and assignment building")
    code_paths = build_code_paths(cluster_iterations)
    if set(code_paths) != set(unique_codes):
        missing = sorted(set(unique_codes) - set(code_paths))
        extra = sorted(set(code_paths) - set(unique_codes))
        raise ValueError(
            f"Hierarchy/code mismatch. Missing={missing!r}; extra={extra!r}."
        )
    assignments = build_assignments(generation_result, segments, code_paths)
    summary, codes, examples = build_result_tables(assignments, code_paths)
    LOGGER.info(
        "Assignments built: occurrences=%d final_themes=%d examples=%d",
        len(assignments),
        len(summary),
        len(examples),
    )

    LOGGER.info("Stage 5/5: writing analysis exports")
    level_count = len(cluster_iterations)
    assignment_fields = [
        "record_id",
        "segment_id",
        "segment_type",
        "segment_text",
        "initial_code",
    ] + [f"level_{level}_theme" for level in range(1, level_count + 1)] + [
        "final_theme"
    ]
    _write_csv(output_dir / "assignments.csv", assignments, assignment_fields)
    _write_csv(
        output_dir / "theme_summary.csv",
        summary,
        [
            "final_theme",
            "code_occurrences",
            "unique_initial_codes",
            "segments",
            "records",
            "share_of_code_occurrences",
        ],
    )
    _write_csv(
        output_dir / "theme_codes.csv",
        codes,
        ["final_theme", "initial_code", "occurrences", "hierarchy_path"],
    )
    _write_csv(
        output_dir / "theme_examples.csv",
        examples,
        [
            "final_theme",
            "record_id",
            "segment_id",
            "segment_text",
            "theme_code_count",
        ],
    )

    final_to_codes = defaultdict(list)
    for code, path in code_paths.items():
        final_to_codes[path[-1]].append(code)
    hierarchy = {
        "iterations": cluster_iterations,
        "code_paths": code_paths,
        "final_theme_to_initial_codes": {
            theme: sorted(values) for theme, values in sorted(final_to_codes.items())
        },
    }
    _write_json(output_dir / "hierarchy.json", hierarchy)

    stop_reason = (
        "target_final_themes"
        if target_final_themes is not None
        and len(cluster_iterations[-1]) <= target_final_themes
        else "max_cluster_iterations"
    )
    counts = {
        "input_records": preprocessing_report["source_rows_scanned"],
        "included_records": len(cleaned_records),
        "excluded_records": len(excluded_records),
        "segments": len(segments),
        "relevant_segments": len(relevant_segment_ids),
        "irrelevant_segments": len(irrelevant_segments),
        "irrelevant_segment_percentage": len(irrelevant_segments) / len(segments),
        "code_occurrences": len(assignments),
        "unique_codes": len(unique_codes),
        "cluster_iterations": len(cluster_iterations),
        "final_themes": len(cluster_iterations[-1]),
    }
    run_config = {
        "generation_model": GENERATION_MODEL,
        "cluster_model": CLUSTER_MODEL,
        "generation_reasoning_effort": GENERATION_REASONING_EFFORT,
        "cluster_reasoning_effort": CLUSTER_REASONING_EFFORT,
        "coding_background": profile["background"],
        "coding_goal": profile["research_question"],
        "prompt_profile_name": profile["name"],
        "prompt_profile": profile,
        "generation_workers": generation_workers,
        "clustering_workers": clustering_workers,
        "execution_attempt_id": attempt_id,
        **settings,
        "mode": mode,
        "text_column": text_column,
        "stop_reason": stop_reason,
        "counts": counts,
    }
    _write_json(output_dir / "run_config.json", run_config)

    finish_execution_attempt(output_dir, attempt_id, "completed")

    LOGGER.info(
        "HiCode complete: records=%d segments=%d relevant_segments=%d code_occurrences=%d unique_codes=%d hierarchy_levels=%d final_themes=%d",
        counts["included_records"],
        counts["segments"],
        counts["relevant_segments"],
        counts["code_occurrences"],
        counts["unique_codes"],
        counts["cluster_iterations"],
        counts["final_themes"],
    )
    LOGGER.info("Outputs written to %s", output_dir)
    LOGGER.info("HiCode pipeline finished successfully")
    LOGGER.info("=" * 72)
    return run_config


def run_pipeline(
    *,
    run_name,
    mode,
    runs_dir,
    max_usable_records=None,
    input_csv,
    text_column,
    prompt_file,
    generation_workers=DEFAULT_GENERATION_WORKERS,
    clustering_workers=DEFAULT_CLUSTERING_WORKERS,
    cluster_batch_size=CLUSTER_BATCH_SIZE,
    target_final_themes=TARGET_FINAL_THEMES,
    max_cluster_iterations,
    include_record_ids=None,
):
    """Run one named HiCode analysis with bounded LLM concurrency."""
    attempt_id = uuid.uuid4().hex
    try:
        return _run_pipeline_impl(
            run_name=run_name,
            mode=mode,
            runs_dir=runs_dir,
            max_usable_records=max_usable_records,
            input_csv=input_csv,
            text_column=text_column,
            prompt_file=prompt_file,
            generation_workers=generation_workers,
            clustering_workers=clustering_workers,
            cluster_batch_size=cluster_batch_size,
            target_final_themes=target_final_themes,
            max_cluster_iterations=max_cluster_iterations,
            include_record_ids=include_record_ids,
            attempt_id=attempt_id,
        )
    except BaseException as exc:
        try:
            _, output_dir = resolve_run_directory(runs_dir, run_name)
            finish_execution_attempt(
                output_dir,
                attempt_id,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=exc,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            LOGGER.debug("Could not update execution attempt status", exc_info=True)
        raise


def _parse_args(argv=None):
    def positive_int(value):
        number = int(value)
        if number < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return number

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("new", "resume"), required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--max-usable-records",
        type=positive_int,
        default=None,
        help="Maximum number of usable records to process (default: unlimited).",
    )
    parser.add_argument(
        "--generation-workers",
        type=positive_int,
        default=DEFAULT_GENERATION_WORKERS,
        help=f"Concurrent label-generation requests (default: {DEFAULT_GENERATION_WORKERS}).",
    )
    parser.add_argument(
        "--clustering-workers",
        type=positive_int,
        default=DEFAULT_CLUSTERING_WORKERS,
        help=f"Concurrent clustering requests per level (default: {DEFAULT_CLUSTERING_WORKERS}).",
    )
    parser.add_argument(
        "--cluster-batch-size",
        type=positive_int,
        default=CLUSTER_BATCH_SIZE,
        help=f"Labels per clustering batch (default: {CLUSTER_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--target-final-themes",
        type=positive_int,
        default=TARGET_FINAL_THEMES,
        help="Stop clustering at or below this number of themes (default: disabled).",
    )
    parser.add_argument(
        "--max-cluster-iterations",
        type=positive_int,
        required=True,
        help="Maximum hierarchy levels to generate.",
    )
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    run_pipeline(
        run_name=args.run_name,
        mode=args.mode,
        runs_dir=args.runs_dir,
        input_csv=args.input_csv,
        text_column=args.text_column,
        prompt_file=args.prompt_file,
        max_usable_records=args.max_usable_records,
        generation_workers=args.generation_workers,
        clustering_workers=args.clustering_workers,
        cluster_batch_size=args.cluster_batch_size,
        target_final_themes=args.target_final_themes,
        max_cluster_iterations=args.max_cluster_iterations,
    )


if __name__ == "__main__":
    main()
