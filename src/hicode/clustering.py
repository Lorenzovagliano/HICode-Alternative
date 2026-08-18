from collections import Counter
import json
import logging
import os
import random
import threading
import uuid
from pathlib import Path

from openai import OpenAI
from .concurrency import (
    ConcurrentTaskError,
    run_bounded_tasks,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)
_CLIENT_LOCAL = threading.local()


def process_labels(data):
    labels = []
    for record in data.values():
        for segment in record.get("LLM_Annotation", []):
            labels.extend(segment["label"])
    return sorted(set(labels))


def make_clustering_prompt(goal=None, instructions=None):
    if goal is None:
        raise ValueError(
            "The description of the goal of inductive coding must be provided."
        )

    extra_guidance = _format_optional_instructions(instructions)
    return f"""Given the current inductive labels, cluster semantically similar labels into meaningful and insightful themes for:

{goal.strip()}

Return only a JSON object.
Each key must be the name of a synthesized theme.
Each value must be a list containing only labels supplied in the input.
Every supplied input label must belong to exactly one theme.
Do not add explanations or rationales.
{extra_guidance}
"""


def _format_optional_instructions(instructions):
    if not instructions:
        return ""
    if isinstance(instructions, str):
        instructions = [instructions]
    if not isinstance(instructions, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in instructions
    ):
        raise ValueError("Clustering instructions must be a string or non-empty strings.")
    return "\nAdditional task guidance:\n" + "\n".join(
        f"- {item.strip()}" for item in instructions
    )


def validate_clustering_output(model_output, input_labels):
    """Require exact, one-to-one coverage of a clustering batch."""
    errors = []
    flattened = []

    if not isinstance(model_output, dict) or not model_output:
        raise ValueError("Clustering output must be a non-empty JSON object.")

    for theme, children in model_output.items():
        if not isinstance(theme, str) or not theme.strip():
            errors.append(f"invalid theme name {theme!r}")
        if not isinstance(children, list) or not children:
            errors.append(f"theme {theme!r} must contain a non-empty list")
            continue
        for child in children:
            if not isinstance(child, str) or not child:
                errors.append(f"theme {theme!r} contains invalid label {child!r}")
            else:
                flattened.append(child)

    expected = Counter(input_labels)
    actual = Counter(flattened)
    missing = sorted((expected - actual).elements())
    invented = sorted((actual - expected).elements())
    duplicates = sorted(
        label for label, count in actual.items() if count > expected.get(label, 0)
    )
    if missing:
        errors.append(f"missing labels: {missing!r}")
    if invented:
        errors.append(f"invented labels: {invented!r}")
    if duplicates:
        errors.append(f"duplicate labels: {duplicates!r}")

    if errors:
        raise ValueError("; ".join(errors))
    return model_output


def _run_batch(client, system_prompt, cluster_model_name, input_labels, config):
    max_attempts = config.get("max_cluster_attempts", 3)
    if max_attempts < 1:
        raise ValueError("max_cluster_attempts must be at least 1.")

    if client is None:
        client = _get_openai_client()
    input_payload = json.dumps(input_labels, ensure_ascii=False)
    last_error = None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_payload},
    ]
    for attempt in range(max_attempts):
        if last_error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous JSON was invalid: "
                        f"{last_error}\nCorrect that JSON so every label from the "
                        "original input array appears verbatim exactly once. Keep all "
                        "valid assignments, add every missing label, remove duplicates "
                        "and invented labels, and return the complete corrected JSON "
                        "object only."
                    ),
                }
            )

        request = {
            "model": cluster_model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": config.get("max_completion_tokens", 8192),
        }
        reasoning_effort = config.get("reasoning_effort")
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort

        LOGGER.info(
            "Clustering API request: labels=%d attempt=%d/%d model=%s",
            len(input_labels),
            attempt + 1,
            max_attempts,
            cluster_model_name,
        )
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content
        try:
            model_output = json.loads(content) if content else None
            return validate_clustering_output(model_output, input_labels)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            LOGGER.warning(
                "Clustering response rejected: labels=%d attempt=%d/%d error=%s",
                len(input_labels),
                attempt + 1,
                max_attempts,
                last_error,
            )
            messages.append({"role": "assistant", "content": content or ""})

    raise RuntimeError(
        f"Clustering output remained invalid after {max_attempts} attempts: {last_error}"
    )


def _run_batch_with_split(
    client,
    system_prompt,
    cluster_model_name,
    input_labels,
    config,
    split_depth=0,
):
    """Retry a failed batch as smaller exact-coverage batches."""
    try:
        return _run_batch(
            client, system_prompt, cluster_model_name, input_labels, config
        )
    except RuntimeError:
        max_split_depth = config.get("max_batch_split_depth", 2)
        if split_depth >= max_split_depth or len(input_labels) < 2:
            raise
        midpoint = len(input_labels) // 2
        child_batches = [input_labels[:midpoint], input_labels[midpoint:]]
        LOGGER.warning(
            "Clustering batch exhausted normal retries; splitting %d labels into %d and %d labels (depth %d/%d)",
            len(input_labels),
            len(child_batches[0]),
            len(child_batches[1]),
            split_depth + 1,
            max_split_depth,
        )
        combined = {}
        for child in child_batches:
            child_output = _run_batch_with_split(
                client,
                system_prompt,
                cluster_model_name,
                child,
                config,
                split_depth=split_depth + 1,
            )
            for theme, labels in child_output.items():
                combined.setdefault(theme, []).extend(labels)
        return validate_clustering_output(combined, input_labels)


def _result_dir(config, output_id):
    configured = config.get("cluster_run_dir")
    if configured is not None:
        return Path(configured)
    return Path(config["cluster_output_dir"]) / f"clustering_{output_id}"


def _iteration_path(config, output_id, iteration):
    return _result_dir(config, output_id) / f"cluster_iter_{iteration}.json"


def _checkpoint_path(config, output_id, iteration, batch_index):
    return (
        _result_dir(config, output_id)
        / "checkpoints"
        / f"iter_{iteration}"
        / f"batch_{batch_index:05d}.json"
    )


def _load_batch_checkpoint(path, iteration, batch_index, input_labels):
    with Path(path).open("r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    if checkpoint.get("iteration") != iteration:
        raise ValueError(f"Checkpoint {path} has the wrong iteration.")
    if checkpoint.get("batch_index") != batch_index:
        raise ValueError(f"Checkpoint {path} has the wrong batch index.")
    if checkpoint.get("input_labels") != input_labels:
        raise ValueError(
            f"Checkpoint {path} does not match the deterministic input batch."
        )
    return validate_clustering_output(checkpoint.get("model_output"), input_labels)


def _clear_iteration_checkpoints(config, output_id, iteration):
    checkpoint_dir = _checkpoint_path(config, output_id, iteration, 0).parent
    if not checkpoint_dir.is_dir():
        return
    for pattern in ("batch_*.json", "batch_*.json.tmp", "failures.json", "failures.json.tmp"):
        for path in checkpoint_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    if any(checkpoint_dir.iterdir()):
        return
    checkpoint_dir.rmdir()
    parent = checkpoint_dir.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def cluster_labels_gpt(
    generation_result,
    system_prompt,
    config,
    save_intermediate=True,
    gen_result_id=None,
    max_n_iter=10,
):
    if isinstance(generation_result, str):
        with open(generation_result, "r", encoding="utf-8") as f:
            generation_result = json.load(f)

    labels_to_cluster = process_labels(generation_result)
    if not labels_to_cluster:
        raise ValueError("No generated labels are available for clustering.")

    cluster_model_name = config["cluster_model_name"]
    max_n_iter = config.get("max_n_iter", max_n_iter)
    batch_size = config.get("batch_size", 100)
    max_workers = config.get("max_workers", 1)
    target_final_themes = config.get("target_final_themes")
    random_seed = config.get("random_seed", 42)
    resume_clustering = config.get("resume_clustering", False)
    checkpoint_batches = config.get("checkpoint_batches", save_intermediate)
    if max_n_iter < 1:
        raise ValueError("max_n_iter must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")
    if target_final_themes is not None and (
        isinstance(target_final_themes, bool)
        or not isinstance(target_final_themes, int)
        or target_final_themes < 1
    ):
        raise ValueError("target_final_themes must be at least 1.")

    LOGGER.info(
        "Clustering started: model=%s unique_labels=%d batch_size=%d target_themes=%s max_iterations=%d seed=%d reasoning=%s",
        cluster_model_name,
        len(labels_to_cluster),
        batch_size,
        target_final_themes,
        max_n_iter,
        random_seed,
        config.get("reasoning_effort"),
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before clustering.")
    if gen_result_id is None:
        output_id = uuid.uuid1()
    else:
        output_id = gen_result_id

    cluster_list = []
    for iteration in range(max_n_iter):
        ordered_labels = sorted(labels_to_cluster)
        random.Random(random_seed + iteration).shuffle(ordered_labels)
        batches = [
            ordered_labels[start : start + batch_size]
            for start in range(0, len(ordered_labels), batch_size)
        ]
        LOGGER.info(
            "Clustering iteration %d/%d: input_labels=%d batches=%d",
            iteration + 1,
            max_n_iter,
            len(ordered_labels),
            len(batches),
        )

        completed_iteration_path = _iteration_path(config, output_id, iteration)
        if resume_clustering and completed_iteration_path.is_file():
            with completed_iteration_path.open("r", encoding="utf-8") as f:
                completed_cluster = json.load(f)
            validate_clustering_output(completed_cluster, ordered_labels)
            cluster_list.append(completed_cluster)
            LOGGER.info(
                "Resumed completed clustering level %d: themes=%d path=%s",
                iteration + 1,
                len(completed_cluster),
                completed_iteration_path,
            )
            if target_final_themes is not None and len(completed_cluster) <= target_final_themes:
                break
            labels_to_cluster = sorted(completed_cluster.keys())
            continue

        prompt_for_iteration = (
            system_prompt(iteration)
            if callable(system_prompt)
            else system_prompt
        )
        batch_results = {}
        missing_batches = []
        for batch_index, batch in enumerate(batches):
            checkpoint_path = _checkpoint_path(config, output_id, iteration, batch_index)
            if resume_clustering and checkpoint_path.is_file():
                batch_results[batch_index] = _load_batch_checkpoint(
                    checkpoint_path, iteration, batch_index, batch
                )
                LOGGER.info(
                    "Resumed clustering checkpoint: level=%d batch=%d/%d themes=%d",
                    iteration + 1,
                    batch_index + 1,
                    len(batches),
                    len(batch_results[batch_index]),
                )
            else:
                missing_batches.append((batch_index, batch))

        def cluster_one(batch_index, batch):
            LOGGER.info(
                "Clustering batch %d/%d at level %d: labels=%d",
                batch_index + 1,
                len(batches),
                iteration + 1,
                len(batch),
            )
            model_output = _run_batch_with_split(
                _get_openai_client(),
                prompt_for_iteration,
                cluster_model_name,
                batch,
                config,
            )
            if checkpoint_batches:
                checkpoint_path = _checkpoint_path(
                    config, output_id, iteration, batch_index
                )
                write_json_atomic(
                    checkpoint_path,
                    {
                        "iteration": iteration,
                        "batch_index": batch_index,
                        "input_labels": batch,
                        "model_output": model_output,
                    },
                )
                LOGGER.info(
                    "Saved clustering checkpoint: level=%d batch=%d/%d path=%s",
                    iteration + 1,
                    batch_index + 1,
                    len(batches),
                    checkpoint_path,
                )
            return model_output

        try:
            batch_results.update(
                run_bounded_tasks(
                    missing_batches,
                    cluster_one,
                    max_workers,
                    f"clustering level {iteration + 1}",
                )
            )
        except ConcurrentTaskError as exc:
            failure_path = _checkpoint_path(config, output_id, iteration, 0).parent / "failures.json"
            write_json_atomic(failure_path, exc.as_dict())
            raise RuntimeError(str(exc)) from exc

        cluster = {}
        for batch_index in range(len(batches)):
            model_output = batch_results[batch_index]
            LOGGER.info(
                "Clustering batch accepted: level=%d batch=%d/%d themes=%d",
                iteration + 1,
                batch_index + 1,
                len(batches),
                len(model_output),
            )
            for theme, children in model_output.items():
                cluster.setdefault(theme, []).extend(children)

        validate_clustering_output(cluster, ordered_labels)

        cluster_list.append(cluster)
        if save_intermediate:
            result_path = save_iteration(cluster, iteration, config, output_id)
            LOGGER.info(
                "Clustering level %d saved: themes=%d path=%s",
                iteration + 1,
                len(cluster),
                result_path,
            )
            _clear_iteration_checkpoints(config, output_id, iteration)

        if target_final_themes is not None and len(cluster) <= target_final_themes:
            LOGGER.info(
                "Clustering stopping: %d themes is at or below target %d",
                len(cluster),
                target_final_themes,
            )
            break
        labels_to_cluster = sorted(cluster.keys())

    LOGGER.info(
        "Clustering complete: levels=%d final_themes=%d",
        len(cluster_list),
        len(cluster_list[-1]),
    )
    return cluster_list


def save_iteration(cluster_result, n_iter, config, output_id):
    result_dir = config.get("cluster_run_dir")
    if result_dir is None:
        result_dir = os.path.join(
            config["cluster_output_dir"], f"clustering_{output_id}"
        )
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"cluster_iter_{n_iter}.json")
    write_json_atomic(result_path, cluster_result)
    return result_path


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
