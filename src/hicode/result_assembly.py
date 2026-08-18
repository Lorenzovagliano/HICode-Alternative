"""Build hierarchy paths and user-facing tables from pipeline results."""

from collections import Counter, defaultdict
import logging


LOGGER = logging.getLogger(__name__)
MAX_EXAMPLES_PER_THEME = 10


def build_code_paths(cluster_iterations):
    if not cluster_iterations:
        raise ValueError("At least one clustering iteration is required.")

    inverse_iterations = []
    for iteration_index, clusters in enumerate(cluster_iterations):
        inverse = {}
        for theme, children in clusters.items():
            for child in children:
                if child in inverse:
                    raise ValueError(
                        f"Iteration {iteration_index} assigns {child!r} more than once."
                    )
                inverse[child] = theme
        inverse_iterations.append(inverse)

    code_paths = {}
    for initial_code in sorted(inverse_iterations[0]):
        current = initial_code
        path = []
        for iteration_index, inverse in enumerate(inverse_iterations):
            if current not in inverse:
                raise ValueError(
                    f"Hierarchy lost {current!r} at iteration {iteration_index}."
                )
            current = inverse[current]
            path.append(current)
        code_paths[initial_code] = path
    LOGGER.info(
        "Hierarchy reconstructed: initial_codes=%d levels=%d",
        len(code_paths),
        len(cluster_iterations),
    )
    return code_paths


def build_assignments(generation_result, segments, code_paths):
    segment_lookup = {segment["segment_id"]: segment for segment in segments}
    assignments = []
    for record_id, record in generation_result.items():
        for annotation in record.get("LLM_Annotation", []):
            segment_id = annotation["segment_id"]
            if segment_id not in segment_lookup:
                raise ValueError(f"Unknown generated segment ID: {segment_id}")
            segment = segment_lookup[segment_id]
            for code in annotation["label"]:
                if code not in code_paths:
                    raise ValueError(f"Initial code {code!r} has no hierarchy path.")
                path = code_paths[code]
                row = {
                    "record_id": record_id,
                    "segment_id": segment_id,
                    "segment_type": segment["segment_type"],
                    "segment_text": segment["text"],
                    "initial_code": code,
                    "final_theme": path[-1],
                }
                for level, theme in enumerate(path, start=1):
                    row[f"level_{level}_theme"] = theme
                assignments.append(row)

    return sorted(
        assignments,
        key=lambda row: (row["segment_id"], row["initial_code"], row["final_theme"]),
    )


def build_result_tables(assignments, code_paths):
    if not assignments:
        raise ValueError("No code assignments were generated.")

    total_occurrences = len(assignments)
    theme_occurrences = Counter(row["final_theme"] for row in assignments)
    theme_codes = defaultdict(set)
    theme_segments = defaultdict(set)
    theme_records = defaultdict(set)
    for row in assignments:
        theme = row["final_theme"]
        theme_codes[theme].add(row["initial_code"])
        theme_segments[theme].add(row["segment_id"])
        theme_records[theme].add(row["record_id"])

    summary = [
        {
            "final_theme": theme,
            "code_occurrences": count,
            "unique_initial_codes": len(theme_codes[theme]),
            "segments": len(theme_segments[theme]),
            "records": len(theme_records[theme]),
            "share_of_code_occurrences": count / total_occurrences,
        }
        for theme, count in theme_occurrences.items()
    ]
    summary.sort(key=lambda row: (-row["code_occurrences"], row["final_theme"]))
    theme_rank = {row["final_theme"]: index for index, row in enumerate(summary)}

    code_occurrences = Counter(
        (row["final_theme"], row["initial_code"]) for row in assignments
    )
    codes = [
        {
            "final_theme": theme,
            "initial_code": code,
            "occurrences": count,
            "hierarchy_path": " -> ".join([code] + code_paths[code]),
        }
        for (theme, code), count in code_occurrences.items()
    ]
    codes.sort(
        key=lambda row: (
            theme_rank[row["final_theme"]],
            -row["occurrences"],
            row["initial_code"],
        )
    )

    by_theme_segment = defaultdict(list)
    for row in assignments:
        by_theme_segment[(row["final_theme"], row["segment_id"])].append(row)
    candidates = defaultdict(list)
    for (theme, segment_id), rows in by_theme_segment.items():
        first = rows[0]
        candidates[theme].append(
            {
                "final_theme": theme,
                "record_id": first["record_id"],
                "segment_id": segment_id,
                "segment_text": first["segment_text"],
                "theme_code_count": len(rows),
            }
        )

    examples = []
    for theme in sorted(candidates, key=lambda item: theme_rank[item]):
        ranked = sorted(
            candidates[theme],
            key=lambda row: (-row["theme_code_count"], row["segment_id"]),
        )
        seen_text = set()
        for row in ranked:
            if row["segment_text"] in seen_text:
                continue
            seen_text.add(row["segment_text"])
            examples.append(row)
            if len(seen_text) >= MAX_EXAMPLES_PER_THEME:
                break
    return summary, codes, examples
