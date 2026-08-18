"""Read-only loading, validation, and aggregation helpers for HiCode results."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FILES = (
    "preprocessing_report.json",
    "run_config.json",
    "hierarchy.json",
    "assignments.csv",
    "theme_summary.csv",
    "theme_codes.csv",
    "theme_examples.csv",
    "segments.csv",
)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer {field!r} in visualization input: {row!r}") from exc


def load_results(input_dir):
    """Load and validate an existing HiCode run without changing it."""
    input_dir = Path(input_dir)
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"HiCode visualization requires files missing from {input_dir}: {', '.join(missing)}"
        )
    results = {
        "input_dir": input_dir,
        "preprocessing_report": _read_json(input_dir / "preprocessing_report.json"),
        "run_config": _read_json(input_dir / "run_config.json"),
        "hierarchy": _read_json(input_dir / "hierarchy.json"),
        "assignments": _read_csv(input_dir / "assignments.csv"),
        "summary": _read_csv(input_dir / "theme_summary.csv"),
        "codes": _read_csv(input_dir / "theme_codes.csv"),
        "examples": _read_csv(input_dir / "theme_examples.csv"),
        "segments": _read_csv(input_dir / "segments.csv"),
    }
    validate_results(results)
    return results


def level_columns(assignments):
    columns = [column for column in assignments[0] if column.startswith("level_") and column.endswith("_theme")]
    return sorted(columns, key=lambda column: int(column.split("_")[1]))


def ranking_levels(assignments):
    """Return Overview ranking choices from the finest to the coarsest level."""
    if not assignments:
        raise ValueError("Cannot derive ranking levels from empty assignments.")

    columns = level_columns(assignments)
    levels = [("Initial codes", "initial_code")]
    for index, column in enumerate(columns, start=1):
        label = f"Level {index}"
        if index == len(columns):
            label += " (final themes)"
        levels.append((label, column))
    return levels


def ranking_summary(results, grouping_column, final_themes=None):
    """Aggregate assignment metrics for an Overview ranking level.

    ``final_themes`` is intentionally applied to the assignment rows before
    aggregation. This makes an earlier-level ranking show only groups that
    contain the selected final-theme descendants, while the share denominator
    remains the complete corpus for consistency with the final-theme summary.
    """
    assignments = results["assignments"]
    valid_columns = {"initial_code", *level_columns(assignments)}
    if grouping_column not in valid_columns:
        raise ValueError(f"Unknown ranking column: {grouping_column!r}")

    selected_final_themes = set(final_themes or [])
    visible_assignments = (
        assignments
        if not selected_final_themes
        else [row for row in assignments if row["final_theme"] in selected_final_themes]
    )
    total_occurrences = len(assignments)
    grouped = defaultdict(lambda: {"initial_codes": set(), "segments": set(), "records": set()})
    occurrences = Counter()
    for row in visible_assignments:
        theme = row[grouping_column]
        occurrences[theme] += 1
        grouped[theme]["initial_codes"].add(row["initial_code"])
        grouped[theme]["segments"].add(row["segment_id"])
        grouped[theme]["records"].add(row["record_id"])

    summary = [
        {
            "theme": theme,
            "code_occurrences": count,
            "unique_initial_codes": len(grouped[theme]["initial_codes"]),
            "segments": len(grouped[theme]["segments"]),
            "records": len(grouped[theme]["records"]),
            "share_of_code_occurrences": count / total_occurrences if total_occurrences else 0.0,
        }
        for theme, count in occurrences.items()
    ]
    return sorted(summary, key=lambda row: (-row["code_occurrences"], row["theme"]))


def _validate_grouping_column(results, grouping_column):
    assignments = results["assignments"]
    valid_columns = {"initial_code", *level_columns(assignments)}
    if grouping_column not in valid_columns:
        raise ValueError(f"Unknown grouping column: {grouping_column!r}")


def group_assignments(results, grouping_column, group, final_themes=None):
    """Return assignments for a selected level/group and final-theme filter."""
    _validate_grouping_column(results, grouping_column)
    selected_final_themes = set(final_themes or [])
    rows = [
        row
        for row in results["assignments"]
        if row[grouping_column] == group
        and (not selected_final_themes or row["final_theme"] in selected_final_themes)
    ]
    return sorted(
        rows,
        key=lambda row: (row["segment_id"], row["initial_code"], row["final_theme"]),
    )


def group_code_rows(results, grouping_column, group, final_themes=None, limit=None):
    """Aggregate initial-code rows for a selected level/group."""
    rows = group_assignments(results, grouping_column, group, final_themes)
    occurrences = Counter(row["initial_code"] for row in rows)
    final_themes_by_code = defaultdict(set)
    for row in rows:
        final_themes_by_code[row["initial_code"]].add(row["final_theme"])

    code_rows = []
    for initial_code, count in occurrences.items():
        code_rows.append(
            {
                "final_theme": ", ".join(sorted(final_themes_by_code[initial_code])),
                "initial_code": initial_code,
                "occurrences": str(count),
                "hierarchy_path": " -> ".join(
                    [initial_code] + results["hierarchy"]["code_paths"][initial_code]
                ),
            }
        )
    code_rows.sort(key=lambda row: (-int(row["occurrences"]), row["initial_code"]))
    return code_rows if limit is None else code_rows[:limit]


def all_code_rows(results, final_themes=None, limit=None):
    """Aggregate initial-code rows across all selected final-theme descendants."""
    selected_final_themes = set(final_themes or [])
    rows = [
        row
        for row in results["assignments"]
        if not selected_final_themes or row["final_theme"] in selected_final_themes
    ]
    occurrences = Counter(row["initial_code"] for row in rows)
    final_themes_by_code = defaultdict(set)
    for row in rows:
        final_themes_by_code[row["initial_code"]].add(row["final_theme"])

    code_rows = [
        {
            "final_theme": ", ".join(sorted(final_themes_by_code[initial_code])),
            "initial_code": initial_code,
            "occurrences": str(count),
            "hierarchy_path": " -> ".join(
                [initial_code] + results["hierarchy"]["code_paths"][initial_code]
            ),
        }
        for initial_code, count in occurrences.items()
    ]
    code_rows.sort(key=lambda row: (-int(row["occurrences"]), row["initial_code"]))
    return code_rows if limit is None else code_rows[:limit]


def group_examples(results, grouping_column, group, final_themes=None, limit=None):
    """Return representative evidence for a selected level/group.

    Final-theme groups reuse the exported examples table so the existing final
    view remains byte-for-byte equivalent. Earlier levels derive examples from
    the filtered assignments while retaining the existing example schema.
    """
    _validate_grouping_column(results, grouping_column)
    final_column = level_columns(results["assignments"])[-1]
    if grouping_column in {"final_theme", final_column}:
        if final_themes and group not in set(final_themes):
            return []
        return theme_examples(results, group, limit=limit)

    rows = group_assignments(results, grouping_column, group, final_themes)
    by_segment = defaultdict(list)
    for row in rows:
        by_segment[(row["record_id"], row["segment_id"])].append(row)

    candidates = []
    for (record_id, segment_id), segment_rows in by_segment.items():
        first = segment_rows[0]
        candidates.append(
            {
                "final_theme": first["final_theme"],
                "record_id": record_id,
                "segment_id": segment_id,
                "segment_text": first["segment_text"],
                "theme_code_count": len(segment_rows),
            }
        )
    candidates.sort(key=lambda row: (-row["theme_code_count"], row["segment_id"]))

    examples = []
    seen_text = set()
    for row in candidates:
        if row["segment_text"] in seen_text:
            continue
        seen_text.add(row["segment_text"])
        examples.append(row)
        if len(examples) >= 10:
            break
    return examples if limit is None else examples[:limit]


def validate_results(results):
    """Fail closed when exported tables are not a coherent HiCode result."""
    assignments = results["assignments"]
    summary = results["summary"]
    hierarchy = results["hierarchy"]
    if not assignments:
        raise ValueError("assignments.csv is empty; there is no evidence to visualize.")
    required_assignment_columns = {
        "record_id", "segment_id", "segment_type", "segment_text", "initial_code", "final_theme"
    }
    missing_columns = required_assignment_columns - set(assignments[0])
    if missing_columns:
        raise ValueError(f"assignments.csv is missing columns: {sorted(missing_columns)!r}")
    levels = level_columns(assignments)
    if not levels:
        raise ValueError("assignments.csv has no hierarchy level columns.")

    final_themes = {row["final_theme"] for row in assignments}
    summary_themes = {row["final_theme"] for row in summary}
    if final_themes != summary_themes:
        raise ValueError(
            "Theme-summary coverage mismatch: "
            f"missing={sorted(final_themes - summary_themes)!r}; "
            f"extra={sorted(summary_themes - final_themes)!r}."
        )
    paths = hierarchy.get("code_paths")
    reverse = hierarchy.get("final_theme_to_initial_codes")
    iterations = hierarchy.get("iterations")
    if not isinstance(paths, dict) or not isinstance(reverse, dict) or not isinstance(iterations, list):
        raise ValueError("hierarchy.json lacks code_paths, final_theme_to_initial_codes, or iterations.")
    assigned_codes = {row["initial_code"] for row in assignments}
    if assigned_codes != set(paths):
        raise ValueError(
            "Hierarchy initial-code coverage mismatch: "
            f"missing={sorted(assigned_codes - set(paths))[:10]!r}; "
            f"extra={sorted(set(paths) - assigned_codes)[:10]!r}."
        )
    if len(iterations) != len(levels):
        raise ValueError(
            f"Hierarchy has {len(iterations)} iterations but assignments have {len(levels)} levels."
        )
    for code, path in paths.items():
        if not isinstance(path, list) or len(path) != len(levels) or not path or path[-1] not in final_themes:
            raise ValueError(f"Invalid complete hierarchy path for initial code {code!r}.")
    reverse_codes = {code for code_list in reverse.values() for code in code_list}
    if reverse_codes != assigned_codes:
        raise ValueError("hierarchy.json reverse theme mapping does not cover every initial code exactly once.")
    for theme, code_list in reverse.items():
        if theme not in final_themes or len(code_list) != len(set(code_list)):
            raise ValueError(f"Invalid reverse hierarchy mapping for final theme {theme!r}.")

    occurrence_total = len(assignments)
    if sum(_as_int(row, "code_occurrences") for row in summary) != occurrence_total:
        raise ValueError("theme_summary.csv code-occurrence total does not match assignments.csv.")
    configured = results["run_config"].get("counts", {})
    expected = {
        "code_occurrences": occurrence_total,
        "unique_codes": len(assigned_codes),
        "final_themes": len(final_themes),
    }
    for field, actual in expected.items():
        if configured.get(field) != actual:
            raise ValueError(
                f"run_config.json count {field!r}={configured.get(field)!r} does not match {actual}."
            )


def sorted_summary(results):
    return sorted(
        results["summary"],
        key=lambda row: (-_as_int(row, "code_occurrences"), row["final_theme"]),
    )


def theme_codes(results, theme, limit=None):
    rows = [row for row in results["codes"] if row["final_theme"] == theme]
    rows.sort(key=lambda row: (-_as_int(row, "occurrences"), row["initial_code"]))
    return rows if limit is None else rows[:limit]


def theme_examples(results, theme, limit=None):
    rows = [row for row in results["examples"] if row["final_theme"] == theme]
    rows.sort(key=lambda row: (-_as_int(row, "theme_code_count"), row["segment_id"]))
    return rows if limit is None else rows[:limit]


def filter_assignments(results, themes=None, initial_code=None, segment_types=None, query=""):
    """Deterministically filter assignment evidence for dashboard controls."""
    themes = set(themes or [])
    segment_types = set(segment_types or [])
    query = " ".join(query.casefold().split())
    rows = []
    for row in results["assignments"]:
        if themes and row["final_theme"] not in themes:
            continue
        if initial_code and row["initial_code"] != initial_code:
            continue
        if segment_types and row["segment_type"] not in segment_types:
            continue
        haystack = " ".join(str(value) for value in row.values()).casefold()
        if query and query not in haystack:
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: (row["segment_id"], row["initial_code"], row["final_theme"]))


def hierarchy_nodes(results, themes=None, code_limit_per_theme=None):
    """Aggregate occurrence-weighted hierarchy nodes with path-qualified IDs.

    A label can occur beneath more than one parent in a generated hierarchy.
    Using only ``level:label`` as the node ID would merge those branches while
    retaining only one parent, which produces invalid totals for a Plotly
    Sunburst with ``branchvalues="total"``.  Include the complete path in IDs
    so every node has exactly one parent while the display label stays concise.
    """
    themes = set(themes or [])
    codes = results["codes"]
    if themes:
        codes = [row for row in codes if row["final_theme"] in themes]
    if code_limit_per_theme is not None:
        limited = []
        for theme in sorted({row["final_theme"] for row in codes}):
            limited.extend(theme_codes(results, theme, code_limit_per_theme))
        codes = limited
    node_values = Counter()
    parents = {}
    labels = {"root": "All selected code occurrences"}
    for row in codes:
        code = row["initial_code"]
        path = results["hierarchy"]["code_paths"][code]
        value = _as_int(row, "occurrences")
        parent = "root"
        node_values[parent] += value
        for level, label in enumerate(path, start=1):
            # JSON gives us an unambiguous, deterministic representation even
            # when labels contain punctuation used by the dashboard's IDs.
            node_id = f"{level}:{json.dumps(path[:level], ensure_ascii=False, separators=(',', ':'))}"
            labels[node_id] = label
            parents[node_id] = parent
            node_values[node_id] += value
            parent = node_id
    rows = [
        {"id": node_id, "parent": parents.get(node_id, ""), "label": labels[node_id], "value": value}
        for node_id, value in node_values.items()
    ]
    return sorted(rows, key=lambda row: (row["id"] != "root", row["id"]))


def lineage_edges(
    results,
    theme=None,
    initial_code=None,
    final_themes=None,
    max_level=None,
):
    """Return aggregated, level-aware Sankey edges for a focused theme or code."""
    level_count = len(level_columns(results["assignments"]))
    if max_level is not None and not 0 <= max_level <= level_count:
        raise ValueError(f"max_level must be between 0 and {level_count}, got {max_level!r}")
    rows = results["codes"]
    if theme:
        rows = [row for row in rows if row["final_theme"] == theme]
    if initial_code:
        rows = [row for row in rows if row["initial_code"] == initial_code]
    selected_final_themes = set(final_themes or [])
    if selected_final_themes:
        rows = [row for row in rows if row["final_theme"] in selected_final_themes]
    edges = Counter()
    for row in rows:
        code = row["initial_code"]
        path = results["hierarchy"]["code_paths"][code]
        value = _as_int(row, "occurrences")
        previous = f"0:{code}"
        visible_path = path if max_level is None else path[:max_level]
        for level, label in enumerate(visible_path, start=1):
            current = f"{level}:{label}"
            edges[(previous, current)] += value
            previous = current
    return [
        {"source": source, "target": target, "value": value}
        for (source, target), value in sorted(edges.items())
    ]


def lineage_metrics(results, edges):
    """Return occurrence, share, segment, and record metrics for a Sankey view.

    A Sankey can display only a selected subset of initial-code paths.  These
    metrics deliberately use that same subset as their denominator, so a
    node's share means its share of the code occurrences currently visible in
    the diagram rather than its share of the entire corpus.
    """
    selected_codes = {
        edge["source"].split(":", 1)[1]
        for edge in edges
        if edge["source"].startswith("0:")
    }
    assignments = [
        row for row in results["assignments"] if row["initial_code"] in selected_codes
    ]
    total = len(assignments)
    node_occurrences = Counter()
    node_segments = defaultdict(set)
    node_records = defaultdict(set)
    link_occurrences = Counter()
    link_segments = defaultdict(set)
    link_records = defaultdict(set)
    for row in assignments:
        code = row["initial_code"]
        path = results["hierarchy"]["code_paths"][code]
        previous = f"0:{code}"
        node_occurrences[previous] += 1
        node_segments[previous].add(row["segment_id"])
        node_records[previous].add(row["record_id"])
        for level, label in enumerate(path, start=1):
            current = f"{level}:{label}"
            node_occurrences[current] += 1
            node_segments[current].add(row["segment_id"])
            node_records[current].add(row["record_id"])
            edge = (previous, current)
            link_occurrences[edge] += 1
            link_segments[edge].add(row["segment_id"])
            link_records[edge].add(row["record_id"])
            previous = current

    def metrics(occurrences, segments, records):
        return {
            "code_occurrences": occurrences,
            "share_of_code_occurrences": occurrences / total if total else 0.0,
            "segments": len(segments),
            "records": len(records),
        }

    return {
        "total_code_occurrences": total,
        "nodes": {
            node: metrics(node_occurrences[node], node_segments[node], node_records[node])
            for node in node_occurrences
        },
        "links": {
            edge: metrics(link_occurrences[edge], link_segments[edge], link_records[edge])
            for edge in link_occurrences
        },
    }
