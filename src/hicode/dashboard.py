#!/usr/bin/env python3
"""Local, read-only Streamlit explorer for completed HiCode results."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from .visualization import (
    all_code_rows,
    filter_assignments,
    group_assignments,
    group_code_rows,
    group_examples,
    level_columns,
    lineage_edges,
    lineage_metrics,
    load_results,
    ranking_levels,
    ranking_summary,
    sorted_summary,
    theme_codes,
)


def _argument_input_dir():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input-dir", required=True)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return str(Path(args.input_dir).resolve())


@st.cache_data(show_spinner="Loading and validating completed HiCode outputs...")
def _load(input_dir):
    return load_results(input_dir)


def _csv_bytes(rows):
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _sankey(results, edges, title):
    if not edges:
        return None
    names = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
    index = {name: position for position, name in enumerate(names)}
    full_labels = [f"L{name.split(':', 1)[0]} · {name.split(':', 1)[1]}" for name in names]
    labels = [label if len(label) <= 52 else f"{label[:49]}…" for label in full_labels]
    metrics = lineage_metrics(results, edges)

    def hover_data(name, label):
        values = metrics["nodes"][name]
        return [
            label,
            values["code_occurrences"],
            values["share_of_code_occurrences"],
            values["segments"],
            values["records"],
        ]

    nodes_by_level = {}
    for name in names:
        level = int(name.split(":", 1)[0])
        nodes_by_level[level] = nodes_by_level.get(level, 0) + 1
    height = min(2600, max(620, 150 + 34 * max(nodes_by_level.values())))
    return go.Figure(
        go.Sankey(
            node=dict(
                label=labels,
                customdata=[hover_data(name, label) for name, label in zip(names, full_labels)],
                hovertemplate=(
                    "%{customdata[0]}<br>Value: %{customdata[1]:,} code occurrences"
                    "<br>Share of visible code occurrences: %{customdata[2]:.1%}"
                    "<br>Segments: %{customdata[3]:,}<br>Records: %{customdata[4]:,}<extra></extra>"
                ),
                pad=18,
                thickness=16,
                color="#7f9ad1",
            ),
            link=dict(
                source=[index[edge["source"]] for edge in edges],
                target=[index[edge["target"]] for edge in edges],
                value=[edge["value"] for edge in edges],
                color="rgba(50,102,204,.28)",
                customdata=[
                    [
                        metrics["links"][(edge["source"], edge["target"])]["code_occurrences"],
                        metrics["links"][(edge["source"], edge["target"])]["share_of_code_occurrences"],
                        metrics["links"][(edge["source"], edge["target"])]["segments"],
                        metrics["links"][(edge["source"], edge["target"])]["records"],
                    ]
                    for edge in edges
                ],
                hovertemplate=(
                    "%{source.label} → %{target.label}<br>Value: %{customdata[0]:,} code occurrences"
                    "<br>Share of visible code occurrences: %{customdata[1]:.1%}"
                    "<br>Segments: %{customdata[2]:,}<br>Records: %{customdata[3]:,}<extra></extra>"
                ),
            ),
        )
    ).update_layout(title=title, font_size=12, height=height, margin=dict(l=10, r=10, t=42, b=10))


def _overview(results, active_themes):
    levels = ranking_levels(results["assignments"])
    level_labels = [label for label, _ in levels]
    selected_level = st.selectbox(
        "Rank at clustering level",
        level_labels,
        index=len(level_labels) - 1,
        key="overview_ranking_level",
        help="Choose initial codes or any retained clustering iteration.",
    )
    grouping_column = dict(levels)[selected_level]
    display_limit = st.selectbox(
        "Display",
        [25, 50, 100, 250, "All"],
        index=1,
        format_func=lambda value: "All groups" if value == "All" else f"Top {value} groups",
        key="overview_ranking_limit",
        help="Limit the number of bars shown when a level contains many groups.",
    )
    metric_label = st.radio(
        "Rank groups by", ["Code occurrences", "Distinct segments", "Distinct records", "Unique initial codes"], horizontal=True
    )
    metric = {
        "Code occurrences": "code_occurrences",
        "Distinct segments": "segments",
        "Distinct records": "records",
        "Unique initial codes": "unique_initial_codes",
    }[metric_label]
    summary = ranking_summary(results, grouping_column, final_themes=active_themes)
    ordered = sorted(summary, key=lambda row: (-int(row[metric]), row["theme"]))
    total_groups = len(ordered)
    if display_limit != "All":
        ordered = ordered[:display_limit]
    figure = go.Figure(
        go.Bar(
            x=[int(row[metric]) for row in ordered], y=[row["theme"] for row in ordered], orientation="h", marker_color="#3266cc",
            customdata=[[row["share_of_code_occurrences"], row["segments"], row["records"]] for row in ordered],
            hovertemplate="%{y}<br>Value: %{x:,}<br>Share of code occurrences: %{customdata[0]:.1%}<br>Segments: %{customdata[1]:,}<br>Records: %{customdata[2]:,}<extra></extra>",
        )
    ).update_layout(height=min(2400, max(520, len(ordered) * 28)), yaxis=dict(autorange="reversed"), xaxis_title=metric_label, margin=dict(l=310, r=20, t=25, b=45))
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
    limit_caption = "all" if display_limit == "All" else f"the top {display_limit}"
    filter_caption = "The shared final-theme filter limits the underlying assignments before aggregation. " if active_themes else ""
    st.caption(
        f"Ranking {selected_level}; showing {limit_caption} of {total_groups:,} groups. "
        f"{filter_caption}Theme and Hierarchy Explorer selectors are independent; Evidence Explorer remains unchanged."
    )
    st.subheader("Hierarchy Sankey")
    if grouping_column == "initial_code":
        st.info("Initial codes are already the leaf level; choose a clustering level to view parent paths.")
    else:
        displayed_paths = st.select_slider(
            "Sankey detail: number of leading initial-code paths",
            options=[10, 20, 30, 50, 75, 100, 150],
            value=50,
            key="overview_sankey_detail",
            help="Use a smaller number to inspect a clean all-group diagram. The Sankey expands vertically as more paths are shown.",
        )
        top = all_code_rows(results, final_themes=active_themes, limit=displayed_paths)
        filtered_results = dict(results)
        filtered_results["codes"] = top
        edges = lineage_edges(
            filtered_results,
            final_themes=active_themes,
            max_level=int(grouping_column.split("_")[1]),
        )
        sankey = _sankey(
            filtered_results,
            edges,
            f"All {selected_level} groups: {len(top):,} most frequent initial-code paths",
        )
        if sankey:
            st.plotly_chart(sankey, use_container_width=True, config={"displaylogo": False})
        st.caption(
            f"Showing all selected-level groups represented by the {len(top):,} visible initial-code paths. Shares use code occurrences visible in this Sankey. The final-theme sidebar filter applies before path selection."
        )


def _level_context(results, label, key):
    levels = ranking_levels(results["assignments"])
    labels = [name for name, _ in levels]
    selected = st.selectbox(
        label,
        labels,
        index=len(labels) - 1,
        key=key,
        help="Choose initial codes or any retained clustering iteration.",
    )
    grouping_column = dict(levels)[selected]
    depth = 0 if grouping_column == "initial_code" else int(grouping_column.split("_")[1])
    return selected, grouping_column, depth


def _group_summary(results, grouping_column, active_themes):
    summary = ranking_summary(results, grouping_column, final_themes=active_themes)
    if not summary:
        st.info("No groups match the current final-theme filter.")
        st.stop()
    return summary


def _default_group(results, summary, grouping_column, active_themes):
    if len(active_themes) == 1:
        descendants = {
            row[grouping_column]
            for row in results["assignments"]
            if row["final_theme"] == active_themes[0]
        }
        for row in summary:
            if row["theme"] in descendants:
                return row["theme"]
    return summary[0]["theme"]


def _theme_explorer(results, active_themes):
    selected_level, grouping_column, _ = _level_context(
        results, "Inspect clustering level", "theme_explorer_level"
    )
    summaries = _group_summary(results, grouping_column, active_themes)
    groups = [row["theme"] for row in summaries]
    default = _default_group(results, summaries, grouping_column, active_themes)
    group_label = "Theme" if grouping_column == level_columns(results["assignments"])[-1] else f"{selected_level} group"
    group = st.selectbox(group_label, groups, index=groups.index(default), key="theme_explorer_group")
    summary = next(row for row in summaries if row["theme"] == group)
    columns = st.columns(4)
    for column, (label, key) in zip(columns, (("Code occurrences", "code_occurrences"), ("Unique initial codes", "unique_initial_codes"), ("Distinct segments", "segments"), ("Distinct records", "records"))):
        column.metric(label, f"{int(summary[key]):,}")
    st.subheader(f"Leading initial codes in {group}")
    codes = group_code_rows(results, grouping_column, group, final_themes=active_themes)
    st.dataframe(codes, use_container_width=True, height=340, hide_index=True)
    st.download_button("Download all codes for this group", _csv_bytes(codes), f"{group[:60]}_codes.csv", "text/csv")
    st.subheader(f"Representative source evidence for {group}")
    for example in group_examples(results, grouping_column, group, final_themes=active_themes):
        with st.expander(f"{example['segment_id']} · {example['theme_code_count']} codes"):
            st.caption(example["record_id"])
            st.write(example["segment_text"])
    st.subheader(f"All matching assignments for {group}")
    rows = group_assignments(results, grouping_column, group, final_themes=active_themes)
    st.dataframe(rows[:500], use_container_width=True, height=360, hide_index=True)
    filter_caption = " Final-theme filtering is active." if active_themes else ""
    st.caption(f"Showing the first 500 of {len(rows):,} assignments for {selected_level}.{filter_caption}")
    st.download_button("Download group assignments", _csv_bytes(rows), f"{group[:60]}_assignments.csv", "text/csv")


def _hierarchy_explorer(results, active_themes):
    mode = st.radio("Trace direction", ["Final theme → initial codes", "Initial code → final theme"], horizontal=True)
    if mode == "Final theme → initial codes":
        themes = [row["final_theme"] for row in sorted_summary(results)]
        default = active_themes[0] if len(active_themes) == 1 else themes[0]
        theme = st.selectbox("Final theme to trace", themes, index=themes.index(default), key="lineage_theme")
        displayed_paths = st.select_slider(
            "Detail level: number of leading initial-code paths",
            options=[10, 20, 30, 50, 75, 100, 150],
            value=30,
            help="Use a smaller number to inspect a clean diagram. The Sankey expands vertically as you add paths.",
        )
        top = theme_codes(results, theme, limit=displayed_paths)
        focused = {row["initial_code"] for row in top}
        filtered_results = dict(results)
        filtered_results["codes"] = top
        figure = _sankey(filtered_results, lineage_edges(filtered_results, theme=theme), f"{theme}: {len(focused):,} most frequent initial-code paths")
        st.caption(
            f"Sankey diagrams do not support true pan/zoom. Use this detail control instead: the view starts at 30 readable paths and grows vertically as more are shown. Hover a shortened label for its full text and its code-occurrence, segment, and record measures. Shares use the code occurrences visible in the current Sankey. The complete {int(next(row for row in results['summary'] if row['final_theme'] == theme)['unique_initial_codes']):,}-code mapping is downloadable in Theme explorer."
        )
    else:
        all_codes = sorted(results["hierarchy"]["code_paths"])
        code = st.selectbox("Initial code", all_codes, key="lineage_code")
        figure = _sankey(results, lineage_edges(results, initial_code=code), f"Exact path for: {code}")
        st.code(" → ".join([code] + results["hierarchy"]["code_paths"][code]))
    if figure:
        st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})


def _evidence_explorer(results, active_themes):
    all_types = sorted({row["segment_type"] for row in results["assignments"]})
    segment_types = st.multiselect("Segment types", all_types, default=all_types)
    query = st.text_input("Search source text, code, record ID, or segment ID")
    available_codes = sorted({row["initial_code"] for row in filter_assignments(results, themes=active_themes, segment_types=segment_types)})
    initial_code = st.selectbox("Exact initial code (optional)", [""] + available_codes, format_func=lambda value: value or "All codes")
    rows = filter_assignments(results, themes=active_themes, initial_code=initial_code or None, segment_types=segment_types, query=query)
    st.write(f"**{len(rows):,}** matching code occurrences")
    st.dataframe(rows[:500], use_container_width=True, height=420, hide_index=True)
    st.download_button("Download filtered evidence", _csv_bytes(rows), "hicode_filtered_assignments.csv", "text/csv")
    if rows:
        choices = {f"{row['segment_id']} · {row['initial_code']}": row for row in rows[:500]}
        chosen = st.selectbox("Inspect an evidence record", list(choices))
        row = choices[chosen]
        st.write(row["segment_text"])
        levels = [row[column] for column in level_columns(results["assignments"])]
        st.code(" → ".join([row["initial_code"]] + levels))


def main():
    st.set_page_config(page_title="HiCode Results Explorer", page_icon="◌", layout="wide")
    st.title("HiCode Results Explorer")
    st.caption("Read-only explorer for the completed analysis. No API calls and no modifications to HiCode outputs.")
    try:
        results = _load(_argument_input_dir())
    except Exception as exc:
        st.error(f"Could not load a coherent completed HiCode run: {exc}")
        st.stop()
    counts = results["run_config"]["counts"]
    metrics = st.columns(5)
    for column, label, value in zip(metrics, ("Records", "Segments", "Code occurrences", "Unique codes", "Final themes"), (counts["included_records"], counts["segments"], counts["code_occurrences"], counts["unique_codes"], counts["final_themes"])):
        column.metric(label, f"{value:,}")
    themes = [row["final_theme"] for row in sorted_summary(results)]
    with st.sidebar:
        st.header("Linked filters")
        active_themes = st.multiselect("Final themes", themes, help="Applies to overview, theme, hierarchy, and evidence views.")
        st.divider()
        st.subheader("Methodology")
        st.write(results["run_config"].get("coding_goal", ""))
        st.caption("Theme frequency counts code occurrences; clustering was performed on exact unique code strings.")
        st.caption(f"Source: {results['input_dir']}")
    overview, hierarchy, theme, evidence = st.tabs(["Overview", "Hierarchy explorer", "Theme explorer", "Evidence explorer"])
    with overview:
        _overview(results, active_themes)
    with hierarchy:
        _hierarchy_explorer(results, active_themes)
    with theme:
        _theme_explorer(results, active_themes)
    with evidence:
        _evidence_explorer(results, active_themes)


if __name__ == "__main__":
    main()
