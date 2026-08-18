import unittest

from hicode.visualization import (
    all_code_rows,
    filter_assignments,
    group_assignments,
    group_code_rows,
    group_examples,
    hierarchy_nodes,
    lineage_edges,
    lineage_metrics,
    ranking_levels,
    ranking_summary,
    sorted_summary,
    theme_codes,
    theme_examples,
    validate_results,
)


def fixture_results():
    assignments = [
        {
            "record_id": "record_000001", "segment_id": "record_000001_0", "segment_type": "text",
            "segment_text": "Alpha evidence", "initial_code": "Alpha", "level_1_theme": "Group A", "level_2_theme": "Theme A", "final_theme": "Theme A",
        },
        {
            "record_id": "record_000001", "segment_id": "record_000001_0", "segment_type": "text",
            "segment_text": "Alpha evidence", "initial_code": "Beta", "level_1_theme": "Group A", "level_2_theme": "Theme A", "final_theme": "Theme A",
        },
        {
            "record_id": "record_000002", "segment_id": "record_000002_0", "segment_type": "text",
            "segment_text": "Gamma evidence", "initial_code": "Gamma", "level_1_theme": "Group B", "level_2_theme": "Theme B", "final_theme": "Theme B",
        },
    ]
    return {
        "assignments": assignments,
        "summary": [
            {"final_theme": "Theme A", "code_occurrences": "2", "unique_initial_codes": "2", "segments": "1", "records": "1", "share_of_code_occurrences": "0.6667"},
            {"final_theme": "Theme B", "code_occurrences": "1", "unique_initial_codes": "1", "segments": "1", "records": "1", "share_of_code_occurrences": "0.3333"},
        ],
        "codes": [
            {"final_theme": "Theme A", "initial_code": "Alpha", "occurrences": "1", "hierarchy_path": "Alpha -> Group A -> Theme A"},
            {"final_theme": "Theme A", "initial_code": "Beta", "occurrences": "1", "hierarchy_path": "Beta -> Group A -> Theme A"},
            {"final_theme": "Theme B", "initial_code": "Gamma", "occurrences": "1", "hierarchy_path": "Gamma -> Group B -> Theme B"},
        ],
        "examples": [], "segments": assignments,
        "hierarchy": {
            "iterations": [{"Group A": ["Alpha", "Beta"], "Group B": ["Gamma"]}, {"Theme A": ["Group A"], "Theme B": ["Group B"]}],
            "code_paths": {"Alpha": ["Group A", "Theme A"], "Beta": ["Group A", "Theme A"], "Gamma": ["Group B", "Theme B"]},
            "final_theme_to_initial_codes": {"Theme A": ["Alpha", "Beta"], "Theme B": ["Gamma"]},
        },
        "run_config": {"counts": {"code_occurrences": 3, "unique_codes": 3, "final_themes": 2}},
        "preprocessing_report": {},
    }


class VisualizationHelpersTest(unittest.TestCase):
    def test_validation_and_bidirectional_lineage(self):
        results = fixture_results()
        validate_results(results)
        nodes = hierarchy_nodes(results, themes=["Theme A"])
        self.assertEqual(next(row for row in nodes if row["id"] == "root")["value"], 2)
        edges = lineage_edges(results, initial_code="Alpha")
        self.assertEqual([(edge["source"], edge["target"]) for edge in edges], [("0:Alpha", "1:Group A"), ("1:Group A", "2:Theme A")])
        metrics = lineage_metrics(results, edges)
        self.assertEqual(metrics["total_code_occurrences"], 1)
        self.assertEqual(metrics["nodes"]["1:Group A"], {
            "code_occurrences": 1,
            "share_of_code_occurrences": 1.0,
            "segments": 1,
            "records": 1,
        })

    def test_deterministic_evidence_filtering(self):
        results = fixture_results()
        filtered = filter_assignments(results, themes=["Theme A"], query="alpha evidence")
        self.assertEqual([row["initial_code"] for row in filtered], ["Alpha", "Beta"])
        self.assertEqual(filter_assignments(results, segment_types=["text"])[0]["initial_code"], "Alpha")

    def test_ranking_levels_include_initial_and_all_clustering_iterations(self):
        self.assertEqual(
            ranking_levels(fixture_results()["assignments"]),
            [
                ("Initial codes", "initial_code"),
                ("Level 1", "level_1_theme"),
                ("Level 2 (final themes)", "level_2_theme"),
            ],
        )

    def test_ranking_summary_aggregates_metrics_and_merges_children(self):
        results = fixture_results()
        summary = ranking_summary(results, "level_1_theme")
        self.assertEqual([row["theme"] for row in summary], ["Group A", "Group B"])
        group_a = summary[0]
        self.assertEqual(group_a["code_occurrences"], 2)
        self.assertEqual(group_a["unique_initial_codes"], 2)
        self.assertEqual(group_a["segments"], 1)
        self.assertEqual(group_a["records"], 1)
        self.assertAlmostEqual(group_a["share_of_code_occurrences"], 2 / 3)

        initial_codes = ranking_summary(results, "initial_code")
        self.assertEqual([row["theme"] for row in initial_codes], ["Alpha", "Beta", "Gamma"])
        self.assertTrue(all(row["unique_initial_codes"] == 1 for row in initial_codes))

    def test_ranking_summary_uses_descendant_filter_and_global_share(self):
        summary = ranking_summary(fixture_results(), "level_1_theme", final_themes=["Theme B"])
        self.assertEqual(summary, [{
            "theme": "Group B",
            "code_occurrences": 1,
            "unique_initial_codes": 1,
            "segments": 1,
            "records": 1,
            "share_of_code_occurrences": 1 / 3,
        }])

    def test_group_helpers_aggregate_codes_examples_and_assignments(self):
        results = fixture_results()
        rows = group_assignments(results, "level_1_theme", "Group A")
        self.assertEqual([row["initial_code"] for row in rows], ["Alpha", "Beta"])

        codes = group_code_rows(results, "level_1_theme", "Group A")
        self.assertEqual(codes, [
            {
                "final_theme": "Theme A",
                "initial_code": "Alpha",
                "occurrences": "1",
                "hierarchy_path": "Alpha -> Group A -> Theme A",
            },
            {
                "final_theme": "Theme A",
                "initial_code": "Beta",
                "occurrences": "1",
                "hierarchy_path": "Beta -> Group A -> Theme A",
            },
        ])
        examples = group_examples(results, "level_1_theme", "Group A")
        self.assertEqual([row["segment_id"] for row in examples], ["record_000001_0"])
        self.assertEqual(examples[0]["theme_code_count"], 2)

    def test_group_helpers_apply_final_theme_descendant_filter(self):
        results = fixture_results()
        self.assertEqual(
            [row["initial_code"] for row in group_assignments(
                results, "level_1_theme", "Group A", final_themes=["Theme B"]
            )],
            [],
        )
        self.assertEqual(
            group_code_rows(results, "level_1_theme", "Group B", final_themes=["Theme B"])[0]["initial_code"],
            "Gamma",
        )

    def test_global_code_rows_are_ordered_and_filterable(self):
        results = fixture_results()
        self.assertEqual(all_code_rows(results), results["codes"])
        self.assertEqual(
            [row["initial_code"] for row in all_code_rows(results, final_themes=["Theme A"])],
            ["Alpha", "Beta"],
        )
        self.assertEqual(
            [row["initial_code"] for row in all_code_rows(results, limit=2)],
            ["Alpha", "Beta"],
        )

    def test_overview_lineage_edges_include_all_groups_and_stop_at_level(self):
        results = fixture_results()
        edges = lineage_edges(
            results,
            max_level=1,
        )
        self.assertEqual(
            [(edge["source"], edge["target"], edge["value"]) for edge in edges],
            [
                ("0:Alpha", "1:Group A", 1),
                ("0:Beta", "1:Group A", 1),
                ("0:Gamma", "1:Group B", 1),
            ],
        )
        metrics = lineage_metrics(results, edges)
        self.assertEqual(metrics["total_code_occurrences"], 3)
        self.assertEqual(metrics["nodes"]["1:Group A"]["segments"], 1)
        self.assertEqual(
            lineage_edges(
                results,
                final_themes=["Theme B"],
                max_level=1,
            ),
            [{"source": "0:Gamma", "target": "1:Group B", "value": 1}],
        )
        self.assertEqual(lineage_edges(results, max_level=0), [])

    def test_final_group_helpers_and_paths_match_existing_behavior(self):
        results = fixture_results()
        existing_codes = theme_codes(results, "Theme A")
        selected_codes = group_code_rows(results, "level_2_theme", "Theme A")
        self.assertEqual(selected_codes, existing_codes)
        self.assertEqual(
            group_examples(results, "level_2_theme", "Theme A"),
            theme_examples(results, "Theme A"),
        )
        self.assertEqual(
            {row["label"] for row in hierarchy_nodes(results)},
            {"All selected code occurrences", "Group A", "Group B", "Theme A", "Theme B"},
        )
        self.assertEqual(
            lineage_edges(results, theme="Theme A"),
            lineage_edges(
                results,
                max_level=2,
                final_themes=["Theme A"],
            ),
        )

    def test_final_ranking_matches_existing_summary(self):
        results = fixture_results()
        existing = sorted_summary(results)
        selected = ranking_summary(results, "level_2_theme")
        self.assertEqual(
            [(row["theme"], row["code_occurrences"], row["unique_initial_codes"], row["segments"], row["records"])
             for row in selected],
            [(row["final_theme"], int(row["code_occurrences"]), int(row["unique_initial_codes"]), int(row["segments"]), int(row["records"]))
             for row in existing],
        )
        for selected_row, existing_row in zip(selected, existing):
            self.assertAlmostEqual(
                selected_row["share_of_code_occurrences"],
                float(existing_row["share_of_code_occurrences"]),
                places=3,
            )

    def test_hierarchy_nodes_keep_repeated_labels_in_separate_branches(self):
        results = fixture_results()
        results["hierarchy"]["code_paths"].update(
            {
                "Alpha": ["Group A", "Shared theme"],
                "Beta": ["Group A", "Shared theme"],
                "Gamma": ["Group B", "Shared theme"],
            }
        )

        nodes = hierarchy_nodes(results)
        shared = [node for node in nodes if node["label"] == "Shared theme"]
        self.assertEqual(len(shared), 2)
        self.assertEqual({node["parent"] for node in shared}, {"1:[\"Group A\"]", "1:[\"Group B\"]"})

        children = {}
        for node in nodes:
            children.setdefault(node["parent"], []).append(node)
        for node in nodes:
            if node["id"] in children:
                self.assertEqual(sum(child["value"] for child in children[node["id"]]), node["value"])

    def test_validation_rejects_bad_config_counts(self):
        results = fixture_results()
        results["run_config"]["counts"]["unique_codes"] = 99
        with self.assertRaisesRegex(ValueError, "unique_codes"):
            validate_results(results)


if __name__ == "__main__":
    unittest.main()
