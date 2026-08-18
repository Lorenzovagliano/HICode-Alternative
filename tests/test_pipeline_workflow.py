import csv
from pathlib import Path
import tempfile
import unittest

from hicode.run_management import (
    load_prompt_profile,
    make_manifest,
    resolve_run_directory,
    validate_manifest,
    write_manifest,
)
from hicode.clustering import make_clustering_prompt
from hicode.generation import make_generation_prompt
from hicode.pipeline import (
    BASE_DIR,
    _clustering_prompt_factory,
    _prepare_output_directory,
    preprocess_records,
    run_pipeline,
)


PROFILE_PATH = BASE_DIR / "configs" / "prompt_profiles" / "clothing_reviews.json"
PAPER_TEMPLATE_PATH = BASE_DIR / "configs" / "prompt_profiles" / "template.json"


class RunWorkflowTest(unittest.TestCase):
    def write_csv(self, directory, rows, fieldname="body"):
        path = Path(directory) / "records.csv"
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=[fieldname])
            writer.writeheader()
            writer.writerows([{fieldname: row} for row in rows])
        return path

    def test_custom_text_column_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_csv(tmp, ["first post", "second post", "third post"])
            cleaned, segments, _, report = preprocess_records(
                source, max_usable_records=10, text_column="body",
                include_record_ids=["record_000001", "record_000002"]
            )
            self.assertEqual([row["record_id"] for row in cleaned], ["record_000001", "record_000002"])
            self.assertEqual([row["record_id"] for row in segments], ["record_000001", "record_000002"])
            self.assertEqual(report["text_column"], "body")
            with self.assertRaisesRegex(ValueError, "missing"):
                preprocess_records(source, max_usable_records=10, text_column="missing")

    def test_manifest_validation_and_new_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.write_csv(tmp, ["a"])
            profile = load_prompt_profile(PROFILE_PATH)
            root, run_dir = resolve_run_directory(Path(tmp) / "runs", "trial-1")
            root.mkdir()
            run_dir.mkdir()
            manifest = make_manifest(
                run_name="trial-1", input_csv=source, text_column="body", profile=profile,
                settings={"seed": 42}, base_dir=BASE_DIR,
                selected_record_ids=None,
            )
            write_manifest(run_dir / "run_manifest.json", manifest)
            validate_manifest(run_dir / "run_manifest.json", manifest)
            changed = dict(manifest)
            changed["text_column"] = "other"
            with self.assertRaisesRegex(ValueError, "text_column"):
                validate_manifest(run_dir / "run_manifest.json", changed)
            with self.assertRaises(FileExistsError):
                run_pipeline(
                    run_name="trial-1", mode="new", runs_dir=root, input_csv=source,
                    text_column="body", prompt_file=PROFILE_PATH, max_usable_records=1,
                    generation_workers=1, clustering_workers=1,
                    max_cluster_iterations=10,
                )

    def test_safe_prepare_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            sentinel = run_dir / "generation.json"
            sentinel.write_text('{"keep": true}', encoding="utf-8")
            _prepare_output_directory(run_dir)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"keep": true}')

    def test_prompt_profile_levels_and_generic_fallback(self):
        profile = load_prompt_profile(PROFILE_PATH)
        factory = _clustering_prompt_factory(profile)
        self.assertIn("recurring customer experiences", factory(0))
        self.assertIn("list containing", factory(1))
        self.assertIn("insightful themes", factory(3))
        self.assertNotIn("Additional task guidance", make_generation_prompt("b", "g"))
        self.assertNotIn("Additional task guidance", make_clustering_prompt(goal="g"))

    def test_paper_template_rejects_unconfigured_placeholders(self):
        with self.assertRaisesRegex(ValueError, "replace it before running"):
            load_prompt_profile(PAPER_TEMPLATE_PATH)

if __name__ == "__main__":
    unittest.main()
