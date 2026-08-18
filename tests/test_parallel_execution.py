import contextlib
import csv
import io
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hicode import concurrency
from hicode import clustering
from hicode import generation
from hicode import pipeline
from hicode.run_management import finish_execution_attempt, start_execution_attempt
from hicode.generation import _text_sha256


def _chat_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class BoundedExecutionTest(unittest.TestCase):
    def test_worker_limit_and_out_of_order_results(self):
        lock = threading.Lock()
        active = 0
        maximum = 0

        def worker(task_id, delay):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(delay)
            with lock:
                active -= 1
            return task_id

        results = concurrency.run_bounded_tasks(
            [(0, 0.04), (1, 0.01), (2, 0.03), (3, 0.01)],
            worker,
            max_workers=2,
            stage="test",
        )
        self.assertEqual(maximum, 2)
        self.assertEqual(set(results), {0, 1, 2, 3})

    def test_one_worker_preserves_task_order(self):
        called = []

        def worker(task_id, _payload):
            called.append(task_id)
            return task_id

        self.assertEqual(
            concurrency.run_bounded_tasks(
                [(0, None), (1, None), (2, None)], worker, 1, "test"
            ),
            {0: 0, 1: 1, 2: 2},
        )
        self.assertEqual(called, [0, 1, 2])

    def test_failure_stops_new_scheduling_after_active_work(self):
        called = []

        def worker(task_id, _payload):
            called.append(task_id)
            if task_id == 1:
                raise RuntimeError("boom")
            time.sleep(0.03)
            return task_id

        with self.assertRaises(concurrency.ConcurrentTaskError) as context:
            concurrency.run_bounded_tasks(
                [(0, None), (1, None), (2, None), (3, None)],
                worker,
                max_workers=2,
                stage="test",
            )
        self.assertEqual(set(called), {0, 1})
        self.assertEqual(context.exception.completed, [0])
        self.assertEqual(context.exception.unscheduled, [2, 3])


class FakeGenerationClient:
    def __init__(self):
        self.calls = []

        class Completions:
            def __init__(inner, outer):
                inner.outer = outer

            def create(inner, **request):
                text = request["messages"][-1]["content"]
                inner.outer.calls.append(text)
                time.sleep(0.01 if text.endswith("0") else 0.03)
                return _chat_response(f"LABEL: [Signal {text[-1]}]")

        self.chat = SimpleNamespace(completions=Completions(self))


class FakeClusteringClient:
    def __init__(self):
        self.calls = []

        class Completions:
            def __init__(inner, outer):
                inner.outer = outer

            def create(inner, **request):
                labels = json.loads(request["messages"][-1]["content"])
                inner.outer.calls.append(labels)
                time.sleep(0.01 if labels[0].endswith("2") else 0.03)
                return _chat_response(json.dumps({f"Theme {labels[0]}": labels}))

        self.chat = SimpleNamespace(completions=Completions(self))


class TrackingClusteringClient:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.active_iteration = None
        self.overlapped_iterations = False
        self.lock = threading.Lock()

        class Completions:
            def __init__(inner, outer):
                inner.outer = outer

            def create(inner, **request):
                outer = inner.outer
                iteration = request["messages"][0]["content"]
                labels = json.loads(request["messages"][-1]["content"])
                with outer.lock:
                    outer.active += 1
                    outer.max_active = max(outer.max_active, outer.active)
                    if outer.active_iteration is not None and outer.active_iteration != iteration:
                        outer.overlapped_iterations = True
                    outer.active_iteration = iteration
                    outer.calls.append(labels)
                try:
                    time.sleep(0.02)
                    return _chat_response(json.dumps({f"Theme {labels[0]}": labels}))
                finally:
                    with outer.lock:
                        outer.active -= 1
                        if outer.active == 0:
                            outer.active_iteration = None

        self.chat = SimpleNamespace(completions=Completions(self))


class ParallelStagesTest(unittest.TestCase):
    def setUp(self):
        generation._CLIENT_LOCAL.client = None
        clustering._CLIENT_LOCAL.client = None

    def test_generation_is_ordered_and_resumable(self):
        fake = FakeGenerationClient()
        data = {
            "record_000001_0": "signal 0",
            "record_000001_1": "signal 1",
            "record_000001_2": "signal 2",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ), patch("hicode.generation.OpenAI", return_value=fake):
            config = {
                "model_name": "gpt-test",
                "max_workers": 3,
                "max_generation_attempts": 1,
                "checkpoint_dir": Path(tmp) / "generation" / "checkpoints",
                "failure_path": Path(tmp) / "generation" / "failures.json",
            }
            first = generation.generate_labels(data, "prompt", config)
            self.assertEqual(
                [
                    annotation["segment_id"]
                    for annotation in first["record_000001"]["LLM_Annotation"]
                ],
                list(data),
            )
            self.assertEqual(len(fake.calls), 3)

            generation._CLIENT_LOCAL.client = None
            second = generation.generate_labels(
                data,
                "prompt",
                {**config, "resume_generation": True},
            )
            self.assertEqual(first, second)
            self.assertEqual(len(fake.calls), 3)

    def test_generation_failure_checkpoints_active_successes(self):
        class FailingClient:
            def __init__(self):
                self.calls = []

                class Completions:
                    def __init__(inner, outer):
                        inner.outer = outer

                    def create(inner, **request):
                        text = request["messages"][-1]["content"]
                        inner.outer.calls.append(text)
                        if text == "fail":
                            raise RuntimeError("exhausted")
                        time.sleep(0.02)
                        return _chat_response("LABEL: [Signal]")

                self.chat = SimpleNamespace(completions=Completions(self))

        fake = FailingClient()
        data = {
            "record_000001_0": "ok",
            "record_000001_1": "fail",
            "record_000001_2": "unscheduled 2",
            "record_000001_3": "unscheduled 3",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ), patch("hicode.generation.OpenAI", return_value=fake):
            checkpoint_dir = Path(tmp) / "generation" / "checkpoints"
            with self.assertRaises(RuntimeError):
                generation.generate_labels(
                    data,
                    "prompt",
                    {
                        "model_name": "gpt-test",
                        "max_workers": 2,
                        "max_generation_attempts": 1,
                        "checkpoint_dir": checkpoint_dir,
                        "failure_path": Path(tmp) / "generation" / "failures.json",
                    },
                )
            self.assertTrue((checkpoint_dir / "segment_00000000.json").is_file())
            failure = json.loads(
                (Path(tmp) / "generation" / "failures.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["unscheduled"], [2, 3])

    def test_generation_rejects_mismatched_checkpoint_and_omits_irrelevant(self):
        fake = FakeGenerationClient()
        data = {
            "record_000001_0": "irrelevant text",
            "record_000001_1": "signal 1",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ), patch("hicode.generation.OpenAI", return_value=fake):
            checkpoint_dir = Path(tmp) / "generation" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint_path = checkpoint_dir / "segment_00000000.json"
            checkpoint = {
                "segment_index": 0,
                            "segment_id": "record_000001_0",
                "text_sha256": _text_sha256("changed text"),
                "labels": [],
            }
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            config = {
                "model_name": "gpt-test",
                "max_workers": 2,
                "max_generation_attempts": 1,
                "checkpoint_dir": checkpoint_dir,
                "resume_generation": True,
            }
            with self.assertRaisesRegex(ValueError, "source text"):
                generation.generate_labels(data, "prompt", config)

            checkpoint["text_sha256"] = _text_sha256(data["record_000001_0"])
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            output = generation.generate_labels(data, "prompt", config)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(
                [item["segment_id"] for item in output["record_000001"]["LLM_Annotation"]],
                ["record_000001_1"],
            )

    def test_clustering_is_ordered_and_resumable(self):
        fake = FakeClusteringClient()
        generation = {
                "record": {
                "LLM_Annotation": [
                    {"label": ["label 0", "label 1", "label 2", "label 3"]}
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ), patch("hicode.clustering.OpenAI", return_value=fake):
            config = {
                "cluster_model_name": "gpt-test",
                "cluster_run_dir": str(Path(tmp) / "clustering"),
                "batch_size": 2,
                "max_workers": 2,
                "max_n_iter": 1,
                "max_cluster_attempts": 1,
                "random_seed": 42,
                "checkpoint_batches": True,
            }
            first = clustering.cluster_labels_gpt(
                generation, "prompt", config, gen_result_id="parallel"
            )
            self.assertEqual(len(fake.calls), 2)
            expected_order = ["label 0", "label 1", "label 2", "label 3"]
            random.Random(42).shuffle(expected_order)
            self.assertEqual(
                list(first[0].values())[0], expected_order[:2]
            )

            clustering._CLIENT_LOCAL.client = None
            second = clustering.cluster_labels_gpt(
                generation,
                "prompt",
                {**config, "resume_clustering": True},
                gen_result_id="parallel",
            )
            self.assertEqual(first, second)
            self.assertEqual(len(fake.calls), 2)

    def test_clustering_resumes_partial_batches_and_rejects_mismatch(self):
        fake = FakeClusteringClient()
        generation = {
                "record": {
                "LLM_Annotation": [
                    {"label": ["label 0", "label 1", "label 2", "label 3"]}
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ), patch("hicode.clustering.OpenAI", return_value=fake):
            cluster_dir = Path(tmp) / "clustering"
            checkpoint_dir = cluster_dir / "checkpoints" / "iter_0"
            checkpoint_dir.mkdir(parents=True)
            ordered = sorted(["label 0", "label 1", "label 2", "label 3"])
            random.Random(42).shuffle(ordered)
            checkpoint_path = checkpoint_dir / "batch_00000.json"
            checkpoint = {
                "iteration": 0,
                "batch_index": 0,
                "input_labels": ["wrong 0", "wrong 1"],
                "model_output": {"Precomputed": ordered[:2]},
            }
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            config = {
                "cluster_model_name": "gpt-test",
                "cluster_run_dir": str(cluster_dir),
                "batch_size": 2,
                "max_workers": 2,
                "max_n_iter": 1,
                "target_final_themes": 1,
                "max_cluster_attempts": 1,
                "random_seed": 42,
                "checkpoint_batches": True,
                "resume_clustering": True,
            }
            with self.assertRaisesRegex(ValueError, "deterministic input batch"):
                clustering.cluster_labels_gpt(
                    generation, "prompt", config, gen_result_id="partial"
                )

            checkpoint["input_labels"] = ordered[:2]
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
            result = clustering.cluster_labels_gpt(
                generation, "prompt", config, gen_result_id="partial"
            )
            self.assertEqual(len(fake.calls), 1)
            flattened = [label for children in result[0].values() for label in children]
            self.assertEqual(sorted(flattened), sorted(ordered))

    def test_clustering_iterations_do_not_overlap_and_keep_coverage(self):
        fake = TrackingClusteringClient()
        labels = [f"label {index}" for index in range(4)]
        generation = {"record": {"LLM_Annotation": [{"label": labels}]}}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ), patch("hicode.clustering.OpenAI", return_value=fake):
            result = clustering.cluster_labels_gpt(
                generation,
                lambda iteration: f"iteration {iteration}",
                {
                    "cluster_model_name": "gpt-test",
                    "cluster_run_dir": str(Path(tmp) / "clustering"),
                    "batch_size": 2,
                    "max_workers": 2,
                    "max_n_iter": 2,
                    "target_final_themes": 1,
                    "max_cluster_attempts": 1,
                    "random_seed": 42,
                    "checkpoint_batches": True,
                },
                gen_result_id="iterations",
            )
            self.assertEqual(len(result), 2)
            self.assertEqual(fake.max_active, 2)
            self.assertFalse(fake.overlapped_iterations)
            self.assertEqual(
                sorted(label for children in result[0].values() for label in children),
                sorted(labels),
            )
            self.assertEqual(
                sorted(label for children in result[1].values() for label in children),
                sorted(result[0]),
            )

    def test_attempt_history_records_changeable_worker_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first_id = start_execution_attempt(output_dir, "new", 8, 4)
            finish_execution_attempt(output_dir, first_id, "completed")
            second_id = start_execution_attempt(output_dir, "resume", 2, 1)
            finish_execution_attempt(
                output_dir, second_id, "failed", RuntimeError("test failure")
            )

            attempts = json.loads(
                (output_dir / "execution_attempts.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [(item["generation_workers"], item["clustering_workers"]) for item in attempts],
                [(8, 4), (2, 1)],
            )
            self.assertEqual([item["status"] for item in attempts], ["completed", "failed"])

    def test_pipeline_records_workers_and_cleans_generation_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ), patch.object(
            pipeline,
            "generate_labels",
            return_value={
                "record_000000": {
                    "LLM_Annotation": [
                        {
                            "segment_id": "record_000000_0",
                            "sentence": "hello",
                            "label": ["Signal"],
                        }
                    ]
                }
            },
        ) as generate_mock, patch.object(
            pipeline,
            "cluster_labels_gpt",
            return_value=[{"Theme": ["Signal"]}],
        ) as cluster_mock:
            source = Path(tmp) / "records.csv"
            source.write_text("text\nhello\n", encoding="utf-8")
            result = pipeline.run_pipeline(
                run_name="parallel-pipeline",
                mode="new",
                runs_dir=Path(tmp) / "runs",
                max_usable_records=1,
                input_csv=source,
                text_column="text",
                prompt_file=pipeline.BASE_DIR
                / "configs"
                / "prompt_profiles"
                / "clothing_reviews.json",
                generation_workers=3,
                clustering_workers=2,
                max_cluster_iterations=10,
            )

            self.assertEqual(generate_mock.call_args.args[2]["max_workers"], 3)
            self.assertEqual(cluster_mock.call_args.args[2]["max_workers"], 2)
            output_dir = Path(tmp) / "runs" / "parallel-pipeline"
            self.assertEqual(result["generation_workers"], 3)
            self.assertEqual(result["clustering_workers"], 2)
            cleaned_header = (output_dir / "cleaned_records.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertEqual(
                cleaned_header,
                "source_row_index,record_id,text,cleaning_reasons",
            )
            excluded_header = (output_dir / "excluded_records.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertEqual(
                excluded_header,
                "source_row_index,record_id,exclusion_reason,raw_text",
            )
            with (output_dir / "assignments.csv").open(
                encoding="utf-8", newline=""
            ) as assignments_file:
                assignments = list(csv.DictReader(assignments_file))
            self.assertEqual(assignments[0]["record_id"], "record_000000")
            self.assertEqual(assignments[0]["segment_type"], "text")
            with (output_dir / "theme_summary.csv").open(
                encoding="utf-8", newline=""
            ) as summary_file:
                summary = list(csv.DictReader(summary_file))
            self.assertIn("records", summary[0])
            self.assertFalse((output_dir / "generation" / "checkpoints").exists())
            attempts = json.loads(
                (output_dir / "execution_attempts.json").read_text(encoding="utf-8")
            )
            self.assertEqual(attempts[-1]["status"], "completed")

    def test_cli_requires_positive_worker_counts(self):
        common = [
            "--input-csv", "input.csv",
            "--text-column", "text",
            "--run-name", "run",
            "--mode", "new",
            "--runs-dir", "runs",
            "--prompt-file", "profile.json",
            "--max-cluster-iterations", "10",
            "--max-usable-records", "1",
        ]
        for option in ("--generation-workers", "--clustering-workers"):
            args = common + [
                "--generation-workers", "1",
                "--clustering-workers", "1",
            ]
            args[args.index(option) + 1] = "0"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline._parse_args(args)

    def test_cli_requires_max_iterations_and_parses_algorithm_parameters(self):
        common = [
            "--input-csv", "input.csv",
            "--text-column", "text",
            "--run-name", "run",
            "--mode", "new",
            "--runs-dir", "runs",
            "--prompt-file", "profile.json",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline._parse_args(common)
        args = pipeline._parse_args([
            *common,
            "--max-cluster-iterations", "10",
        ])
        self.assertIsNone(args.max_usable_records)
        self.assertEqual(args.generation_workers, 8)
        self.assertEqual(args.clustering_workers, 4)
        self.assertEqual(args.cluster_batch_size, 100)
        self.assertIsNone(args.target_final_themes)
        self.assertEqual(args.max_cluster_iterations, 10)


if __name__ == "__main__":
    unittest.main()
