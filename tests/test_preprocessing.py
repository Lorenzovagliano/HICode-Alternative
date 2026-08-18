import csv
from pathlib import Path
import tempfile
import unittest

from hicode.preprocessing import (
    MAX_SEGMENT_CHARS,
    normalize_text,
    preprocess_records,
    segment_text,
)


class PreprocessingTest(unittest.TestCase):
    def test_normalize_text_preserves_meaningful_text(self):
        raw = "\ufeff  #tag   @person 😄‍👩‍👧\r\n\x00 Keep punctuation!?  "
        self.assertEqual(
            normalize_text(raw),
            "#tag @person 😄‍👩‍👧\nKeep punctuation!?",
        )
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(" \t\r\n "), "")

    def test_segmentation_is_sentence_aware_and_bounded(self):
        text = "First sentence. Second sentence! " + ("x" * 1500)
        segments = segment_text(text, target_chars=100, max_chars=200)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(0 < len(segment) <= 200 for segment in segments))

        oversized_token = segment_text("z" * 25, target_chars=10, max_chars=10)
        self.assertEqual(oversized_token, ["z" * 10, "z" * 10, "z" * 5])

    def test_preprocess_records_uses_selected_column_and_generic_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "records.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "body"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"id": "a", "body": "  first #signal  "},
                        {"id": "b", "body": ""},
                        {"id": "c", "body": '{"utterances": "literal text"}'},
                    ]
                )

            records, segments, excluded, report = preprocess_records(
                source,
                max_usable_records=None,
                text_column="body",
            )

            self.assertEqual([row["record_id"] for row in records], ["record_000000", "record_000002"])
            self.assertEqual([row["record_id"] for row in segments], ["record_000000", "record_000002"])
            self.assertTrue(all(row["segment_type"] == "text" for row in segments))
            self.assertIn('{"utterances": "literal text"}', segments[-1]["text"])
            self.assertEqual(excluded, [])
            self.assertEqual(report["blank_records"], 1)
            self.assertEqual(report["cleaned_records"], 2)

            with self.assertRaisesRegex(ValueError, "missing"):
                preprocess_records(source, None, "missing")

    def test_preprocess_records_never_exceeds_default_segment_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "records.csv"
            source.write_text("text\n" + ("word " * 500) + "\n", encoding="utf-8")
            _, segments, _, _ = preprocess_records(source, None, "text")
            self.assertTrue(all(len(segment["text"]) <= MAX_SEGMENT_CHARS for segment in segments))


if __name__ == "__main__":
    unittest.main()
