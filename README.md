# HICode Alternative

This repo is an alternative implementation of the method presented in the [HICode: Hierarchical Inductive Coding with LLMs](https://arxiv.org/abs/2509.17946) paper by Mian Zhong, Pristina Wang & Anjalie Field. The intent is making it more practical to run, and providing a rich visualization dashboard.

**This implementation introduces some differences that may yield different results. For the original implementation, refer to the [Original Repo](https://github.com/mianzg/HICode)**. This is similar enough for personal use, but not for repoduction or evaluation.

Run hierarchical inductive coding on any CSV text corpus, then inspect the resulting themes and source evidence. The caller supplies the CSV, the text column, and a research-specific prompt profile.

## Quick start

Run commands from the repository root. Input, output, prompt, and run identity
parameters are required. Processing limits and concurrency settings have
built-in defaults.

The repository is organized as follows:

```text
data/                    example source datasets
configs/prompt_profiles/ prompt profiles used by the pipeline (ignored by gitignore)
src/hicode/              importable pipeline and dashboard code
tests/                   automated tests
docs/                    paper and methodology notes
output/                  local generated run artifacts (ignored by gitignore)
```

There is an example run for clothing reviews included in the repo.

### 1. Install with UV

```bash
uv sync
```

### 2. Configure Environment Variables

Create `.env`:

```bash
cp .env.template
```

Then, edit with your own credentials and settings.

### 3. Run a small check

The checked-in example CSV has a single `Review_Text` column. Any CSV can be used by
passing the name of its text column with `--text-column`.

```bash
uv run hicode \
  --input-csv data/clothing_reviews.csv \
  --text-column Review_Text \
  --run-name smoke-30 \
  --mode new \
  --runs-dir output/runs \
  --prompt-file configs/prompt_profiles/clothing_reviews.json \
  --max-usable-records 30 \
  --max-cluster-iterations 5
```

Prompt profiles are deliberately explicit because the research question and coding instructions vary by corpus. The included profiles are examples and can be replaced with any JSON profile matching the documented schema. `configs/prompt_profiles/template.json` is a paper prompt template, not a ready-to-run profile. Its background and research-question fields intentionally contain `<ADD YOUR OWN: ...>` placeholders. Replace both with the corpus-specific components specified by the paper before running; HiCode rejects the unchanged template.

You can also alter the rest of the prompt profile to deviate further from HICode if desired.

### 4. Run the full corpus

Use a new run name. By default, the full corpus is processed:

```bash
uv run hicode \
  --input-csv path/to/your_records.csv \
  --text-column body \
  --run-name analysis-v1 \
  --mode new \
  --runs-dir output/runs \
  --prompt-file configs/prompt_profiles/your_profile.json
  --max-cluster-iterations N
```

`--max-usable-records` can impose a limit after blank values are skipped.

### 5. Resume an interrupted run

Pass the same input, text column, prompt profile, run name, and algorithm
parameters again with `--mode resume`:

```bash
uv run hicode \
  --input-csv path/to/records.csv \
  --text-column body \
  --run-name analysis-v1 \
  --mode resume \
  --runs-dir output/runs \
  --prompt-file configs/prompt_profiles/your_profile.json
```

A fresh run never overwrites an existing run, and a completed run cannot be
resumed. Worker counts may change when resuming; completed checkpoints are
reused. Algorithm parameters are part of the run configuration and must remain
the same when resuming.

### 6. Explore completed results

```bash
uv run streamlit run dashboard.py -- \
  --input-dir output/runs/analysis-v1
```

Already recorded example (clothing reviews):
```bash
uv run streamlit run dashboard.py -- \
  --input-dir output/runs/clothing_reviews
```

The dashboard is read-only and makes no API calls.

## Runner parameters

```text
uv run hicode [options]
```

| Parameter | Required? | Meaning |
| --- | --- | --- |
| `--input-csv PATH` | Yes | Source CSV. |
| `--text-column NAME` | Yes | CSV column containing the text to code. |
| `--run-name NAME` | Yes | Output run name. A `new` run name must not already exist. |
| `--mode new\|resume` | Yes | `new` creates a run; `resume` continues an incomplete matching run. |
| `--runs-dir PATH` | Yes | Parent directory for named runs. |
| `--prompt-file PATH` | Yes | JSON research prompt profile. |
| `--max-cluster-iterations N` | Yes | Maximum number of hierarchy levels and safety cap. |
| `--max-usable-records N` | No | Positive integer limit after normalization. Blank rows do not count. Default: unlimited. |
| `--generation-workers N` | No | Number of concurrent label-generation requests. Default: 8. |
| `--clustering-workers N` | No | Number of concurrent clustering-batch requests per hierarchy level. Default: 4. |
| `--cluster-batch-size N` | No | Number of labels sent to each clustering request. Default: 100. |
| `--target-final-themes N` | No | Optional stopping threshold; clustering stops at or below this number of themes. Default: disabled. |

## Text preprocessing

HiCode normalizes Unicode, line endings, control characters, and whitespace.
It skips blank values, preserves meaningful punctuation/hashtags/mentions/emoji,
and splits long text into sentence-aware chunks no larger than the configured
maximum. It does not apply platform-specific parsing or remove duplicate source
records.

## Outputs

Each completed run contains:

- `cleaned_records.csv`, `segments.csv`, and `excluded_records.csv`: normalized
  input records, model segments, and exclusions.
- `generation.json` and `irrelevant_segments.csv`: generated initial codes and
  irrelevant segments.
- `assignments.csv`, `theme_summary.csv`, `theme_codes.csv`, and
  `theme_examples.csv`: code-to-theme assignments, counts, and evidence.
- `hierarchy.json`, `clustering/`, `run_manifest.json`, and `run_config.json`:
  hierarchy, exact clustering outputs, run identity, and configuration.
- `execution_attempts.json` and stage checkpoint directories: resumable
  execution history and completed concurrent work.

Counts are code occurrences: one segment can receive multiple codes.

## Tests

Run the test suite from the repository root:

```bash
uv run python -m unittest discover -s tests -t .
```
