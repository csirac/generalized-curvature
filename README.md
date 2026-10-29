# Curvature Beyond Positivity: Greedy Guarantees for Arbitrary Submodular Functions

Experiment code for **Yixin Chen and Alan Kuhnle, NeurIPS 2026**, submission 32052.
The algorithms include greedy with pruning, standard and distorted greedy, fixed-pool
random greedy, local search, and measured continuous greedy. Signed-objective
baseline runs are heuristic; see the paper for the guarantees and their hypotheses.
D-optimal design, coverage, and log-determinant feature selection use removal-marginal
certificates. Large-instance ratios use a non-certified best-known reference.

## License

The code is released under the [MIT license](LICENSE).

## Results archive

Download both assets from the [v1.0 release](https://github.com/csirac/generalized-curvature/releases/tag/v1.0):

| Asset | SHA256 |
|---|---|
| `camera-ready-results.tar.gz` | `c57f8ede5306e708b182fa1db1120f1df8a92228841ee326c1c4e5928f157d8b` |
| `exp19_lambda_sweep.json` | `6da01de9c316ca1ce6329a0205c198ec4a33b931bd3d37c12d9031c06826625f` |

The archive contains the paper's raw results. Extract it into a new directory,
then point `GC_RESULTS` to its `experiments/results` directory. Point `GC_EXP19`
to the separately downloaded, cleaned `exp19_lambda_sweep.json`. For example,
from the repository root with the two downloads in `downloads/`:

```bash
mkdir archived-results
tar -xzf downloads/camera-ready-results.tar.gz -C archived-results
export GC_RESULTS="$PWD/archived-results/experiments/results"
export GC_EXP19="$PWD/downloads/exp19_lambda_sweep.json"
python finalize_camera_ready.py
python summarize_camera_ready.py
python gen_exp19_panels.py
```

`GC_RESULTS` selects the archived measurements for tables and figures. `GC_EXP19`
selects the original greedy selections, numerical diagnostics and stored ROUGE
scores; it redraws Figure 2(b,c) exactly from those measurements. Generated
summaries and dataset text are not distributed. The archive retains original
source hashes for provenance; those hashes describe the pre-release source,
not this portable release checkout. Configuration endpoints and credentials,
local paths and archive owner names have been removed or replaced.

## Environment and installation

Tested on CPython **3.13.5** on Linux. Install from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
# To use the exact dependency versions tested for this release instead:
# python -m pip install -r requirements-tested.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

NumPy 2.0 or newer is required for `bitwise_count` in the exact MaxCut reference.
Run all commands below from the repository root. The defaults are `results/`,
`results/figures/`, `cache/multinews_matrices/`, and `data/multinews_validation.json`.
Environment variables `GC_RESULTS`, `GC_CACHE`, `MULTINEWS_DATA`, and `GC_EXP19`
override the results directory, matrix cache, validation JSON, and exp19 reference
JSON respectively. Overrides propagate to worker processes. No private endpoint
or language-model service is called by the core experiment runners. The optional
summary evaluator below calls only endpoints you explicitly configure.

## Reproduction

The paper's runs used **at most 12 worker processes**. Keep BLAS thread counts at one
when using process workers. First reproduce the legacy comparison (four-group
features and the explicitly labelled shrinking-pool heuristic), freeze D-optimality
costs, and generate the current cells:

```bash
python run_all.py --legacy --exp coverage --seeds 10
python run_all.py --legacy --exp features --seeds 10
python check_dopt_validity.py
python run_camera_ready.py --stage tier1 --workers 12 --only-missing
python run_camera_ready.py --stage tier2 --workers 12 --only-missing
python run_camera_ready.py --stage features --workers 12 --only-missing
python run_camera_ready.py --stage maxcut --workers 12 --only-missing
python run_camera_ready.py --stage scale --n 500 --workers 12 --only-missing
python run_camera_ready.py --stage scale --n 1000 --workers 12 --only-missing
python run_camera_ready.py --stage scale --n 2000 --workers 12 --only-missing
python finalize_camera_ready.py
python gen_fig1.py
python gen_reference_comparisons.py
```

`check_dopt_validity.py` refuses to overwrite its frozen validation. Do not retune the
cost grid after examining outcomes. Each camera-ready cell records parameters,
seeds, source hashes, oracle counts and wall time. `tier1` retains the original
four-group features for comparisons; `features` uses the three-group, correlation-0.9
redesign used in Table 1. The legacy commands are needed before Tier 1 to populate
Table 9. Without their JSON files, current experiments still run but legacy columns
are omitted. `run_tier2.py` and `run_maxcut.py` retain the older runners; use
`run_camera_ready.py` for the paper's six-baseline sweeps.

The following numbering is from the compiled camera-ready paper. Commands in the
middle column follow the generation commands above. All paths are under the default
`results/` directory. Table snippets are text files containing LaTeX tabular output;
these scripts never require or edit the manuscript. Display formatting can differ
from the paper; the commands regenerate the underlying numerical entries and curves.

| Paper item | Exact command | Output file (table label where shared) |
|---|---|---|
| Figure 1, curvature spectrum | `python gen_fig1.py` | `results/figures/fig1_curvature_spectrum.pdf` |
| Figure 2(a), uniform-query reference | `python gen_reference_comparisons.py` | `results/figures/PMovieRec.pdf` |
| Figure 2(b,c), Multi-News sweep | `python gen_exp19_panels.py` | `results/figures/fig_exp19_panel_a.pdf`, `fig_exp19_panel_b.pdf` (requires exp19 reference) |
| Figure 3, Tier 1 cost sweeps | `python finalize_camera_ready.py` | `results/figures/fig_t1_design_sweep.pdf`, `fig_t1_coverage_sweep.pdf` |
| Figure 4, curvature versus ratios | `python finalize_camera_ready.py` | `results/figures/fig_t1_design_tightness.pdf`, `fig_t1_coverage_tightness.pdf` |
| Figure 5, Tier 2 baselines | `python summarize_camera_ready.py` | `results/figures/fig_t2_design.pdf`, `fig_t2_coverage.pdf` |
| Table 1, Tier 1 selected costs | `python finalize_camera_ready.py` | `results/camera_ready/primary_tables.txt` (`tab:tier1`); `paper_summary.json` holds unrounded values |
| Table 2, D-optimal validation | `python check_dopt_validity.py` then `python finalize_camera_ready.py` | `results/camera_ready/dopt_validation.json`; `primary_tables.txt` (`tab:dopt-validation`) |
| Table 3, Tier 2 selected costs | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:tier2`) |
| Table 4, illustrative HFWK comparison | `python gen_reference_comparisons.py` | `results/comparison.csv` |
| Table 5, MaxCut | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:maxcut`) |
| Table 6, full Tier 1 baseline grid | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-tier1`) |
| Table 7, full Tier 2 baseline grid | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-tier2`) |
| Table 8, redesigned features | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-features`) |
| Table 9, submitted legacy ratios | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-legacy`); requires legacy commands before Tier 1 |
| Table 10, oracle counts, first block | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-queries-0`) |
| Table 11, diagnostics, first block | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-diagnostics-0`) |
| Table 12, oracle counts, second block | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-queries-18`) |
| Table 13, diagnostics, second block | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-diagnostics-18`) |
| Table 14, scale grid | `python summarize_camera_ready.py` | `results/camera_ready/tables.txt` (`tab:cr-scale`) |
| Table 15, Multi-News lambda=0.1 | `python finalize_camera_ready.py --multinews` | `results/camera_ready/multinews_tables.txt` (`tab:multinews-0-1`) |
| Table 16, Multi-News lambda=0.25 | `python finalize_camera_ready.py --multinews` | `results/camera_ready/multinews_tables.txt` (`tab:multinews-0-25`) |
| Table 17, Multi-News lambda=0.5 | `python finalize_camera_ready.py --multinews` | `results/camera_ready/multinews_tables.txt` (`tab:multinews-0-5`) |
| Table 18, Multi-News lambda=0.75 | `python finalize_camera_ready.py --multinews` | `results/camera_ready/multinews_tables.txt` (`tab:multinews-0-75`) |
| Table 19, Multi-News lambda=1.0 | `python finalize_camera_ready.py --multinews` | `results/camera_ready/multinews_tables.txt` (`tab:multinews-1-0`) |
| Table 20, Multi-News lambda=1.5 | `python finalize_camera_ready.py --multinews` | `results/camera_ready/multinews_tables.txt` (`tab:multinews-1-5`) |

Table 21 summarizes theoretical guarantees; it is not an experimental table.
The extra exp19 panel `fig_exp19_panel_c.pdf` shows original ROUGE-2 values and is not
included in Figure 2 of this camera-ready version. Figure 2(a)'s generator is a new
rendering of the same analytical curves, rather than the original plotting source.
Its partial-monotonicity curves use Mualem and Feldman (NeurIPS 2022), Theorems 4.1,
4.2 and 6.1, [Using Partial Monotonicity in Submodular Maximization](https://arxiv.org/abs/2202.03051).

### Approximate runtimes

These are historical measurements from the paper's per-cell logs, not performance
guarantees. Median cell wall times were approximately 3.1 seconds (Tier 1), 3.1
seconds (Tier 2), 3.3 seconds (redesigned features), 1.6 seconds (MaxCut), and 7.9
seconds (scale). The slowest scale cell took about 283 seconds. Summed cell wall
times were about 9, 8, 4, 7 and 100 minutes respectively; process parallelism reduces
elapsed time, while scheduling and matrix generation add overhead. Multi-News costs
vary with the number of passages; inspect recorded `wall_time` rather than assuming
synthetic-instance timings apply. Plotting and aggregation take seconds on the tested
host. Full sweeps use 760 synthetic cells and 600 Multi-News cells.

## Multi-News data and original reference

Obtain the **untruncated validation split**, following the terms in the
[official Multi-News release](https://github.com/Alex-Fabbri/Multi-News).
The original experiment loader used the authors' public Hugging Face files
[`data/val.src.cleaned` and `data/val.tgt`](https://huggingface.co/datasets/alexfabbri/multi_news/tree/main/data).
Download exactly those two files into `data/`; for example:

```bash
mkdir -p data
curl -L --fail https://huggingface.co/datasets/alexfabbri/multi_news/resolve/main/data/val.src.cleaned -o data/val.src.cleaned
curl -L --fail https://huggingface.co/datasets/alexfabbri/multi_news/resolve/main/data/val.tgt -o data/val.tgt
```

Each source line contains articles separated by `|||||`. Preserve literal
`NEWLINE_CHAR` tokens, as the original loader did; replacing them with newlines
changes tokenization. Do not substitute a truncated or retokenized release.

```bash
python -I prepare_multinews.py --documents data/val.src.cleaned --summaries data/val.tgt \
  --output data/multinews_validation.json
```

The converter writes a list of `{"articles": ["..."], "summary": "...", "idx": 0}`
records in original validation order. There should be 5,622 records. The experiment
loader selects 300 with seed 42, then the runner uses the first 100, matching the
original experiment. Dataset text stays in ignored `data/` and is never committed.
The full validation JSON is about 75 MB; it was not downloaded as part of release
smoke testing; an existing local validation copy was used for an isolated
six-cell reproduction check, one complete baseline cell, and a converter round-trip. For alternate locations, set `MULTINEWS_DATA` before running.

Download the cleaned `exp19_lambda_sweep.json` asset from the v1.0 release and
select it with `GC_EXP19`, or place it in `results/`. It contains selections,
numerical results and original ROUGE scores, without article/passages or
generated summaries. Figure 2(b,c), the exact-original reproduction gate and
the baseline runner use this reference JSON. The dataset is still needed for
rerunning selection and the baseline sweep; plotting stored results does not
require downloading the dataset.

With both inputs available:

```bash
python -I run_multinews.py reproduce --workers 12
python -I run_multinews.py run --workers 12
python finalize_camera_ready.py --multinews
python gen_exp19_panels.py
```

The reproduction gate checks all 600 original selected lists and numerical values
before baseline runs. The baseline sweep makes no LLM calls and computes no new
ROUGE. Recovering the original generated summaries would require a separate
artifact; the JSON above contains scores rather than summary text.

## LLM summaries and ROUGE-2

After preparing the Multi-News validation data, install the optional evaluator
requirements and run the original 100-example, six-lambda sweep:

```bash
python -m pip install -r requirements-llm.txt
GC_RESULTS=results/llm-rerun python -I run_llm_rouge.py --endpoint "$OPENAI_BASE_URL"
```

`--endpoint` is required unless `OPENAI_BASE_URL` is set; there is no default
host. Repeat `--endpoint` for multiple endpoints. `--model` defaults to
`openai/gpt-oss-120b`. `OPENAI_API_KEY` supplies the key, with the harmless
`not-needed` default for local services that do not require one.

The prompt is `Summarize the following passages.` followed by a blank line
and numbered `[Passage 1] ...` blocks separated by blank lines. The original
settings are temperature **0.3** and **1024** maximum tokens. Calls rotate
through endpoints, starting at index 1 when multiple endpoints are supplied;
responses shorter than ten words fail over to the next endpoint. Empty
content falls back to the model's reasoning-content field, as in the original
experiment. Greedy-selected passages and the shared top-k TF-IDF baseline are
scored against the unmodified reference summary. ROUGE-1, ROUGE-2 and ROUGE-L
use `rouge-score`'s default tokenizer, Porter stemming and F1 scores.

The output is `results/llm-rerun/exp19_lambda_sweep.json` with the same result
and ROUGE schema as the stored exp19 asset. Endpoint locations and the key
are excluded from the saved configuration. Summaries are sampled at
temperature 0.3, so a rerun gives close but not identical ROUGE-2 scores;
the stored scores in exp19 reproduce the figure exactly. No fresh LLM calls
are needed to redraw stored scores or to run the camera-ready baseline sweep.
Original generated summaries cannot be recovered from stored scores alone.

## Verification and small runs

```bash
python -W error::SyntaxWarning -m compileall -q .
python smoke_test_new_algorithms.py
GC_RESULTS=results/smoke/tier1 python run_all.py --smoke --seeds 1
GC_RESULTS=results/smoke/tier2 python run_tier2.py --smoke --seeds 1
GC_RESULTS=results/smoke/maxcut python run_maxcut.py --smoke --seeds 1
GC_RESULTS=results/smoke/camera python run_camera_ready.py --stage tier1 --smoke --workers 1
python -I run_multinews.py --help
# After complete production reruns, including Multi-News:
python verify_camera_ready.py --check-values
```

Smoke mode changes sizes, seeds, and sampling settings; its outputs are not paper
results. The source-hash verifier checks reruns against this release checkout;
archived pre-release hashes refer to the private experiment history and will differ.
`validate_camera_ready.py`, `verify_extensions.py`, `check_design_validity.py`, and
`recompute_clamp.py` provide supporting checks. The A-optimality checker is retained
as a counterexample diagnostic, not as the design used in current experiments.

## Citation

```bibtex
@inproceedings{chen2026curvature,
  title = {Curvature Beyond Positivity: Greedy Guarantees for Arbitrary Submodular Functions},
  author = {Chen, Yixin and Kuhnle, Alan},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```
