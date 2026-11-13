# Agent Instructions

Agent conduct and engineering rules for `gdfnet-tvdenoise`.

## 1. Code integrity

1. **Root-cause focus.** Make the smallest coherent change that fixes the root
   cause within the requested scope.
2. **Single source of truth.** Declare constants, paths, bounds, tolerances, and
   workloads in `gdfnet_tvdenoise/config.py`.
3. **Strict functional invariance.** Preserve numerical constants, loss weights,
   formulas, feature order, solver steps, and observable behavior unless the task
   explicitly changes them. Move logic verbatim when refactoring.
4. **Fail fast.** Validate shapes, values, and files at meaningful boundaries. Do
   not mask errors with fallbacks.
5. **Standard primitives.** Prefer PyTorch and standard-library constructs over
   bespoke implementations.
6. **Replace, do not alias.** Delete the replaced name and update every caller.
7. **Runtime verification.** Verify correctness through the runtime workflows and
   artifact comparisons.
8. **Style.**
   - Tabs for Python indentation.
   - Maximum line length 100.
   - Ruff for linting and formatting.
   - Strict type annotations on functions and classes.
   - `from __future__ import annotations` in Python modules.
   - Use an environment with `requirements.txt` installed.
9. **Torch/float32 data path.** Use `torch.Tensor` for array data and array math
   (R1). Floats are float32; integers keep the dataset dtype (int32); int64 only
   where torch requires it for indices (R2). A long iterative recurrence may run
   in float64 internally if float32 fails a stated accuracy check; its output is
   float32 (R2a). Scalars are Python floats (summary metrics, selected λ, σ, scalar
   quadrature math); use `math` or `statistics` for scalar math (R3). Stored values
   equal used values: a generated value that a solver uses is float32 before use
   (R4). Use numpy only if necessary and torch has no good alternative; allowed
   uses are `.npz` read/write, PIL Lanczos resize (LiTS), and scipy `quad`, `ndtr`,
   `ndtri` (R5). Public entry points may accept numpy input and cast it to torch
   immediately: CLI `--image .npy` and `predict_gdf` (R5a). No round trips: switch
   library at most once per data path; a segment that starts and ends at a
   numpy-only library, with no torch consumer in between, stays numpy (LiTS NIfTI
   → PIL → `.npz`) (R6).

## 2. Scientific rigor and communication

1. **Direct communication.** Answer in the first sentence. Use plain language.
   Avoid filler and unsolicited summaries.
2. **Plain math in chat.** Format math with Unicode and plain text. Use fenced
   LaTeX only when explicitly asked.
3. **Exact reporting.** Report exact empirical observations without added margins.
4. **Objective evaluation.** State numerical criteria before running a check.
   Report null results, regressions, and failures directly.
5. **Anti-sycophancy.** Stress-test proposals against invariants and data. When
   the user identifies an error, concede immediately and correct it.
6. **Simplified Technical English.** Use short declarative sentences, active
   voice, and simple tenses. Keep technical nomenclature.

## 3. Pre-flight and documentation sync

1. Read the relevant module and its documentation before changing behavior.
2. When behavior or artifact fields change, update `README.md`,
   `docs/REFERENCE.md`, or `docs/REPORT_SCHEMA.md` in the same turn.
3. Keep `__version__` in `gdfnet_tvdenoise/__init__.py` and `version` in
   `CITATION.cff` in sync.
4. Keep mathematical derivations and data-split rationale in the accompanying
   publication.

## 4. Execution safety

1. **Read-only by default.** Questions, explanations, and reviews authorize only
   read-only inspection. Edits and compute runs require explicit instructions.
2. **Decouple code from workloads.** Approval to edit code never authorizes a
   computational run. Every run needs its own explicit command.
3. **Protect the working tree.** Never run `git checkout`, `git restore`,
   `git reset`, or `git clean` to discard working-tree changes without explicit
   instruction. Use surgical edits.
4. **Use `scratch/`.** Put temporary files, logs, and scratch scripts there.
5. **Use file tools for edits.** Use shell commands for searches and execution,
   not for rewriting repository files.
6. **Treat a successful edit as sufficient proof.** Do not run Git commands or
   dry-run builds solely to re-verify an edit.

## 5. Project constraints

- Keep changes inside the existing model family and benchmark suites. Ask before
  adding a new model family, benchmark family, scoring system, or packaged
  distribution.
- Committed artifacts contain no absolute paths, internal identifiers, registry
  filenames, backfill markers, or decision-log references.

## 6. Verification

Run the function or CLI that uses the changed code. Inspect its output or
artifact.

- Inference or features: run `predict_gdf` and compare the output.
- Model or checkpoint: load all three models.
- Benchmark: run the affected suite and inspect the report.
- Data or calibration: reload the changed artifacts and check their schema.
- Documentation: check changed links and snippets.

Run `ruff check .` and `ruff format --check .` for Python changes.
