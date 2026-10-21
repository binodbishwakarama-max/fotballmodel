# Data and evaluation contract

This document is normative for ingestion, training, prediction, scoring, and publication. A source that cannot meet a requirement is excluded from the affected feature or comparison and its coverage is reported; it is never silently relabelled or filled with a different kind of data.

## Time and availability

- Store match kickoff and source-event times as timezone-aware UTC instants. Parse source-local dates using the source's documented timezone and daylight-saving rules; reject ambiguous or nonexistent local times rather than guessing.
- Every value that can affect a forecast carries an `available_at` timestamp: when the value first became observable to this system, not when a historical download happened. A value is eligible for a fixture only when `available_at < forecast_cutoff < kickoff_utc`.
- Record retrieval time separately from source publication/observation time. Retrieval time is not evidence that a historical value was available at an earlier date.
- For sources without reliable historical publication times, store the first time this system observed the value as `available_at`. This makes the value usable only for forecasts made after ingestion and never retroactively available in a historical fold.
- Do not train or report a leakage-controlled backtest with features whose historical availability cannot be established. Retrospective datasets without availability history may be used only in explicitly labelled exploratory analysis, not the production-equivalent evaluation.
- Predictions require a UTC kickoff strictly in the future at the time of publication. Late fixtures are skipped and counted as incidents; they are not backfilled as live forecasts.

## Odds and market benchmark

- Store odds with their source, kind (`open` or `close`), observed/available timestamp, retrieval timestamp, and raw decimal prices.
- Opening odds are never a fallback for closing odds. Missing closing odds remain missing.
- De-vig probabilities from closing prices only for the market benchmark. Keep the de-vig method and input prices with the evaluation record.
- Historical closing prices without trustworthy observation timestamps may support a retrospective closing-line benchmark, but must be labelled as such; they cannot be used as live features or represented as a real-time executable forecast comparison.
- Report market coverage and compare model, Elo, and market only on the exact same fixtures with valid outcomes and valid closing prices. Also report model/Elo results on their broader available fixture set separately.

## Features and forecast policy

- Build training and inference features through the same per-fixture as-of interface. Each row has its own UTC cutoff; all source records are filtered by `available_at < cutoff` before aggregation.
- Feature computation must not read the target fixture's result, goals, xG, odds, or any other post-cutoff value.
- Historical xG is eligible only if its availability before the relevant cutoff can be established. Do not assume that a downloaded historical value was known at the time.
- Understat's embedded match datetime without an offset is interpreted as UTC. When this system first retrieves historical Understat xG, record that retrieval instant as both `retrieved_at` and `available_at`; this does not establish availability before retrieval.
- Publish one canonical forecast per fixture, model version, and forecast-policy version. The initial policy is `t-minus-24h-v1`: publish in the window from 30 hours to 24 hours before kickoff, using the first successful scheduled run in that window. The exact cutoff is the UTC publication time and is recorded with the forecast. If the window is missed, record a missed-publication incident; do not create a late substitute. Retries are idempotent and cannot alter a published payload.
- If a future product needs multiple forecast horizons, each horizon must have a distinct policy identifier and separate scoring; never silently overwrite the canonical forecast.

## Evaluation

- Use expanding walk-forward folds: every test season is predicted by a model fitted only on earlier seasons. Hyperparameter selection and calibration use chronological slices strictly inside the training period.
- Multiclass Brier score is the per-fixture sum of the three squared class-probability errors, averaged across fixtures. RPS uses the cumulative H/D/A probabilities in that order and averages the squared cumulative errors over the two cut points. Report these definitions alongside results.
- ECE is one-vs-rest per outcome with 10 fixed-width confidence bins on `[0, 1]`; report per-outcome values and the macro average. Include reliability-bin counts.
- Report log loss, Brier, RPS, ECE, sample counts, and coverage by league and season, plus pooled values. Accuracy is secondary.
- Confidence intervals for model-minus-baseline loss differences use paired resampling of matchweeks within league-season, preserving each fixture's paired outcomes. Document the bootstrap seed, replicate count, and interval method.
- Never assert that a market benchmark must outperform a model as a correctness test. A surprising result is an investigation trigger; the report must state the measured value and uncertainty.

## Ledger and verification

- Persist canonical JSON with sorted keys, fixed six-decimal probability formatting, UTC timestamps, and a documented schema version. Hash the exact canonical bytes with SHA-256.
- The application database is an operational copy, not the independent source of truth. Database triggers prevent accidental mutation but do not defend against a privileged database operator.
- Publish each ledger file to a dedicated public repository with protected append-only history. Verification recomputes the file hash and checks the published repository object/reference and expected batch metadata; it must not trust a database row as the only comparator.
- A Git commit timestamp alone is not an independently trusted timestamp. Anchor published hashes with an external timestamp/transparency service (for example, an OpenTimestamps proof) and report whether the proof is pending or verified.
- A failed hash, unavailable public reference, mismatched batch metadata, or invalid timestamp proof returns a non-zero verification result with a clear reason.

## Reproducibility and data quality

- Commit `uv.lock`; CI and documented reproduction commands use the locked dependency graph.
- Full-season match-count expectations apply only to completed seasons. The current season is partial by definition.
- Log counts for download failures, rejected rows, missing columns, unknown team names, unmatched fixtures, and unavailable features. Unknown team aliases fail ingestion until explicitly mapped.
- Keep credentials, raw downloads, caches, and generated model artifacts out of version control unless an explicit public-data publication policy says otherwise.
