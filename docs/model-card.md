# Model card: BWB football outcome forecasting

## Status

No model is currently approved for production. The backtest is intentionally blocked when historical label availability cannot be established before the relevant forecast cutoff. Do not interpret this document as evidence of a measured advantage over Elo or the closing market.

## Intended use

- Estimate home/draw/away probabilities for Premier League and La Liga fixtures.
- Support transparent, pre-match research and public scorekeeping.
- Use the canonical `t-minus-24h-v1` policy only after an eligible model is explicitly promoted.

The forecasts are not guarantees and should not be represented as certainty or personalized betting advice.

## Model and baselines

- Candidate models: LightGBM with chronological temperature calibration and regularized multinomial logistic regression.
- Baseline: availability-gated Elo with home advantage and learned H/D/A probabilities.
- Evaluation: expanding season folds; hyperparameter selection and calibration remain inside the training interval.
- Metrics: log loss, multiclass Brier score, ranked probability score, class-wise ECE, reliability-bin counts, paired matchweek bootstrap intervals, and feature-group ablations.

## Data and provenance

Supported sources include football-data.co.uk historical match/closing odds, football-data.org live fixtures/results, and Understat xG. Values are used only when their system-observed availability timestamp precedes the fixture forecast cutoff. Retrospective closing odds without verified timestamps are benchmark-only and are labelled unverified.

Historical CSV records imported after their match dates cannot establish that labels or odds were available at the historical forecast cutoff. Such records are not silently backdated.

## Known limitations

- No leakage-safe historical advantage has been measured or approved yet.
- Production database, credentials, public ledger repository, independent timestamp proof, and hosting are not configured in this workspace.
- Team aliases require explicit mapping when source names differ.
- xG retrieval is first-observed; historic xG becomes usable only for forecasts made after ingestion.
- Backtest output is limited by eligible seasons, source timestamp coverage, and closing-market coverage.

## Promotion criteria

Promotion requires sufficient paired test fixtures, leakage-safe availability, documented calibration and ablation results, paired confidence intervals against baselines, model artifact provenance, and explicit review of coverage and high-confidence errors. No production promotion is performed automatically by the backtest command.
