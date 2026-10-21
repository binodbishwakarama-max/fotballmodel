# Mid-season forecasting report

> Replace every bracketed field from a generated, leakage-safe report. If a value is unavailable, say so; do not estimate or backfill it.

## Reporting period

- Generated at (UTC): [timestamp]
- League(s): [PL / PD]
- Seasons and test folds: [eligible seasons]
- Forecast policy: `t-minus-24h-v1`
- Data snapshot / commit: [identifier]

## Data coverage and provenance

- Settled fixtures: [count]
- Eligible out-of-sample forecasts: [count]
- Result-availability eligible training folds: [count]
- Verified closing-market coverage: [count / percentage]
- Retrospective unverified closing-market coverage: [count / percentage]
- xG coverage and first-observed cutoff: [count / timestamp]
- Exclusions and incidents: [list]

## Forecast quality

| League | Season | Forecaster | Matches | Log loss | Brier | RPS | ECE H | ECE D | ECE A | ECE macro |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [league] | [season] | [model / Elo / market] | [n] | [value] | [value] | [value] | [value] | [value] | [value] | [value] |

Include pooled metrics, reliability-bin counts, the calibration plot, and the exact eligible fixture set.

## Paired comparisons

For each baseline, report the model-minus-baseline log-loss difference on identical fixtures, its paired matchweek-clustered 95% interval, bootstrap seed, replicate count, and market availability status.

## Error analysis and ablations

Summarize outcome-class errors, high-confidence incorrect forecasts, feature-group ablations, and any observed drift. State explicitly when the sample is too small to interpret.

## Ledger and reproducibility

- Ledger batch IDs and SHA-256 hashes: [values]
- Public reference verification: [verified / pending / unavailable]
- Independent timestamp proof: [verified / pending / unavailable]
- Model versions and artifact hashes: [values]

## Conclusions

[State measured findings only. Do not claim model superiority unless paired evidence supports it.]
