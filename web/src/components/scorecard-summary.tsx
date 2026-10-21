import {
  asRows,
  formatDate,
  formatNumber,
  isRecord,
  readNumber,
  readText,
  type ApiRecord,
} from "@/lib/api";
import { DataState } from "@/components/data-state";

const METRICS = [
  { key: "log_loss", label: "Log loss" },
  { key: "brier", label: "Brier score" },
  { key: "rps", label: "RPS" },
] as const;

const FORECASTERS = [
  { key: "model", label: "BWB model" },
  { key: "market", label: "Market" },
  { key: "elo", label: "Elo" },
] as const;

type Score = {
  forecaster: string;
  metric: string;
  value: number;
  matches?: number;
  updatedAt?: string;
  scope: Record<string, string>;
};

function canonicalMetric(value: string): string | undefined {
  const normalized = value.toLowerCase().replace(/[\s-]+/g, "_");
  if (normalized.includes("log_loss") || normalized === "logloss") return "log_loss";
  if (normalized.includes("brier")) return "brier";
  if (normalized === "rps" || normalized.includes("ranked_probability")) return "rps";
  return undefined;
}

function canonicalForecaster(value: string): string | undefined {
  const normalized = value.toLowerCase();
  if (normalized.includes("market")) return "market";
  if (normalized.includes("elo")) return "elo";
  if (normalized.includes("model") || normalized.includes("bwb")) return "model";
  return undefined;
}

function scopeOf(record: ApiRecord): Record<string, string> {
  const scope: Record<string, string> = {};
  for (const key of ["league", "season", "slice", "cohort"]) {
    const value = readText(record, key);
    if (value !== undefined) scope[key] = value;
  }
  return scope;
}

function scoresFrom(response: ApiRecord): Score[] {
  const rows = asRows(response, ["scores", "items", "data", "results", "scorecard"]);
  const normalized: Score[] = [];

  for (const row of rows) {
    const metricName = readText(row, "metric", "metric_name", "name");
    const scoreValue = readNumber(row, "value", "score", "metric_value");
    const sourceName =
      readText(row, "forecaster", "model", "source", "baseline", "name") ?? "";
    const metric = metricName ? canonicalMetric(metricName) : undefined;
    const forecaster = canonicalForecaster(sourceName);
    if (metric && forecaster && scoreValue !== undefined) {
      normalized.push({
        forecaster,
        metric,
        value: scoreValue,
        matches: readNumber(row, "matches", "sample_count", "n", "count"),
        updatedAt: readText(row, "last_updated", "updated_at", "computed_at"),
        scope: scopeOf(row),
      });
      continue;
    }

    const groupedForecaster = canonicalForecaster(sourceName);
    if (!groupedForecaster) continue;
    const metrics = isRecord(row.metrics) ? row.metrics : row;
    for (const [key, value] of Object.entries(metrics)) {
      const groupedMetric = canonicalMetric(key);
      if (!groupedMetric) continue;
      const metricRecord = isRecord(value) ? value : undefined;
      const numericValue =
        typeof value === "number"
          ? value
          : metricRecord
            ? readNumber(metricRecord, "value", "score", "metric_value")
            : undefined;
      if (numericValue === undefined) continue;
      normalized.push({
        forecaster: groupedForecaster,
        metric: groupedMetric,
        value: numericValue,
        matches: metricRecord
          ? readNumber(metricRecord, "matches", "sample_count", "n", "count") ??
            readNumber(row, "matches", "sample_count", "n", "count")
          : readNumber(row, "matches", "sample_count", "n", "count"),
        updatedAt:
          (metricRecord && readText(metricRecord, "last_updated", "updated_at", "computed_at")) ??
          readText(row, "last_updated", "updated_at", "computed_at"),
        scope: scopeOf(row),
      });
    }
  }

  if (normalized.length === 0) {
    for (const [name, value] of Object.entries(response)) {
      const forecaster = canonicalForecaster(name);
      if (!forecaster || !isRecord(value)) continue;
      const metrics = isRecord(value.metrics) ? value.metrics : value;
      for (const [key, metricValue] of Object.entries(metrics)) {
        const metric = canonicalMetric(key);
        if (!metric) continue;
        const metricRecord = isRecord(metricValue) ? metricValue : undefined;
        const numericValue =
          typeof metricValue === "number"
            ? metricValue
            : metricRecord
              ? readNumber(metricRecord, "value", "score", "metric_value")
              : undefined;
        if (numericValue === undefined) continue;
        normalized.push({
          forecaster,
          metric,
          value: numericValue,
          matches: (metricRecord && readNumber(metricRecord, "matches", "sample_count", "n", "count")) ??
            readNumber(value, "matches", "sample_count", "n", "count"),
          updatedAt:
            (metricRecord && readText(metricRecord, "last_updated", "updated_at", "computed_at")) ??
            readText(value, "last_updated", "updated_at", "computed_at"),
          scope: scopeOf(value),
        });
      }
    }
  }

  return normalized;
}

function behindMarketNote(scores: Score[]): string | undefined {
  const modelScores = scores.filter((score) => score.forecaster === "model");
  const marketScores = scores.filter((score) => score.forecaster === "market");
  if (!modelScores.length || !marketScores.length) return undefined;

  const comparable = METRICS.flatMap(({ key }) => {
    const model = modelScores.find((score) => score.metric === key);
    const market = marketScores.find((score) => score.metric === key);
    if (
      !model ||
      !market ||
      model.matches === undefined ||
      market.matches === undefined ||
      model.matches !== market.matches
    ) {
      return [];
    }
    const scopeKeys = Object.keys(model.scope);
    if (
      scopeKeys.length === 0 ||
      scopeKeys.length !== Object.keys(market.scope).length ||
      !scopeKeys.every((scopeKey) => model.scope[scopeKey] === market.scope[scopeKey])
    ) {
      return [];
    }
    return [{ metric: key, model, market }];
  });
  if (!comparable.length || !comparable.every(({ model, market }) => model.value > market.value)) {
    return undefined;
  }

  const { model } = comparable[0];
  const scopeLabel = Object.values(model.scope).join(" · ");
  const metricLabels = comparable.map(
    ({ metric }) => METRICS.find((item) => item.key === metric)?.label ?? metric,
  );
  return `On ${scopeLabel} (${formatNumber(model.matches!, 0)} matches), the model scored higher than the market on ${metricLabels.join(", ")}. Lower is better for these loss metrics.`;
}

export function ScorecardSummary({
  response,
  loading,
  error,
  onRetry,
}: {
  response?: ApiRecord;
  loading: boolean;
  error?: string;
  onRetry: () => void;
}) {
  const scores = response ? scoresFrom(response) : [];
  const updatedAt =
    (response &&
      readText(response, "last_updated", "updated_at", "computed_at", "generated_at")) ??
    scores.map((score) => score.updatedAt).filter((value): value is string => Boolean(value)).sort().at(-1);
  const rowsAvailable = scores.length > 0;
  const marketNote = behindMarketNote(scores);

  return (
    <section aria-labelledby="scorecard-heading" className="mb-12">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">Measured performance</p>
          <h2 id="scorecard-heading" className="text-xl font-semibold sm:text-2xl">Model vs market vs Elo</h2>
          <p className="mt-2 text-sm text-[var(--muted)]">Historical loss scores and sample sizes from the scorecard API.</p>
        </div>
        {updatedAt && !loading && !error && (
          <p className="text-xs text-[var(--muted)]">Last updated: {formatDate(updatedAt)}</p>
        )}
      </div>

      <DataState
        loading={loading}
        error={error}
        empty={!loading && !error && !rowsAvailable}
        onRetry={onRetry}
      />

      {!loading && !error && rowsAvailable && (
        <>
          <div className="overflow-x-auto rounded-2xl border border-[var(--line)]">
            <table className="w-full min-w-[620px] border-collapse text-left text-sm">
              <thead className="bg-[var(--surface-raised)] text-xs uppercase tracking-wider text-[var(--muted)]">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">Forecaster</th>
                  {METRICS.map((metric) => (
                    <th scope="col" key={metric.key} className="px-4 py-3 font-medium">{metric.label}</th>
                  ))}
                  <th scope="col" className="px-4 py-3 font-medium">Matches</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--line)] bg-[var(--surface)]">
                {FORECASTERS.map((forecaster) => {
                  const forecasterScores = scores.filter((score) => score.forecaster === forecaster.key);
                  if (!forecasterScores.length) return null;
                  const matchCounts = [...new Set(forecasterScores.map((score) => score.matches).filter((n): n is number => n !== undefined))];
                  return (
                    <tr key={forecaster.key}>
                      <th scope="row" className="px-4 py-4 font-medium">{forecaster.label}</th>
                      {METRICS.map((metric) => {
                        const score = forecasterScores.find((item) => item.metric === metric.key);
                        return (
                          <td key={metric.key} className="px-4 py-4 font-mono text-[var(--accent)]">
                            {score ? formatNumber(score.value) : "—"}
                          </td>
                        );
                      })}
                      <td className="px-4 py-4 text-[var(--muted)]">
                        {matchCounts.length === 1 ? formatNumber(matchCounts[0], 0) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-[var(--muted)]">Lower is better for log loss, Brier score, and RPS. Dashes indicate a value not supplied by the API.</p>
          {marketNote && (
            <p className="mt-4 rounded-xl border border-amber-700/50 bg-amber-950/25 px-4 py-3 text-sm text-amber-100">
              {marketNote}
            </p>
          )}
        </>
      )}
    </section>
  );
}
