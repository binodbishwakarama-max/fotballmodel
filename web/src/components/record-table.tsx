import {
  formatDate,
  formatMetricName,
  formatNumber,
  isRecord,
  readNumber,
  readText,
  readValue,
  type ApiRecord,
} from "@/lib/api";

const columns = [
  { key: "league", label: "League" },
  { key: "season", label: "Season" },
  { key: "forecaster", label: "Forecaster" },
  { key: "metric", label: "Metric" },
  { key: "value", label: "Value" },
  { key: "sample_count", label: "Matches" },
];

function cellValue(row: ApiRecord, key: string): string {
  const raw = readValue(row, key);
  if (typeof raw === "number") return formatNumber(raw);
  if (typeof raw === "string") {
    if (key === "metric") return formatMetricName(raw);
    return raw;
  }
  if (key === "forecaster") {
    const name = readText(row, "model", "name", "forecaster_name");
    if (name) return name;
  }
  if (key === "sample_count") {
    const value = readNumber(row, "n", "count", "matches");
    if (value !== undefined) return formatNumber(value, 0);
  }
  if (key === "value") {
    const value = readNumber(row, "score", "metric_value");
    if (value !== undefined) return formatNumber(value);
  }
  return "—";
}

export function RecordTable({ rows }: { rows: ApiRecord[] }) {
  const tableRows = rows.map((row) => {
    if (isRecord(row.match) && isRecord(row.prediction)) {
      return { ...row, ...row.match, ...row.prediction };
    }
    return row;
  });
  return (
    <div className="overflow-x-auto rounded-2xl border border-[var(--line)]">
      <table className="w-full min-w-[680px] border-collapse text-left text-sm">
        <thead className="bg-[var(--surface-raised)] text-xs uppercase tracking-wider text-[var(--muted)]">
          <tr>
            {columns.map(({ key, label }) => (
              <th className="px-4 py-3 font-medium" key={key} scope="col">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--line)] bg-[var(--surface)]">
          {tableRows.map((row, index) => (
            <tr key={readText(row, "id", "match_id") ?? `${readText(row, "metric") ?? "record"}-${index}`} className="hover:bg-[var(--surface-raised)]">
              <td className="px-4 py-4 text-[var(--muted)]">{cellValue(row, "league")}</td>
              <td className="px-4 py-4 text-[var(--muted)]">{cellValue(row, "season")}</td>
              <td className="px-4 py-4 font-medium">{cellValue(row, "forecaster")}</td>
              <td className="px-4 py-4">{cellValue(row, "metric")}</td>
              <td className="px-4 py-4 font-mono text-[var(--accent)]">{cellValue(row, "value")}</td>
              <td className="px-4 py-4 text-[var(--muted)]">{cellValue(row, "sample_count")}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {tableRows.some((row) => readText(row, "kickoff_utc", "kickoff")) && (
        <p className="border-t border-[var(--line)] px-4 py-3 text-xs text-[var(--muted)]">
          Fixture times are UTC. {formatDate(readText(tableRows[0], "kickoff_utc", "kickoff"))}
        </p>
      )}
    </div>
  );
}
