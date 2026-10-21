"use client";

import { useCallback, useState } from "react";
import { DataState, useResource } from "@/components/data-state";
import { PageHeading } from "@/components/page-heading";
import { asRows, fetchCalibration, formatNumber, readNumber, readText, type ApiRecord } from "@/lib/api";

const outcomeColors = {
  Home: "bg-emerald-300",
  Draw: "bg-amber-300",
  Away: "bg-sky-300",
};

function CalibrationBins({ rows }: { rows: ApiRecord[] }) {
  return (
    <div className="space-y-4">
      {rows.map((row, index) => {
        const outcome = readText(row, "outcome", "class", "label") ?? "Outcome";
        const predicted = readNumber(row, "mean_predicted", "predicted", "confidence");
        const observed = readNumber(row, "fraction_positive", "observed", "accuracy");
        const count = readNumber(row, "count", "n", "sample_count");
        const color =
          outcomeColors[outcome as keyof typeof outcomeColors] ?? "bg-[var(--accent)]";
        return (
          <div
            key={readText(row, "id") ?? `${outcome}-${index}`}
            className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4"
          >
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-medium">{outcome}</h3>
              {count !== undefined && <span className="text-xs text-[var(--muted)]">n = {formatNumber(count, 0)}</span>}
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between gap-4">
                <span className="text-[var(--muted)]">Mean predicted</span>
                <span>{predicted === undefined ? "—" : formatNumber(predicted)}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-[var(--muted)]">Observed frequency</span>
                <span>{observed === undefined ? "—" : formatNumber(observed)}</span>
              </div>
            </div>
            {predicted !== undefined && observed !== undefined && (
              <div className="relative mt-4 h-2 overflow-hidden rounded-full bg-[var(--surface-raised)]" aria-label={`${outcome} calibration comparison`}>
                <span className={`absolute top-0 h-2 w-2 rounded-full ${color}`} style={{ left: `${Math.max(0, Math.min(100, predicted * 100))}%` }} aria-hidden="true" />
                <span className="absolute top-0 h-2 w-2 rounded-full bg-white" style={{ left: `${Math.max(0, Math.min(100, observed * 100))}%` }} aria-hidden="true" />
              </div>
            )}
            {predicted !== undefined && observed !== undefined && (
              <div className="mt-2 flex gap-4 text-[10px] text-[var(--muted)]">
                <span><i className={`mr-1 inline-block size-2 rounded-full ${color}`} />Predicted</span>
                <span><i className="mr-1 inline-block size-2 rounded-full bg-white" />Observed</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function CalibrationPage() {
  const [retry, setRetry] = useState(0);
  const load = useCallback(() => fetchCalibration(), []);
  const state = useResource(load, String(retry));
  const rows =
    state.status === "success"
      ? asRows(state.data, ["bins", "outcomes", "items", "data"])
      : [];

  return (
    <>
      <PageHeading
        eyebrow="Probability quality"
        title="Calibration."
        description="See how predicted probabilities align with observed frequencies. Each plotted value and count comes directly from the calibration endpoint."
      />
      <section aria-labelledby="calibration-data" className="max-w-3xl">
        <h2 id="calibration-data" className="mb-4 text-lg font-semibold">Reliability data</h2>
        <DataState
          loading={state.status === "loading"}
          error={state.status === "error" ? state.message : undefined}
          empty={state.status === "success" && rows.length === 0}
          onRetry={() => setRetry((value) => value + 1)}
        />
        {state.status === "success" && rows.length > 0 && <CalibrationBins rows={rows} />}
        <p className="mt-5 text-xs leading-5 text-[var(--muted)]">
          Markers compare predicted and observed rates on the same probability scale. Calibration is descriptive, not a guarantee about an individual fixture.
        </p>
      </section>
    </>
  );
}
