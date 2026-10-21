"use client";

import { useCallback, useState } from "react";
import { DataState, useResource } from "@/components/data-state";
import { PageHeading } from "@/components/page-heading";
import { RecordTable } from "@/components/record-table";
import { asRows, fetchRecentMatches, fetchScorecard, type ApiRecord } from "@/lib/api";

export default function TrackRecordPage() {
  const [retry, setRetry] = useState(0);
  const load = useCallback(async () => {
    const [scorecard, recent] = await Promise.all([fetchScorecard(), fetchRecentMatches()]);
    return {
      scores: asRows(scorecard, ["scores", "items", "data"]),
      matches: asRows(recent, ["items", "matches", "data"]),
    };
  }, []);
  const state = useResource(load, String(retry));
  const rows: ApiRecord[] =
    state.status === "success" ? [...state.data.scores, ...state.data.matches] : [];

  return (
    <>
      <PageHeading
        eyebrow="Measured performance"
        title="The track record."
        description="Historical scores and recent completed fixtures, sourced from the public scoreboard API. Values are shown as published by the service."
      />
      <section aria-labelledby="record-title">
        <h2 id="record-title" className="mb-4 text-lg font-semibold">Scorecard and recent matches</h2>
        <DataState
          loading={state.status === "loading"}
          error={state.status === "error" ? state.message : undefined}
          empty={state.status === "success" && rows.length === 0}
          onRetry={() => setRetry((value) => value + 1)}
        />
        {state.status === "success" && rows.length > 0 && <RecordTable rows={rows} />}
      </section>
      <p className="mt-5 text-xs leading-5 text-[var(--muted)]">
        Comparisons should be read only across matching fixtures and reported coverage. Missing values are not treated as zero.
      </p>
    </>
  );
}
