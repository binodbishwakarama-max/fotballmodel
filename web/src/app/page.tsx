"use client";

import { useCallback, useState } from "react";
import { DataState, useResource } from "@/components/data-state";
import { PageHeading } from "@/components/page-heading";
import { ScorecardSummary } from "@/components/scorecard-summary";
import {
  asRows,
  fetchScorecard,
  fetchUpcoming,
  formatDate,
  formatNumber,
  isRecord,
  readNumber,
  readText,
  type ApiRecord,
  type UpcomingPrediction,
} from "@/lib/api";

function predictionRows(rows: UpcomingPrediction[]): ApiRecord[] {
  return rows.map((row) => {
    const match = isRecord(row.match) ? row.match : {};
    const prediction = isRecord(row.prediction) ? row.prediction : {};
    return { ...row, ...match, ...prediction };
  });
}

function UpcomingList({ rows }: { rows: ApiRecord[] }) {
  return (
    <div className="space-y-3">
      {rows.map((row, index) => {
        const home = readText(row, "home_team", "home_team_name", "home");
        const away = readText(row, "away_team", "away_team_name", "away");
        const kickoff = readText(row, "kickoff_utc", "kickoff", "date");
        const probabilities = isRecord(row.probabilities) ? row.probabilities : {};
        const homeProbability = readNumber(row, "p_home") ?? readNumber(probabilities, "home", "p_home");
        const drawProbability = readNumber(row, "p_draw") ?? readNumber(probabilities, "draw", "p_draw");
        const awayProbability = readNumber(row, "p_away") ?? readNumber(probabilities, "away", "p_away");
        const hasProbabilities =
          homeProbability !== undefined && drawProbability !== undefined && awayProbability !== undefined;
        const key = readText(row, "match_id", "id") ?? `${home ?? "fixture"}-${index}`;

        return (
          <article key={key} className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-4 sm:p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <p className="font-medium">{home ?? "Home team"} <span className="text-[var(--muted)]">vs</span> {away ?? "Away team"}</p>
              <p className="text-xs text-[var(--muted)]">{formatDate(kickoff)}</p>
            </div>
            {hasProbabilities ? (
              <>
                <div
                  role="img"
                  aria-label={`Forecast probabilities — home ${formatNumber(homeProbability * 100, 1)}%, draw ${formatNumber(drawProbability * 100, 1)}%, away ${formatNumber(awayProbability * 100, 1)}%`}
                  className="flex h-2 overflow-hidden rounded-full bg-[var(--surface-raised)]"
                >
                  <span className="bg-[var(--accent)]" style={{ width: `${Math.max(0, Math.min(100, homeProbability * 100))}%` }} />
                  <span className="bg-amber-300" style={{ width: `${Math.max(0, Math.min(100, drawProbability * 100))}%` }} />
                  <span className="bg-sky-300" style={{ width: `${Math.max(0, Math.min(100, awayProbability * 100))}%` }} />
                </div>
                <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
                  {[
                    ["Home", homeProbability, "text-[var(--accent)]"],
                    ["Draw", drawProbability, "text-amber-200"],
                    ["Away", awayProbability, "text-sky-200"],
                  ].map(([label, probability, color]) => (
                    <div key={String(label)}>
                      <dt className="text-[var(--muted)]">{label}</dt>
                      <dd className={`mt-1 font-mono font-medium ${color}`}>{formatNumber(Number(probability) * 100, 1)}%</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : (
              <p className="text-sm text-[var(--muted)]">Probability details are not available for this fixture.</p>
            )}
          </article>
        );
      })}
    </div>
  );
}

export default function HomePage() {
  const [upcomingRetry, setUpcomingRetry] = useState(0);
  const [scorecardRetry, setScorecardRetry] = useState(0);
  const loadUpcoming = useCallback(() => fetchUpcoming(), []);
  const loadScorecard = useCallback(() => fetchScorecard(), []);
  const state = useResource(loadUpcoming, String(upcomingRetry));
  const scorecardState = useResource(loadScorecard, String(scorecardRetry));
  const rows =
    state.status === "success"
      ? asRows(state.data, ["items", "predictions", "data"])
      : ([] as UpcomingPrediction[]);
  const fixtures = predictionRows(rows);

  return (
    <>
      <PageHeading
        eyebrow="The next fixtures"
        title="Forecasts, before kickoff."
        description="Upcoming published home, draw, and away probabilities, with fixture context provided by the scoreboard API."
      />
      <ScorecardSummary
        response={scorecardState.status === "success" ? scorecardState.data : undefined}
        loading={scorecardState.status === "loading"}
        error={scorecardState.status === "error" ? scorecardState.message : undefined}
        onRetry={() => setScorecardRetry((value) => value + 1)}
      />
      <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div>
          <h2 className="text-lg font-semibold">Upcoming predictions</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">Only published records are shown.</p>
        </div>
        {state.status === "success" && fixtures.length > 0 && (
          <span className="text-xs text-[var(--muted)]">{fixtures.length} API records</span>
        )}
      </div>
      <DataState
        loading={state.status === "loading"}
        error={state.status === "error" ? state.message : undefined}
        empty={state.status === "success" && fixtures.length === 0}
        onRetry={() => setUpcomingRetry((value) => value + 1)}
      />
      {state.status === "success" && fixtures.length > 0 && <UpcomingList rows={fixtures} />}
      <div className="mt-8 rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-5 sm:p-6">
        <p className="text-sm font-semibold">A record, not a promise</p>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">
          Forecast probabilities are not guarantees. This public scoreboard keeps forecast publication separate from measured outcomes and reports what the API has actually recorded.
        </p>
      </div>
    </>
  );
}
