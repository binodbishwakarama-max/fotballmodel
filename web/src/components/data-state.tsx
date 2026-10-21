"use client";

import { useEffect, useState } from "react";

type ResourceState<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: T };

export function useResource<T>(load: () => Promise<T>, refreshKey = ""): ResourceState<T> {
  const [state, setState] = useState<ResourceState<T>>({ status: "loading" });

  useEffect(() => {
    let active = true;
    setState({ status: "loading" });
    load()
      .then((data) => {
        if (active) setState({ status: "success", data });
      })
      .catch((error: unknown) => {
        if (active) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : "An unexpected error occurred.",
          });
        }
      });
    return () => {
      active = false;
    };
  }, [load, refreshKey]);

  return state;
}

export function DataState({
  loading,
  error,
  empty,
  onRetry,
}: {
  loading: boolean;
  error?: string;
  empty?: boolean;
  onRetry?: () => void;
}) {
  if (loading) {
    return (
      <div role="status" className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-6 text-sm text-[var(--muted)]">
        Loading scoreboard data…
      </div>
    );
  }
  if (error) {
    return (
      <div role="alert" className="rounded-2xl border border-rose-900/70 bg-rose-950/30 p-6">
        <p className="text-sm font-medium text-rose-100">{error}</p>
        {onRetry && (
          <button
            className="mt-4 rounded-full border border-rose-800 px-4 py-2 text-sm text-rose-100 transition hover:bg-rose-900/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
            onClick={onRetry}
            type="button"
          >
            Try again
          </button>
        )}
      </div>
    );
  }
  if (empty) {
    return (
      <div className="rounded-2xl border border-dashed border-[var(--line)] bg-[var(--surface)] p-8 text-center">
        <p className="text-sm font-medium">No data published yet</p>
        <p className="mt-2 text-sm text-[var(--muted)]">
          This section will fill in when the API has records to show.
        </p>
      </div>
    );
  }
  return null;
}
