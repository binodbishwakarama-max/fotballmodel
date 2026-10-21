export type ApiRecord = Record<string, unknown>;

export interface ScorecardResponse extends ApiRecord {
  scores?: ApiRecord[];
  items?: ApiRecord[];
  data?: ApiRecord[];
}

export interface UpcomingPrediction extends ApiRecord {
  match?: ApiRecord;
  prediction?: ApiRecord;
}

export interface CalibrationResponse extends ApiRecord {
  bins?: ApiRecord[];
  outcomes?: ApiRecord[];
  items?: ApiRecord[];
}

export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiError";
  }
}

const apiBase = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/+$/, "");

async function getJson<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBase}${path}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("Could not connect to the scoreboard API. Check your connection and try again.");
  }

  if (!response.ok) {
    throw new ApiError(`The scoreboard API returned an error (${response.status}).`);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("The scoreboard API returned an unreadable response.");
  }
}

export function fetchScorecard(): Promise<ScorecardResponse> {
  return getJson<ScorecardResponse>("/scorecard");
}

export function fetchUpcoming(): Promise<UpcomingPrediction[] | { items?: UpcomingPrediction[] }> {
  return getJson("/predictions/upcoming");
}

export function fetchRecentMatches(): Promise<ApiRecord[] | { items?: ApiRecord[] }> {
  return getJson("/matches/recent");
}

export function fetchCalibration(): Promise<CalibrationResponse> {
  return getJson("/scorecard/calibration");
}

export function asRows(value: unknown, keys: string[] = ["items", "data"]): ApiRecord[] {
  if (Array.isArray(value)) {
    return value.filter(isRecord);
  }
  if (!isRecord(value)) {
    return [];
  }
  for (const key of keys) {
    if (Array.isArray(value[key])) {
      return (value[key] as unknown[]).filter(isRecord);
    }
  }
  return [];
}

export function isRecord(value: unknown): value is ApiRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function readValue(record: ApiRecord, ...keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) {
      return record[key];
    }
  }
  return undefined;
}

export function readText(record: ApiRecord, ...keys: string[]): string | undefined {
  const value = readValue(record, ...keys);
  return typeof value === "string" ? value : undefined;
}

export function readNumber(record: ApiRecord, ...keys: string[]): number | undefined {
  const value = readValue(record, ...keys);
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function formatNumber(value: number, digits = 3): string {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  }).format(value);
}

export function formatDate(value: string | undefined): string {
  if (!value) return "Time not provided";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(date);
}

export function formatMetricName(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
