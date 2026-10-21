# BWB Forecast scoreboard

A mobile-first Next.js App Router frontend for the public forecast scoreboard.

## Local development

Use a supported Node.js release and npm. From this directory:

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` to the API origin (for example, the local API origin). If it is omitted, requests use same-origin paths.
Set `NEXT_PUBLIC_SITE_URL` to the deployed site origin when publishing so Open Graph and Twitter image URLs resolve correctly; it defaults to `http://localhost:3000` for local development.

## Pages and API

- `/` — upcoming predictions from `GET /predictions/upcoming`
- `/track-record` — scorecard and recent matches from `GET /scorecard` and `GET /matches/recent`
- `/calibration` — reliability data from `GET /scorecard/calibration`
- `/method` — explanation of the evaluation principles

The data pages show explicit loading, API error, and empty states. Numeric values in the UI are read from API responses; no sample performance data is bundled.

The client accepts an array response or an object containing a resource array (commonly `items`, `data`, or an endpoint-specific collection). Score rows are expected to expose fields such as `league`, `season`, `forecaster`, `metric`, `value`, and `sample_count`. Upcoming match rows may expose nested `match` and `prediction` objects. Calibration bins may include an outcome, predicted mean, observed frequency, and count.

## Checks

```powershell
npm run lint
npm run build
```
