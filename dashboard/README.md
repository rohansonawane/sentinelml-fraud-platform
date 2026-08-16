# Dashboard

Next.js App Router (TypeScript strict) on port **3102**. It proxies `/backend/*` to the FastAPI service on **8102**.

Pages: Overview, Transactions, Alerts, Performance, Drift, Models.

All numbers come from the API or `artifacts/*.json`. Empty states render if the API is down — no fake time series.
