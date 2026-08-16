import { Num } from "@/components/ui";
import { fetchJson, type MetricsFile, type Stats } from "@/lib/api";

export default async function Page() {
  const stats = await fetchJson<Stats>("/backend/api/v1/stats");
  const metrics = await fetchJson<MetricsFile>("/backend/api/v1/artifacts/metrics.json");
  if (!stats && !metrics) {
    return (
      <p className="empty">
        API is not reachable on port 8102. Start it with `make serve` after `make smoke`. Numbers are never invented.
      </p>
    );
  }
  const n = stats?.n ?? 0;
  const alerts = stats?.alerts ?? 0;
  const lat = stats?.avg_latency_ms;
  const model = stats?.model_version ?? metrics?.champion?.model ?? "unknown";
  return (
    <div>
      <h1>Overview</h1>
      <p className="note">Live audit stats from the API plus champion metrics from artifacts.</p>
      <div className="grid">
        <div className="card">
          <div className="k">Transactions scored</div>
          <div className="v">{n}</div>
        </div>
        <div className="card">
          <div className="k">Alerts (REVIEW/BLOCK)</div>
          <div className="v">{alerts}</div>
        </div>
        <div className="card">
          <div className="k">Avg latency (ms)</div>
          <div className="v">{lat !== undefined ? lat.toFixed(2) : "n/a"}</div>
        </div>
        <div className="card">
          <div className="k">Current model</div>
          <div className="v" style={{ fontSize: "1.05rem" }}>
            {model}
          </div>
        </div>
        <div className="card">
          <div className="k">ALLOW / REVIEW / BLOCK</div>
          <div className="v" style={{ fontSize: "1.1rem" }}>
            {stats?.allow ?? 0} / {stats?.review ?? 0} / {stats?.block ?? 0}
          </div>
        </div>
        <div className="card">
          <div className="k">Champion test PR-AUC</div>
          <div className="v">
            <Num value={metrics?.champion?.pr_auc} />
          </div>
        </div>
      </div>
    </div>
  );
}
