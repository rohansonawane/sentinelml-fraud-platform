import { fetchJson } from "@/lib/api";

type Drift = {
  drift_detected: boolean;
  alerted_features: string[];
  n_reference: number;
  n_current: number;
  backend: string;
  induced?: Record<string, unknown>;
};

export default async function Page() {
  const report = await fetchJson<Drift>("/backend/api/v1/artifacts/drift_report.json");
  if (!report) return <p className="empty">No artifacts/drift_report.json. Run make drift-demo.</p>;
  return (
    <div>
      <h1>Drift</h1>
      <div className="grid">
        <div className="card">
          <div className="k">Detected</div>
          <div className="v">{report.drift_detected ? "yes" : "no"}</div>
        </div>
        <div className="card">
          <div className="k">Backend</div>
          <div className="v" style={{ fontSize: "1.1rem" }}>
            {report.backend}
          </div>
        </div>
        <div className="card">
          <div className="k">Ref / current n</div>
          <div className="v" style={{ fontSize: "1.1rem" }}>
            {report.n_reference} / {report.n_current}
          </div>
        </div>
      </div>
      <h2>Alerted features</h2>
      <ul>
        {report.alerted_features.map((f) => (
          <li key={f}>{f}</li>
        ))}
      </ul>
    </div>
  );
}
