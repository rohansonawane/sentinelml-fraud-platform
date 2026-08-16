import { Num } from "@/components/ui";
import { fetchJson, type MetricsFile } from "@/lib/api";

export default async function Page() {
  const metrics = await fetchJson<MetricsFile>("/backend/api/v1/artifacts/metrics.json");
  const registry = metrics?.registry;
  if (!metrics?.models) return <p className="empty">No model comparison artifact yet.</p>;
  return (
    <div>
      <h1>Models</h1>
      <p className="note">
        Champion: {registry?.champion ?? "n/a"} · Challenger: {registry?.challenger ?? "n/a"}
      </p>
      <p className="note">{registry?.why}</p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>PR-AUC</th>
            <th>Recall</th>
            <th>FPR</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {metrics.models.map((m) => (
            <tr key={m.model}>
              <td>{m.model}</td>
              <td>
                <Num value={m.pr_auc} />
              </td>
              <td>
                <Num value={m.recall} />
              </td>
              <td>
                <Num value={m.fpr} />
              </td>
              <td>
                <Num value={m.expected_cost} digits={1} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
