import { Num } from "@/components/ui";
import { fetchJson, type MetricsFile } from "@/lib/api";

export default async function Page() {
  const metrics = await fetchJson<MetricsFile>("/backend/api/v1/artifacts/metrics.json");
  if (!metrics?.champion) {
    return <p className="empty">artifacts/metrics.json not available via the API. Run make smoke.</p>;
  }
  const c = metrics.champion;
  const cm = c.test.confusion_matrix;
  return (
    <div>
      <h1>Model performance</h1>
      <p className="note">Values from artifacts/metrics.json (test split). Accuracy is not the objective.</p>
      <div className="grid">
        <div className="card">
          <div className="k">PR-AUC</div>
          <div className="v">
            <Num value={c.pr_auc} />
          </div>
        </div>
        <div className="card">
          <div className="k">ROC-AUC</div>
          <div className="v">
            <Num value={c.test.roc_auc} />
          </div>
        </div>
        <div className="card">
          <div className="k">Precision / Recall / F1</div>
          <div className="v" style={{ fontSize: "1.1rem" }}>
            <Num value={c.test.precision} /> / <Num value={c.recall} /> / <Num value={c.test.f1} />
          </div>
        </div>
        <div className="card">
          <div className="k">FPR</div>
          <div className="v">
            <Num value={c.fpr} />
          </div>
        </div>
        <div className="card">
          <div className="k">Expected cost (assumptions)</div>
          <div className="v">
            <Num value={c.expected_cost} digits={1} />
          </div>
        </div>
      </div>
      <h2>Confusion matrix</h2>
      <table>
        <thead>
          <tr>
            <th />
            <th>Pred 0</th>
            <th>Pred 1</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>True 0</td>
            <td>{cm.tn}</td>
            <td>{cm.fp}</td>
          </tr>
          <tr>
            <td>True 1</td>
            <td>{cm.fn}</td>
            <td>{cm.tp}</td>
          </tr>
        </tbody>
      </table>
      {c.test.pr_curve ? (
        <p className="note">
          PR curve points: {c.test.pr_curve.precision.length}. ROC points: {c.test.roc_curve?.fpr.length ?? 0}.
        </p>
      ) : null}
    </div>
  );
}
