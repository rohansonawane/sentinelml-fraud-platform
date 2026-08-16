import { fetchJson, type TxRow } from "@/lib/api";

export default async function Page() {
  const data = await fetchJson<{ items: TxRow[] }>("/backend/api/v1/alerts");
  const items = data?.items ?? [];
  if (!data) return <p className="empty">API unreachable.</p>;
  if (!items.length) return <p className="empty">No REVIEW/BLOCK alerts in the audit log.</p>;
  return (
    <div>
      <h1>Alert review</h1>
      <p className="note">
        Attribution is feature contribution / model attribution, not causal proof. Real blocking needs legal/business
        validation.
      </p>
      {items.map((row) => (
        <div className="card" key={row.transaction_id} style={{ marginBottom: "1rem" }}>
          <div>
            <strong>{row.transaction_id}</strong>{" "}
            <span className={`badge ${row.decision}`}>{row.decision}</span> p={row.fraud_probability.toFixed(3)} amount=
            {row.amount.toFixed(2)}
          </div>
          <div className="note">
            {row.customer_id} → {row.merchant_id} · {row.model_version}
          </div>
          {row.features ? (
            <table>
              <tbody>
                {Object.entries(row.features)
                  .slice(0, 8)
                  .map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{v.toFixed(4)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : null}
        </div>
      ))}
    </div>
  );
}
