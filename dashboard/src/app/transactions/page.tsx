import { fetchJson, type TxRow } from "@/lib/api";

export default async function Page() {
  const data = await fetchJson<{ items: TxRow[] }>("/backend/api/v1/transactions");
  const items = data?.items ?? [];
  if (!data) return <p className="empty">API unreachable. No placeholder rows are shown.</p>;
  if (!items.length) return <p className="empty">No scored transactions in the audit log yet. POST /api/v1/score or run make stream-demo.</p>;
  return (
    <div>
      <h1>Transaction explorer</h1>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Time</th>
            <th>Customer</th>
            <th>Merchant</th>
            <th>Amount</th>
            <th>P(fraud)</th>
            <th>Decision</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          {items.map((row) => (
            <tr key={row.transaction_id + String(row.event_time)}>
              <td>{row.transaction_id}</td>
              <td>{row.event_time}</td>
              <td>{row.customer_id}</td>
              <td>{row.merchant_id}</td>
              <td>{row.amount.toFixed(2)}</td>
              <td>{row.fraud_probability.toFixed(3)}</td>
              <td>
                <span className={`badge ${row.decision}`}>{row.decision}</span>
              </td>
              <td>{row.latency_ms.toFixed(2)} ms</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
