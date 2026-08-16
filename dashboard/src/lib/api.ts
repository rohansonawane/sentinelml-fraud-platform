export type Stats = {
  n: number;
  alerts: number;
  avg_latency_ms: number;
  avg_probability: number;
  allow?: number;
  review?: number;
  block?: number;
  model?: string;
  model_version?: string;
  threshold?: number;
  champion?: {
    model?: string;
    test_pr_auc?: number;
    test_recall?: number;
    test_fpr?: number;
    why?: string;
  } | null;
};

export type TxRow = {
  transaction_id: string;
  event_time: string | null;
  customer_id: string;
  merchant_id: string;
  amount: number;
  fraud_probability: number;
  decision: string;
  risk_level: string;
  latency_ms: number;
  model_version: string;
  features?: Record<string, number>;
};

export type MetricsFile = {
  champion?: {
    model: string;
    pr_auc: number;
    recall: number;
    fpr: number;
    expected_cost: number;
    test: {
      precision: number;
      f1: number;
      roc_auc: number;
      confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
      pr_curve?: { precision: number[]; recall: number[] };
      roc_curve?: { fpr: number[]; tpr: number[] };
    };
  };
  models?: Array<{ model: string; pr_auc: number; recall: number; fpr: number; expected_cost: number }>;
  registry?: { champion: string; challenger: string | null; why: string };
};

export async function fetchJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}
