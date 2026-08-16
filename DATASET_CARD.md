# Dataset card — SentinelML simulator

## Motivation

Primary training data is a **Fraud Detection Handbook-style simulator**, not a bank extract. The optional OpenML credit-card file is documented in `scripts/download_data.py` and is **not required**.

## Composition

Each row is one transaction:

- `transaction_id`, `event_id`, `event_time` (UTC)
- `customer_id`, `merchant_id`, `amount`, `currency`, `channel`
- `is_fraud` (0/1), `fraud_scenario` (`legit`, `compromised_card`, `compromised_merchant`, `amount_anomaly`)
- `schema_version`

Legitimate spend is concentrated near a customer’s home location and daytime hours. Fraud is injected as compromised cards (unusual amount + distant merchant), compromised merchants, and amount anomalies.

Default smoke scale on this host: 6000 transactions (see `artifacts/split_manifest.json`: 3608/1241/1151 train/val/test), not millions.

## Splits

**Temporal only.** TRAIN is earlier, VAL later, TEST latest. Exact timestamp boundaries and row hashes are in `artifacts/split_manifest.json`. There is no random row split.

## Label delay

Merchant historical fraud rate uses labels only after `label_delay_days` (default 7). Rolling velocity uses only transactions with `event_time` **strictly before** the current event.

## Limitations

- Synthetic. Prevalence, amount distributions, and attack mix are **not** a real portfolio.
- Labels are available at generation time; production labels arrive late and noisy.
- Do not treat expected cost in metrics as bank loss.

## License / PII

Generated IDs are synthetic. No personal data. MIT for the generator code.
