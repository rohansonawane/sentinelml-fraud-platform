import Link from "next/link";
import type { ReactNode } from "react";

const LINKS = [
  ["/", "Overview"],
  ["/transactions", "Transactions"],
  ["/alerts", "Alerts"],
  ["/performance", "Performance"],
  ["/drift", "Drift"],
  ["/models", "Models"],
] as const;

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div>
      <header className="top">
        <div>
          <strong>SentinelML</strong>
          <span className="sub">Real-time fraud detection</span>
        </div>
        <nav>
          {LINKS.map(([href, label]) => (
            <Link key={href} href={href}>
              {label}
            </Link>
          ))}
        </nav>
      </header>
      <main className="wrap">{children}</main>
    </div>
  );
}

export function Empty({ text }: { text: string }) {
  return <p className="empty">{text}</p>;
}

export function Num({ value, digits = 3 }: { value: number | null | undefined; digits?: number }) {
  if (value === null || value === undefined || Number.isNaN(value)) return <span>n/a</span>;
  return <span>{value.toFixed(digits)}</span>;
}
