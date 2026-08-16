import type { ReactNode } from "react";
import "./globals.css";
import { Shell } from "@/components/ui";

export const metadata = {
  title: "SentinelML",
  description: "Real-time fraud detection monitoring",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
