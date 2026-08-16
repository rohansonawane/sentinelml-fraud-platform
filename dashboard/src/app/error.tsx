"use client";

export default function ErrorView({ error }: { error: Error }) {
  return (
    <p className="empty">
      Dashboard error: {error.message}. The UI does not invent metrics; it only renders API/artifact payloads.
    </p>
  );
}
