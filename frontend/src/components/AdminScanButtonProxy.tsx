// src/components/AdminScanButtonProxy.tsx
import React, { useState } from "react";

export default function AdminScanButtonProxy({ endpoint = "/api/admin/models/scan" }: { endpoint?: string }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);

  async function runScan() {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(endpoint, { method: "POST", credentials: "include" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setResult({ error: String(err) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <button onClick={runScan} disabled={loading} className="rounded border px-2 py-1">
        {loading ? "Scanning…" : "Run model scan"}
      </button>
      {result && <pre className="mt-2 text-xs">{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
