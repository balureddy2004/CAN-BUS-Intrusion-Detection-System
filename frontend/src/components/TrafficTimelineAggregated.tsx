// src/components/TrafficTimelineAggregated.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  Line,
} from "recharts";

export interface TrafficTimelineAggregatedProps {
  apiPath?: string;            // e.g. "/api/alerts"
  windowSeconds?: number;      // e.g. 30
  updateIntervalMs?: number;   // e.g. 1000
  canId?: string;              // e.g. "0x2AA" (optional)
  showMetric?: "count" | "bytes";
  className?: string;
}

/** Derive base path from API (removes trailing /api/alerts etc.) */
function baseFromApiPath(apiPath: string) {
  return apiPath.replace(/\/api\/alerts.*/, "");
}

function formatLabel(msBucket: number) {
  return new Date(msBucket).toLocaleTimeString();
}

export default function TrafficTimelineAggregated({
  apiPath = "/api/alerts",            // ✅ same-origin, works with Vite proxy
  windowSeconds = 30,
  updateIntervalMs = 1000,
  canId,
  showMetric = "count",
  className = "",
}: TrafficTimelineAggregatedProps) {
  const [data, setData] = useState<
    Array<{ t: number; count: number; bytes: number; attacks: number; label: string }>
  >([]);
  const [paused, setPaused] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    async function fetchSeries() {
      if (paused) return;
      try {
        const base = baseFromApiPath(apiPath);
        const q = new URLSearchParams({ window: "1s", since: `${windowSeconds}s` });
        if (canId) q.set("can_id", canId);

        // ✅ same-origin so Vite forwards to :8000
        const res = await fetch(`${base}/api/packets/timeseries?${q.toString()}`);
        if (!res.ok) return;

        const json = (await res.json()) as Array<{
          t: number;
          count: number;
          bytes: number;
          attacks: number;
        }>;

        const withLabels = json.map((b) => ({
          ...b,
          label: formatLabel(b.t),
        }));
        setData(withLabels);
      } catch {
        // ignore transient errors (network, WS resets, etc.)
      }
    }

    fetchSeries();
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = window.setInterval(fetchSeries, updateIntervalMs) as unknown as number;

    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [apiPath, windowSeconds, updateIntervalMs, canId, paused]);

  const totals = useMemo(() => {
    let count = 0, bytes = 0, attacks = 0;
    for (const r of data) {
      count += r.count || 0;
      bytes += r.bytes || 0;
      attacks += r.attacks || 0;
    }
    return { count, bytes, attacks };
  }, [data]);

  const humanBytes = (n: number) => {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  };

  // simple palette
  const barColor = "#3b82f6";    // blue
  const bytesColor = "#10b981";  // green
  const attackLine = "#ef4444";  // red

  return (
    <div className={`p-4 bg-white/5 rounded-2xl shadow-sm ${className}`}>
      <div className="text-center mb-2">
        <h3 className="text-lg font-semibold">
          Traffic Overview {canId ? `— ${canId}` : ""}
        </h3>
        <div className="text-sm text-muted-foreground">Last {windowSeconds} seconds</div>
        <div className="mt-2 flex items-center justify-center gap-4 text-sm">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={paused}
              onChange={(e) => setPaused(e.target.checked)}
            />
            <span>{paused ? "Paused" : "Live"}</span>
          </label>
          <div className="flex items-center gap-2">
            <span className="text-xs">Metric:</span>
            <span className="px-2 py-0.5 rounded border text-xs">
              {showMetric === "bytes" ? "Bytes/s" : "Packets/s"}
            </span>
          </div>
        </div>
      </div>

      {/* KPI chips */}
      <div className="flex flex-wrap justify-center gap-2 mb-2 mt-3 text-xs">
        <div
          className="px-2 py-1 rounded-full border flex items-center gap-2"
          title="Total packets in window"
        >
          <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: barColor }} />
          <span>Packets:</span>
          <span className="tabular-nums">{totals.count}</span>
        </div>
        <div
          className="px-2 py-1 rounded-full border flex items-center gap-2"
          title="Total bytes in window"
        >
          <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: bytesColor }} />
          <span>Bytes:</span>
          <span className="tabular-nums">{humanBytes(totals.bytes)}</span>
        </div>
        <div
          className="px-2 py-1 rounded-full border flex items-center gap-2"
          title="Total attacks in window"
        >
          <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: attackLine }} />
          <span>Attacks:</span>
          <span className="tabular-nums">{totals.attacks}</span>
        </div>
      </div>

      {/* Chart */}
      <div style={{ width: "100%", height: 260 }} className="mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis allowDecimals={false} />
            <Tooltip
              formatter={(val: any, key: string) =>
                key === "bytes" ? humanBytes(val as number) : val
              }
              labelFormatter={(l) => l}
            />
            {showMetric === "bytes" ? (
              <Bar dataKey="bytes" name="Bytes" fill={bytesColor}>
                {data.map((_, i) => (
                  <Cell key={i} />
                ))}
              </Bar>
            ) : (
              <Bar dataKey="count" name="Packets" fill={barColor}>
                {data.map((_, i) => (
                  <Cell key={i} />
                ))}
              </Bar>
            )}
            <Line
              type="monotone"
              dataKey="attacks"
              name="Attacks"
              stroke={attackLine}
              dot={false}
              yAxisId={0}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 text-center text-xs text-muted-foreground">
        Data: /api/packets/timeseries • {canId ? `CAN ID: ${canId}` : "All CAN IDs"}
      </div>
    </div>
  );
}
