// src/components/AttackTimeline.tsx
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
} from "recharts";
import { useTrainSocket } from "@/hooks/useTrainSocket";

export type AlertMsg = {
  timestamp?: number | string;
  ts?: number;
  attack_type?: string;
  type?: string;
  severity?: string;
  score?: number;
  [k: string]: any;
};

type AggRow = {
  label: string;
  epochSec: number;
  [attackType: string]: number | string;
};

export interface AttackTimelineProps {
  apiPath?: string;
  windowSeconds?: number;
  updateIntervalMs?: number;
  wsUrl?: string;
  className?: string;
}

const DEFAULT_ATTACKS = [
  "dos",
  "flood",
  "suspicious_frame",
  "impersonation",
  "replay",
  "unknown",
] as const;

const pretty = (s: string) =>
  s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

function epochSecFromMsg(m: AlertMsg) {
  if (typeof m.timestamp === "number") return Math.floor(m.timestamp / 1000);
  if (typeof m.ts === "number") return Math.floor(m.ts / 1000);
  return Math.floor(Date.now() / 1000);
}

function colorFromString(seed: string) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h << 5) - h + seed.charCodeAt(i);
  return `hsl(${Math.abs(h) % 360}, 70%, 55%)`;
}

export default function AttackTimeline({
  apiPath = "/api/alerts",
  windowSeconds = 30,
  updateIntervalMs = 1000,
  wsUrl = "/ws/alerts",
  className = "",
}: AttackTimelineProps) {
  const [agg, setAgg] = useState<AggRow[]>([]);
  const [recent, setRecent] = useState<AlertMsg[]>([]);
  const [paused, setPaused] = useState(false);
  const [ingested, setIngested] = useState(0);

  const pollRef = useRef<number | null>(null);

  // -------------------------
  // Aggregated timeline poll
  // -------------------------
  useEffect(() => {
    async function poll() {
      if (paused) return;
      try {
        const res = await fetch(
          `/api/alerts/timeseries?window=1s&since=${windowSeconds}s`
        );
        if (!res.ok) return;
        const data = await res.json();

        const rows: AggRow[] = data.map((b: any) => {
          const sec = Math.floor(b.t / 1000);
          const row: AggRow = {
            label: new Date(sec * 1000).toLocaleTimeString(),
            epochSec: sec,
          };
          DEFAULT_ATTACKS.forEach((k) => (row[k] = b[k] ?? 0));
          return row;
        });

        setAgg(rows);
      } catch {}
    }

    poll();
    pollRef.current = window.setInterval(poll, updateIntervalMs);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [paused, windowSeconds, updateIntervalMs]);

  // -------------------------
  // Live WebSocket alerts
  // -------------------------
  const fullWs =
    wsUrl.startsWith("ws")
      ? wsUrl
      : `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${wsUrl}`;

  const { lastMessage } = useTrainSocket(fullWs, true);

  useEffect(() => {
    if (!lastMessage) return;
    if ((lastMessage as any).type === "ping") return;

    const append = (msg: AlertMsg) => {
      const attack = msg.attack_type || msg.type || "unknown";
      setRecent((prev) =>
        [
          {
            ...msg,
            attack_type: attack,
            ts: epochSecFromMsg(msg) * 1000,
          },
          ...prev,
        ].slice(0, 10)
      );
      setIngested((c) => c + 1);
    };

    if (
      (lastMessage as any).type === "replay_alerts" &&
      Array.isArray((lastMessage as any).items)
    ) {
      (lastMessage as any).items.forEach(append);
    } else if (typeof lastMessage === "object") {
      append(lastMessage as AlertMsg);
    }
  }, [lastMessage]);

  const attackTypes = useMemo(() => {
    const set = new Set<string>(DEFAULT_ATTACKS as unknown as string[]);
    agg.forEach((r) =>
      Object.keys(r).forEach((k) => {
        if (k !== "label" && k !== "epochSec") set.add(k);
      })
    );
    return Array.from(set);
  }, [agg]);

  const colors = useMemo(() => {
    const m: Record<string, string> = {};
    attackTypes.forEach((t) => (m[t] = colorFromString(t)));
    return m;
  }, [attackTypes]);

  return (
    <div
      className={`rounded-2xl border bg-gradient-to-b from-slate-950/60 to-slate-900/40 backdrop-blur p-4 shadow-xl ${className}`}
    >
      {/* Status row — SAME alignment as Live CAN Traffic */}
      <div className="mb-3 flex items-center justify-between text-xs text-muted-foreground">
        <span>
          Last {windowSeconds}s • ingested {ingested}
        </span>

        <label className="flex items-center gap-1 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={paused}
            onChange={(e) => setPaused(e.target.checked)}
          />
          <span>{paused ? "Paused" : "Live"}</span>
        </label>
      </div>

      {/* Chart */}
      <div className="h-[260px] w-full">
        <ResponsiveContainer>
          <BarChart data={agg}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            {attackTypes.map((t) => (
              <Bar key={t} dataKey={t} stackId="a">
                {agg.map((_, i) => (
                  <Cell key={`${t}-${i}`} fill={colors[t]} />
                ))}
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Attacks */}
      <div className="mt-4">
        <div className="text-center text-sm font-medium mb-2">
          Recent Attacks
        </div>

        {recent.length === 0 ? (
          <div className="text-xs text-center text-muted-foreground">
            No alerts yet
          </div>
        ) : (
          <div className="space-y-1">
            {recent.map((a, i) => {
              const k = a.attack_type || "unknown";
              return (
                <div
                  key={i}
                  className="flex justify-between text-xs border rounded p-2"
                >
                  <div className="flex gap-2 items-center">
                    <span
                      className="w-3 h-3 rounded"
                      style={{ background: colors[k] }}
                    />
                    <b>{pretty(k)}</b>
                    <span className="text-muted-foreground">
                      {a.severity ?? "-"}
                    </span>
                    {a.score && (
                      <span className="text-muted-foreground">
                        score {a.score.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <span className="text-muted-foreground">
                    {new Date(
                      epochSecFromMsg(a) * 1000
                    ).toLocaleTimeString()}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
