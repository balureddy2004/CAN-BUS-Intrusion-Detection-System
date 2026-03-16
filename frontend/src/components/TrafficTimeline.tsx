// src/components/TrafficTimeline.tsx
import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  Cell,
} from "recharts";
import { AlertTriangle, Activity } from "lucide-react";
import { useTrainSocket } from "@/hooks/useTrainSocket";

// ---------------------------------------------------------
// TrafficTimeline.tsx
// Passive real-time CAN traffic visualizer with PAUSE
// ---------------------------------------------------------

interface Packet {
  ts: number;
  id: string;
  can_id: number;
  dlc: number;
  size_bytes: number;
  bar_height?: number;
  attack?: boolean;
  attack_type?: string;
  severity?: "low" | "medium" | "high";
}

type Datum = {
  t: number;
  h: number;
  can_id?: number;
  id?: string;
  attack?: boolean;
  attack_type?: string;
  severity?: string;
};

const severityFill = (s?: string) => {
  switch (s) {
    case "high":
      return "#ef4444";
    case "medium":
      return "#f59e0b";
    case "low":
      return "#22c55e";
    default:
      return "#60a5fa";
  }
};

// 🔥 ORIGINAL animated bar (UNCHANGED)
const AnimatedBar: React.FC<any> = ({ x, y, width, height, payload }) => {
  const isAttack = !!payload?.attack;
  const fill = isAttack ? severityFill(payload.severity) : "url(#barGradient)";

  const Pulse = () => (
    <motion.rect
      x={x - 1}
      y={y - Math.max(2, height * 0.12)}
      width={Math.max(1, width) + 2}
      height={Math.max(0, height + Math.max(2, height * 0.12))}
      rx={6}
      fill={fill}
      opacity={0.16}
      initial={{ scaleY: 0.9, opacity: 0.08 }}
      animate={{ scaleY: [1, 1.25, 1], opacity: [0.18, 0.6, 0.18] }}
      transition={{ repeat: Infinity, duration: 0.9, ease: "easeInOut" }}
      style={{ transformOrigin: "bottom" }}
    />
  );

  return (
    <g>
      {isAttack && <Pulse />}
      <motion.rect
        initial={{ scaleY: 0, opacity: 0 }}
        animate={{ scaleY: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 180, damping: 22, mass: 0.6 }}
        x={x}
        y={y}
        width={Math.max(1, width)}
        height={Math.max(0, height)}
        fill={fill}
        opacity={isAttack ? 0.98 : 0.85}
        rx={3}
        style={{ transformOrigin: "bottom" }}
      />
      {isAttack && (
        <motion.circle
          cx={x + width / 2}
          cy={y - 6}
          r={6}
          fill={fill}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: [0, 1.3, 1], opacity: [0, 0.9, 0.9] }}
          transition={{ duration: 0.36, ease: "easeOut" }}
        />
      )}
    </g>
  );
};

// Tooltip (UNCHANGED)
const BarTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload as Datum;

  return (
    <div className="rounded-xl border bg-white/80 backdrop-blur px-3 py-2 text-xs shadow-xl">
      <div className="font-medium">
        {new Date(p.t).toLocaleTimeString()}
      </div>
      <div>
        ID: <span className="font-mono">{p.can_id?.toString(16)}</span>
      </div>
      {p.attack && (
        <div className="mt-1 flex items-center gap-1">
          <AlertTriangle className="h-3 w-3" />
          <span className="capitalize">{p.attack_type}</span>
        </div>
      )}
    </div>
  );
};

const asDatum = (p: Packet): Datum => ({
  t: p.ts > 2e12 ? p.ts : p.ts * 1000,
  h: Math.max(1, p.bar_height ?? p.size_bytes ?? 1),
  can_id: p.can_id,
  id: p.id,
  attack: p.attack,
  attack_type: p.attack_type,
  severity: p.severity,
});

export default function TrafficTimeline({
  wsUrl = "/ws/traffic",
}: {
  wsUrl?: string;
}) {
  const fullWsUrl =
    wsUrl.startsWith("ws")
      ? wsUrl
      : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}${wsUrl}`;

  const { connected, lastMessage } = useTrainSocket(fullWsUrl, true);
  const [data, setData] = useState<Datum[]>([]);
  const [paused, setPaused] = useState(false);
  const MAX_POINTS = 380;

  useEffect(() => {
    if (paused) return;
    if (!lastMessage) return;
    if ((lastMessage as any).type !== "packet") return;

    const d = asDatum((lastMessage as any).payload as Packet);
    setData((prev) => {
      const merged = [...prev, d];
      return merged.length > MAX_POINTS
        ? merged.slice(-MAX_POINTS)
        : merged;
    });
  }, [lastMessage, paused]);

  const chartData = useMemo(
    () => data.map((d, i) => ({ idx: i, ...d })),
    [data]
  );

  return (
    <div className="rounded-2xl border bg-gradient-to-b from-slate-950/60 to-slate-900/40 backdrop-blur p-4 shadow-xl">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-white">
          <Activity className="h-4 w-4" />
          Live Traffic
        </div>

        <div className="flex items-center gap-4 text-xs">
          <label className="flex items-center gap-1 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={paused}
              onChange={(e) => setPaused(e.target.checked)}
            />
            <span>{paused ? "Paused" : "Live"}</span>
          </label>

          <span className={connected ? "text-green-400" : "text-red-400"}>
            ● {connected ? "WS connected" : "WS disconnected"}
          </span>
        </div>
      </div>

      {/* ✅ SINGLE SOURCE OF HEIGHT */}
      <div className="h-[260px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ left: 8, right: 8, top: 8, bottom: 8 }}
          >
            <defs>
              <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.95} />
                <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.2} />
              </linearGradient>
            </defs>

            <CartesianGrid
              vertical={false}
              strokeDasharray="3 3"
              stroke="#1f2937"
            />

            <XAxis dataKey="idx" hide />

            <YAxis
              domain={[0, "dataMax"]}
              tick={{ fill: "#9ca3af", fontSize: 11 }}
            />

            <Tooltip content={<BarTooltip />} />
            <Legend content={() => null} />

            <Bar dataKey="h" isAnimationActive={false} shape={<AnimatedBar />}>
              {chartData.map((_, i) => (
                <Cell key={i} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
