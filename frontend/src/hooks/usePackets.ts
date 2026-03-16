import { useEffect, useMemo, useState } from "react";
import { useTrainSocket } from "@/hooks/useTrainSocket";

export type Packet = {
  ts: number;           // seconds or ms
  id?: string;
  can_id?: number;
  dlc?: number;
  size_bytes?: number;
  signal?: number;
  bar_height?: number;
  attack?: boolean;
  attack_type?: string;
  severity?: "low" | "medium" | "high";
  score?: number;
  [k: string]: any;
};

export type UsePacketsOptions = {
  wsUrl?: string;     // default "/ws/traffic"
  maxPoints?: number; // default 500
};

function resolveWsUrl(pathOrUrl: string) {
  if (pathOrUrl.startsWith("ws://") || pathOrUrl.startsWith("wss://")) return pathOrUrl;
  const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
  const scheme = isHttps ? "wss" : "ws";
  const host = typeof window !== "undefined" ? window.location.host : "localhost:8000";
  return `${scheme}://${host}${pathOrUrl}`;
}

export function usePackets(opts: UsePacketsOptions = {}) {
  const wsUrl = resolveWsUrl(opts.wsUrl ?? "/ws/traffic");
  const maxPoints = Math.max(50, opts.maxPoints ?? 500);

  const { connected, lastMessage } = useTrainSocket(wsUrl, true);

  const [packets, setPackets] = useState<Packet[]>([]);

  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "ping" || lastMessage.type === "pong" || lastMessage.type === "hello") return;

    if (lastMessage.type === "packet" && lastMessage.payload) {
      const p = lastMessage.payload as Packet;
      setPackets((prev) => {
        const next = [...prev, normalizePacket(p)];
        return next.length > maxPoints ? next.slice(next.length - maxPoints) : next;
      });
      return;
    }

    if (Array.isArray(lastMessage)) {
      const arr = (lastMessage as Packet[]).map(normalizePacket);
      if (arr.length) {
        setPackets((prev) => {
          const next = [...prev, ...arr];
          return next.length > maxPoints ? next.slice(next.length - maxPoints) : next;
        });
      }
    }
  }, [lastMessage, maxPoints]);

  const data = useMemo(() => packets, [packets]);

  return { data, packets, connected, wsUrl };
}

function normalizePacket(p: Packet): Packet {
  const tsMs = p.ts > 2e12 ? p.ts : p.ts * 1000;
  return { ...p, ts: tsMs };
}

export default usePackets;
