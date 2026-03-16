// src/components/LiveMonitorCard.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Signal, CheckCircle2, AlertTriangle, Gauge } from "lucide-react";
import {
  Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";
import { useTrainSocket } from "@/hooks/useTrainSocket";

// Define TrainState locally to avoid type import mismatch
type TrainState = "idle" | "ready" | "training" | "paused" | "stopped" | "completed" | "error";

type Point = { epoch: number; loss?: number; valLoss?: number; accuracy?: number; valAccuracy?: number };

interface LiveMonitorCardProps {
  statusPollUrl?: string; // optional polling fallback
  wsUrl?: string;         // default -> same-origin "/ws/train" (Vite proxy)
  className?: string;
}

function appendLog(setter: React.Dispatch<React.SetStateAction<string>>, msg?: string) {
  if (msg == null) return;
  setter((prev) => (prev ? `${prev}\n${msg}` : msg));
}

export default function LiveMonitorCard({
  wsUrl = "/ws/train",
  statusPollUrl,
  className,
}: LiveMonitorCardProps) {
  // Build a full WS URL if a relative path is provided
  const fullWsUrl = useMemo(() => {
    if (!wsUrl) return "";
    if (wsUrl.startsWith("ws://") || wsUrl.startsWith("wss://")) return wsUrl;
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${window.location.host}${wsUrl}`;
  }, [wsUrl]);

  const { connected, lastMessage } = useTrainSocket(fullWsUrl, Boolean(fullWsUrl));
  const [state, setState] = useState<TrainState>("idle");
  const [progress, setProgress] = useState(0);
  const [series, setSeries] = useState<Point[]>([]);
  const [logs, setLogs] = useState<string>("");

  const MAX_POINTS = 500;

  const logRef = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  // Handle WS messages
  useEffect(() => {
    if (!lastMessage) return;

    switch (lastMessage.type) {
      case "progress": {
        const raw = Number(lastMessage.progress);
        const pct = Number.isFinite(raw) ? Math.max(0, Math.min(100, raw)) : 0;
        setProgress(pct);
        break;
      }
      case "metric": {
        const p: Point = {
          epoch: Number(lastMessage.epoch) || 0,
          loss: typeof lastMessage.loss === "number" ? lastMessage.loss : undefined,
          valLoss: typeof lastMessage.valLoss === "number" ? lastMessage.valLoss : undefined,
          accuracy: typeof lastMessage.accuracy === "number" ? lastMessage.accuracy : undefined,
          valAccuracy: typeof lastMessage.valAccuracy === "number" ? lastMessage.valAccuracy : undefined,
        };
        setSeries((prev) => {
          const next = [...prev, p];
          return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
        });
        break;
      }
      case "status": {
        if (typeof lastMessage.state === "string") {
          setState(lastMessage.state as TrainState);
        }
        appendLog(setLogs, lastMessage.msg);
        break;
      }
      case "log": {
        appendLog(setLogs, lastMessage.msg);
        break;
      }
      // tolerate keepalives / replays silently
      default:
        break;
    }
  }, [lastMessage]);

  // Optional polling fallback (only when no WS URL was provided)
  useEffect(() => {
    if (!statusPollUrl || wsUrl) return;

    let t: number | null = null;
    const poll = async () => {
      try {
        const res = await fetch(statusPollUrl);
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        if (typeof data.progress === "number") {
          const pct = Math.max(0, Math.min(100, data.progress));
          setProgress(pct);
        }
        if (typeof data.state === "string") setState(data.state as TrainState);
        if (data.msg) appendLog(setLogs, String(data.msg));
        if (typeof data.epoch === "number") {
          setSeries((p) => {
            const next = [
              ...p,
              {
                epoch: data.epoch,
                loss: data.loss,
                valLoss: data.valLoss,
                accuracy: data.accuracy,
                valAccuracy: data.valAccuracy,
              },
            ];
            return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next;
          });
        }
      } catch {
        // ignore transient errors
      } finally {
        t = window.setTimeout(poll, 2000);
      }
    };

    poll();
    return () => {
      if (t !== null) {
        clearTimeout(t);
        t = null;
      }
    };
  }, [statusPollUrl, wsUrl]);

  const badge = useMemo(() => {
    const map: Record<TrainState, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
      idle: { label: "Idle", variant: "outline" },
      ready: { label: "Ready", variant: "secondary" },
      training: { label: "Training", variant: "default" },
      paused: { label: "Paused", variant: "secondary" },
      stopped: { label: "Stopped", variant: "outline" },
      completed: { label: "Completed", variant: "secondary" },
      error: { label: "Error", variant: "destructive" },
    };
    return map[state];
  }, [state]);

  return (
    <Card className={`w-full max-w-6xl mx-auto rounded-2xl ${className ?? ""}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            <CardTitle>Live Training Monitor</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={badge.variant}>{badge.label}</Badge>
            <Badge variant="outline" className="gap-1">
              <Gauge className="h-4 w-4" />
              {progress.toFixed(0)}%
            </Badge>
            <Badge variant={connected ? "default" : "outline"} className="gap-1">
              <Signal className="h-4 w-4" />
              {connected ? "WS: connected" : "WS: disconnected"}
            </Badge>
          </div>
        </div>
        <CardDescription>Real-time loss/accuracy via WebSocket with polling fallback.</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div>
          <Progress value={progress} className="h-3" />
        </div>

        <Tabs defaultValue="loss">
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="loss">Loss</TabsTrigger>
            <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
          </TabsList>

          <TabsContent value="loss" className="mt-4">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="loss" dot={false} name="Train Loss" />
                  <Line type="monotone" dataKey="valLoss" dot={false} name="Val Loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="accuracy" className="mt-4">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={series}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" dot={false} name="Train Acc" />
                  <Line type="monotone" dataKey="valAccuracy" dot={false} name="Val Acc" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="logs" className="mt-4">
            <Textarea ref={logRef} className="h-64 resize-none text-xs font-mono" value={logs} readOnly />
          </TabsContent>
        </Tabs>
      </CardContent>

      <CardFooter className="flex items-center gap-2">
        <CheckCircle2 className="h-4 w-4" />
        <p className="text-xs text-muted-foreground">You’ll see points appear as epochs stream in.</p>
        {state === "error" && (
          <span className="flex items-center gap-1 text-xs text-red-500">
            <AlertTriangle className="h-4 w-4" /> Check backend logs.
          </span>
        )}
      </CardFooter>
    </Card>
  );
}
