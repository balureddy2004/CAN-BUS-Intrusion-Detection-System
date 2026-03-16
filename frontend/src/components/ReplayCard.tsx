import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Square, Gauge } from "lucide-react";
import { toast } from "sonner";

/* ----------------------------- Types ----------------------------- */

type ReplayState = {
  running: boolean;
  speed: number;
  mode: string;
  position: number;
  total_frames: number;
  progress_pct: number;
  model_path?: string;
};

/* ----------------------------- Component ----------------------------- */

export default function ReplayCard({
  className = "",
}: {
  className?: string;
}) {
  const [state, setState] = useState<ReplayState | null>(null);
  const [speed, setSpeed] = useState(1.0);
  const [loading, setLoading] = useState(false);

  /* ----------------------------- API Helpers ----------------------------- */

  const fetchState = async () => {
    try {
      const res = await fetch("/api/replay/state");
      const json = await res.json();
      if (json?.state) setState(json.state);
    } catch (err) {
      console.error(err);
    }
  };

  const startReplay = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/replay/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speed,
          mode: "direct",
        }),
      });
      const json = await res.json();
      if (!json.ok) throw new Error(json.error || "Start failed");
      toast.success("Replay started");
      fetchState();
    } catch (e: any) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }
  };

  const stopReplay = async () => {
    try {
      await fetch("/api/replay/stop", { method: "POST" });
      toast.success("Replay stopped");
      fetchState();
    } catch {
      toast.error("Failed to stop replay");
    }
  };

  const setReplaySpeed = async () => {
    try {
      await fetch("/api/replay/speed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed }),
      });
      toast.success(`Speed set to ${speed}×`);
      fetchState();
    } catch {
      toast.error("Failed to set speed");
    }
  };

  /* ----------------------------- Lifecycle ----------------------------- */

  useEffect(() => {
    fetchState();
    const id = setInterval(fetchState, 2000);
    return () => clearInterval(id);
  }, []);

  /* ----------------------------- Render ----------------------------- */

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Gauge className="w-5 h-5" />
          CAN Log Replay (Phase 6)
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Status */}
        <div className="text-sm text-muted-foreground">
          Status:{" "}
          <span className={state?.running ? "text-green-500" : "text-red-500"}>
            {state?.running ? "Running" : "Stopped"}
          </span>
        </div>

        {/* Progress */}
        <div className="w-full bg-muted h-2 rounded">
          <div
            className="h-2 bg-primary rounded transition-all"
            style={{ width: `${state?.progress_pct ?? 0}%` }}
          />
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <Button onClick={startReplay} disabled={loading || state?.running}>
            <Play className="w-4 h-4 mr-2" />
            Start
          </Button>

          <Button
            variant="destructive"
            onClick={stopReplay}
            disabled={!state?.running}
          >
            <Square className="w-4 h-4 mr-2" />
            Stop
          </Button>

          <div className="flex items-center gap-2 ml-auto">
            <input
              type="number"
              min={0.1}
              step={0.1}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-20 px-2 py-1 border rounded"
            />
            <Button variant="secondary" onClick={setReplaySpeed}>
              Apply
            </Button>
          </div>
        </div>

        {/* Metadata */}
        {state?.model_path && (
          <div className="text-xs text-muted-foreground break-all">
            Model: {state.model_path}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
