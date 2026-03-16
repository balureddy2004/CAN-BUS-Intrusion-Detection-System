// src/components/TrainingCard.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Play, Square, RefreshCw, Upload, Download, Cpu, Settings, FolderOpen, Gauge, Pause, Activity } from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { useTrainSocket } from "@/hooks/useTrainSocket";
import ModelSelector from "@/components/ModelSelector";

type TrainStatus = "idle" | "ready" | "training" | "paused" | "stopped" | "completed" | "error";

type MetricEvt = {
  type: "metric";
  epoch: number;
  loss: number;
  valLoss: number;
  accuracy: number;
  valAccuracy: number;
  t?: number;
};
type ProgressEvt = { type: "progress"; progress: number; epoch: number; t?: number };
type LogEvt = { type: "log"; msg: string; t?: number };
type StatusEvt = { type: "status"; state: TrainStatus; msg?: string; t?: number };
type ConfusionEvt = { type: "confusion"; matrix?: number[][]; labels?: string[]; t?: number };

type ReplayMetrics = { type: "replay_metrics"; items: MetricEvt[] };
type ReplayProgress = { type: "replay_progress"; items: ProgressEvt[] };
type ReplayLogs = { type: "replay_logs"; items: LogEvt[] };

type AnyEvt = MetricEvt | ProgressEvt | LogEvt | StatusEvt | ReplayMetrics | ReplayProgress | ReplayLogs | ConfusionEvt | { type?: string; [k: string]: any };

export interface TrainingCardProps {
  className?: string;
  startUrl?: string;
  pauseUrl?: string;
  stopUrl?: string;
  resetUrl?: string;
  downloadModelUrl?: string;
  wsUrl?: string;
}

function clamp(x: number, min = 0, max = 100) {
  if (!Number.isFinite(x)) return min;
  return Math.max(min, Math.min(max, x));
}

async function postJSON(url: string, body?: unknown) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export default function TrainingCard(props: TrainingCardProps) {
  const { className } = props;

  const MODEL_ID_MAP: Record<string, string> = {
    "Can Detector V1": "can_detector_v1",
    "can_detector_v1": "can_detector_v1",
    "RandomForest": "random_forest",
    "random_forest": "random_forest",
    "XGBoost": "xgboost",
    "xgboost": "xgboost",
    "LSTM": "lstm",
    "lstm": "lstm",
  };

  const DATASET_ID_MAP: Record<string, string> = {
    "OCS-Lab": "ocs-lab",
    "ocs-lab": "ocs-lab",
    "Car Hacking": "car-hacking",
    "car-hacking": "car-hacking",
    "DOE-CAN": "doe-can",
    "doe-can": "doe-can",
  };

  const API = "";
  const startUrl = props.startUrl ?? `${API}/train/start`;
  const pauseUrl = props.pauseUrl ?? `${API}/train/pause`;
  const stopUrl = props.stopUrl ?? `${API}/train/stop`;
  const resetUrl = props.resetUrl ?? `${API}/train/reset`;
  const modelUrl = props.downloadModelUrl ?? `${API}/train/model`;

  const fullWsUrl = useMemo(() => {
    const path = props.wsUrl ?? "/ws/train";
    if (path.startsWith("ws:") || path.startsWith("wss:")) return path;
    const loc = typeof window !== "undefined" ? window.location : ({ protocol: "http:", host: "localhost:8000" } as any);
    const scheme = loc.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${loc.host}${path}`;
  }, [props.wsUrl]);

  const { connected, lastMessage } = useTrainSocket(fullWsUrl, true);

  const [status, setStatus] = useState<TrainStatus>("idle");
  const [progress, setProgress] = useState<number>(0);
  const [epoch, setEpoch] = useState<number>(0);
  const [metrics, setMetrics] = useState<MetricEvt[]>([]);
  const [logs, setLogs] = useState<string>("");

  const [confusion, setConfusion] = useState<{ matrix: number[][]; labels: string[] } | null>(null);

  // preset dataset UI state
  const [presetOptions, setPresetOptions] = useState<{ id: string; label: string }[]>([
    { id: "ocs-lab", label: "OCS-Lab" },
    { id: "car-hacking", label: "Car Hacking" },
    { id: "doe-can", label: "DOE-CAN" },
  ]);
  const [datasetsLoading, setDatasetsLoading] = useState<boolean>(false);

  const [presetDataset, setPresetDataset] = useState<string>("ocs-lab");
  const [datasetName, setDatasetName] = useState<string>(""); // filename from file input (UI only)
  const [datasetPath, setDatasetPath] = useState<string>(""); // absolute path input by user
  const [isAugment, setIsAugment] = useState<boolean>(true);
  const [maxRows, setMaxRows] = useState<number>(2000);

  const [hyper, setHyper] = useState({
    model: "RandomForest",
    batchSize: 256,
    epochs: 15,
    learningRate: 0.001,
    optimizer: "adam",
    trainSplit: 0.8,
    seed: 42,
  });

  const logRef = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    let mounted = true;
    async function loadPresets() {
      setDatasetsLoading(true);
      try {
        const res = await fetch("/api/datasets");
        if (!res.ok) throw new Error("no datasets endpoint");
        const json = await res.json();
        if (Array.isArray(json) && json.length) {
          const items = json.map((it: any) =>
            typeof it === "string" ? { id: String(it).trim().toLowerCase().replace(/\s+/g,"-"), label: it } :
            { id: (it.id ?? String(it.label ?? JSON.stringify(it))).toString(), label: it.label ?? String(it.id ?? it) }
          );
          if (mounted && items.length) {
            setPresetOptions(items);
            const current = presetDataset;
            if (!items.some((p) => p.id === current)) {
              setPresetDataset(items[0].id);
            }
          }
        }
      } catch (err) {
        console.debug("[TrainingCard] datasets fetch failed:", err);
      } finally {
        if (mounted) setDatasetsLoading(false);
      }
    }
    loadPresets();
    return () => { mounted = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if ((!presetDataset || presetDataset === "") && presetOptions.length) {
      setPresetDataset(presetOptions[0].id);
    } else if (presetOptions.length && !presetOptions.some((p) => p.id === presetDataset)) {
      setPresetDataset(presetOptions[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetOptions]);

  useEffect(() => {
    if (!lastMessage) return;

    const t = lastMessage as AnyEvt;
    const typ = String(t?.type ?? "");

    switch (typ) {
      case "status": {
        const m = lastMessage as StatusEvt;
        setStatus(m.state);
        if (m.msg) setLogs((prev) => (prev ? prev + "\n" : "") + m.msg);
        break;
      }
      case "metric": {
        const m = lastMessage as MetricEvt;
        const safeEpoch = Number.isFinite(m.epoch) ? m.epoch : 0;
        setEpoch(safeEpoch);
        setMetrics((prev) => {
          const next = [...prev, sanitizeMetric(m)];
          if (next.length > 500) next.splice(0, next.length - 500);
          return next;
        });
        if (hyper.epochs && hyper.epochs > 0) {
          const pct = clamp((safeEpoch / hyper.epochs) * 100, 0, 100);
          setProgress((p) => (pct > p ? pct : p));
        }
        break;
      }
      case "progress": {
        const p = lastMessage as ProgressEvt;
        const pct = clamp(p.progress || 0, 0, 100);
        setProgress(pct);
        if (typeof p.epoch === "number" && Number.isFinite(p.epoch)) setEpoch(p.epoch);
        break;
      }
      case "log": {
        const l = lastMessage as LogEvt;
        if (l.msg) setLogs((prev) => (prev ? prev + "\n" : "") + l.msg);
        break;
      }
      case "replay_metrics": {
        const items = (t as ReplayMetrics).items || [];
        if (items.length) {
          const sanitized = items.map(sanitizeMetric);
          setMetrics((prev) => {
            const merged = [...prev, ...sanitized];
            return merged.length > 500 ? merged.slice(-500) : merged;
          });
          const lastEp = sanitized[sanitized.length - 1]?.epoch ?? 0;
          if (Number.isFinite(lastEp)) setEpoch(lastEp);
          if (hyper.epochs && hyper.epochs > 0) {
            const pct = clamp((lastEp / hyper.epochs) * 100, 0, 100);
            setProgress((p) => (pct > p ? pct : p));
          }
        }
        break;
      }
      case "replay_progress": {
        const items = (t as ReplayProgress).items || [];
        if (items.length) {
          const last = items[items.length - 1];
          const pct = clamp(last.progress || 0, 0, 100);
          setProgress(pct);
          if (typeof last.epoch === "number" && Number.isFinite(last.epoch)) setEpoch(last.epoch);
        }
        break;
      }
      case "replay_logs": {
        const items = (t as ReplayLogs).items || [];
        if (items.length) {
          const chunk = items.map((i) => i.msg ?? "").filter(Boolean).join("\n");
          if (chunk) setLogs((prev) => (prev ? prev + "\n" : "") + chunk);
        }
        break;
      }
      case "confusion": {
        const c = lastMessage as ConfusionEvt;
        let mtx = c.matrix;
        let labels = c.labels || ["neg", "pos"];
        if (!mtx && (lastMessage as any).confusion_matrix) {
          mtx = (lastMessage as any).confusion_matrix;
        }
        if (Array.isArray(mtx) && mtx.length >= 2) {
          setConfusion({ matrix: mtx as number[][], labels });
        } else {
          const obj = lastMessage as any;
          if (obj && ["tn", "tp", "fn", "fp"].every((k) => k in obj)) {
            setConfusion({ matrix: [[Number(obj.tn) || 0, Number(obj.fp) || 0], [Number(obj.fn) || 0, Number(obj.tp) || 0]], labels });
          }
        }
        break;
      }
      default:
        if ((lastMessage as any) && typeof lastMessage === "object") {
          setLogs((prev) => prev + "\n" + JSON.stringify(lastMessage));
        }
        break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastMessage]);

  const chartData = useMemo(
    () =>
      metrics.map((m) => ({
        epoch: m.epoch,
        loss: Number.isFinite(m.loss) ? Number((m.loss as number).toFixed?.(4) ?? m.loss) : 0,
        valLoss: Number.isFinite(m.valLoss) ? Number((m.valLoss as number).toFixed?.(4) ?? m.valLoss) : 0,
        acc: Number.isFinite(m.accuracy) ? Number((m.accuracy as number).toFixed?.(4) ?? m.accuracy) : 0,
        valAcc: Number.isFinite(m.valAccuracy) ? Number((m.valAccuracy as number).toFixed?.(4) ?? m.valAccuracy) : 0,
      })),
    [metrics],
  );

  const onStart = async () => {
    // Resolve dataset send order:
    // 1) dataset_path (absolute path) if provided
    // 2) datasetName (upload filename) if provided
    // 3) presetDataset
    let dataset: string | null = null;
    let dataset_path: string | null = null;

    if (datasetPath && datasetPath.trim()) {
      dataset_path = datasetPath.trim();
    } else if (datasetName && datasetName.trim()) {
      dataset = datasetName.trim();
    } else if (presetDataset) {
      dataset = DATASET_ID_MAP[presetDataset] ?? presetDataset;
    } else {
      dataset = null;
    }

    const resolvedModelId = MODEL_ID_MAP[hyper.model] ?? hyper.model;

    const body: any = {
      model: hyper.model,
      modelId: resolvedModelId,
      epochs: hyper.epochs,
      batchSize: hyper.batchSize,
      learningRate: hyper.learningRate,
      optimizer: hyper.optimizer,
      trainSplit: hyper.trainSplit,
      seed: hyper.seed,
      augment: !!isAugment,
      dataset,
      maxRows: Number(maxRows) || undefined,
    };

    if (dataset_path) {
      body.dataset_path = dataset_path;
    }

    try {
      await postJSON(startUrl, body);
      setStatus("training");
      setLogs("");
      setProgress(0);
      setConfusion(null);
      toast.success("Training started");
    } catch (e: any) {
      toast.error(`Failed to start: ${e?.message ?? e}`);
    }
  };

  const onPauseResume = async () => {
    try {
      await postJSON(pauseUrl);
    } catch (e: any) {
      toast.error(`Failed: ${e?.message ?? e}`);
    }
  };

  const onStop = async () => {
    try {
      await postJSON(stopUrl);
    } catch (e: any) {
      toast.error(`Failed to stop: ${e?.message ?? e}`);
    }
  };

  const onReset = async () => {
    try {
      await postJSON(resetUrl);
    } catch {}
    setMetrics([]);
    setLogs("");
    setProgress(0);
    setStatus("idle");
    setConfusion(null);
  };

  const onDownload = async () => {
    try {
      const res = await fetch(modelUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "trained_model.bin";
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Download started");
    } catch (e: any) {
      toast.error(`Download failed: ${e?.message ?? e}`);
    }
  };

  const statusBadge =
    (({
      idle: { label: "Idle", variant: "outline" as const },
      ready: { label: "Ready", variant: "secondary" as const },
      training: { label: "Training", variant: "default" as const },
      paused: { label: "Paused", variant: "secondary" as const },
      stopped: { label: "Stopped", variant: "outline" as const },
      completed: { label: "Completed", variant: "secondary" as const },
      error: { label: "Error", variant: "destructive" as const },
    } as const)[status] ?? { label: "Unknown", variant: "outline" as const });

  const busy = status === "training" || status === "paused";

  function renderConfusion() {
    if (!confusion) return <p className="text-sm text-muted-foreground">No confusion matrix available.</p>;
    const m = confusion.matrix;
    const labels = confusion.labels || ["neg", "pos"];
    const tn = m?.[0]?.[0] ?? 0;
    const fp = m?.[0]?.[1] ?? 0;
    const fn = m?.[1]?.[0] ?? 0;
    const tp = m?.[1]?.[1] ?? 0;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    const cellStyle = (v: number) => ({
      padding: 12,
      minWidth: 72,
      textAlign: "center" as const,
      borderRadius: 8,
      background: `rgba(255,255,255,${Math.min(0.9, 0.05 + (v / Math.max(1, tp + fp + fn + tn)))})`,
      color: "white",
      fontWeight: 700,
    });

    return (
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-muted-foreground mb-2">Confusion Matrix</div>
          <div className="grid grid-cols-2 gap-2 bg-slate-800 p-3 rounded-lg">
            <div style={cellStyle(tn)}>TN<br/><span style={{fontWeight:400,fontSize:12}}>{tn}</span></div>
            <div style={cellStyle(fp)}>FP<br/><span style={{fontWeight:400,fontSize:12}}>{fp}</span></div>
            <div style={cellStyle(fn)}>FN<br/><span style={{fontWeight:400,fontSize:12}}>{fn}</span></div>
            <div style={cellStyle(tp)}>TP<br/><span style={{fontWeight:400,fontSize:12}}>{tp}</span></div>
          </div>
        </div>

        <div>
          <div className="text-sm text-muted-foreground mb-2">Metrics</div>
          <div className="bg-slate-800 p-3 rounded-lg">
            <div className="text-xs">Precision: <span className="font-mono">{precision.toFixed(4)}</span></div>
            <div className="text-xs">Recall: <span className="font-mono">{recall.toFixed(4)}</span></div>
            <div className="text-xs">F1: <span className="font-mono">{f1.toFixed(4)}</span></div>
            <div className="text-xs mt-2">Labels: <span className="font-mono">{labels.join(", ")}</span></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Card className={"w-full max-w-6xl mx-auto rounded-2xl shadow-md border-muted " + (className ?? "")}>
      <CardHeader className="space-y-2">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
              <Cpu className="h-6 w-6" />
            </motion.div>
            <CardTitle className="text-xl">Model Training</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={statusBadge.variant}>{statusBadge.label}</Badge>
            <Badge variant="outline" className="gap-1">
              <Gauge className="h-4 w-4" />
              {clamp(progress, 0, 100).toFixed(1)}%
            </Badge>
            <Badge variant={connected ? "secondary" : "destructive"} className="gap-1">
              <span className={`inline-block h-2 w-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
              WS: {connected ? "connected" : "disconnected"}
            </Badge>
          </div>
        </div>
        <CardDescription>Live training monitor with WebSocket replay. Pause/resume and download the trained model.</CardDescription>
      </CardHeader>

      <CardContent className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left */}
        <div className="xl:col-span-1 space-y-6">
          <div className="grid gap-4 p-4 border rounded-2xl">
            <div className="grid gap-2">
              <Label className="flex items-center gap-2">
                <FolderOpen className="h-4 w-4" />
                Preset Dataset
              </Label>

              {/* Native select used to avoid custom Select widget issues */}
              {datasetsLoading ? (
                <div className="text-xs text-muted-foreground">Loading datasets…</div>
              ) : (
                <select
                  className="w-full p-2 rounded border bg-transparent text-sm"
                  value={presetDataset}
                  onChange={(e) => {
                    setPresetDataset(e.target.value);
                    console.debug("[TrainingCard] presetDataset ->", e.target.value);
                  }}
                >
                  <option value="" disabled>
                    — select dataset —
                  </option>
                  {presetOptions.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              )}

              <p className="text-xs text-muted-foreground">Passed as name to backend (no file upload).</p>
            </div>

            <div className="grid gap-2">
              <Label className="flex items-center gap-2">
                <FolderOpen className="h-4 w-4" />
                Upload Custom Dataset (optional)
              </Label>
              <div className="flex items-center gap-2">
                <Input
                  id="dataset-file"
                  type="file"
                  accept=".csv,.parquet"
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setDatasetName(e.target.files?.[0]?.name ?? "")
                  }
                />
                <Button
                  variant="secondary"
                  type="button"
                  className="gap-2"
                  onClick={() => (document.getElementById("dataset-file") as HTMLInputElement)?.click()}
                >
                  <Upload className="h-4 w-4" /> Choose
                </Button>
              </div>
              {datasetName && <p className="text-xs text-muted-foreground">Selected: {datasetName}</p>}
              <p className="text-xs text-muted-foreground">Note: backend expects a dataset name or path — use absolute path below for real paths.</p>
            </div>

            {/* New: absolute dataset path input */}
            <div className="grid gap-2">
              <Label className="flex items-center gap-2">
                <FolderOpen className="h-4 w-4" />
                Absolute dataset path (optional)
              </Label>
              <Input
                placeholder="e.g. C:\data\ocs-lab\Attack_free_dataset.csv"
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                className="text-sm"
              />
              <p className="text-xs text-muted-foreground">
                If provided, this path will be sent as <code>dataset_path</code> to the backend and tried first. Only use trusted local paths.
              </p>
            </div>

            {/* New: row limiter */}
            <div className="grid gap-2">
              <Label className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Max rows (row limiter)
              </Label>
              <Input
                type="number"
                min={1}
                value={maxRows}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMaxRows(Number(e.target.value || 0))}
              />
              <p className="text-xs text-muted-foreground">Limit the number of rows loaded for training (safe for big datasets).</p>
            </div>

            {/* Keep ModelSelector without outer label to avoid duplicate label */}
            <div className="grid gap-2">
              <ModelSelector
                value={hyper.model}
                onChange={(v: string) => setHyper({ ...hyper, model: v })}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label>Epochs</Label>
                <Input
                  type="number"
                  min={1}
                  value={hyper.epochs}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setHyper({ ...hyper, epochs: Number(e.target.value) })
                  }
                />
              </div>
              <div>
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  min={1}
                  value={hyper.batchSize}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setHyper({ ...hyper, batchSize: Number(e.target.value) })
                  }
                />
              </div>
              <div>
                <Label>Learning Rate</Label>
                <Input
                  type="number"
                  step="0.0001"
                  value={hyper.learningRate}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setHyper({ ...hyper, learningRate: Number(e.target.value) })
                  }
                />
              </div>
              <div>
                <Label>Train Split</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={hyper.trainSplit}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setHyper({ ...hyper, trainSplit: Number(e.target.value) })
                  }
                />
              </div>
            </div>

            <div className="flex items-center justify-between p-3 rounded-xl border">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                <div>
                  <Label>Augment / Balance</Label>
                  <p className="text-xs text-muted-foreground">SMOTE/scale/normalize</p>
                </div>
              </div>
              <Switch checked={isAugment} onCheckedChange={(v: boolean) => setIsAugment(v)} />
            </div>

            <div className="flex gap-2">
              <Button className="gap-2 flex-1" onClick={onStart} disabled={status === "training"}>
                <Play className="h-4 w-4" /> Start
              </Button>
              <Button variant="secondary" className="gap-2" onClick={onPauseResume} disabled={!busy} title="Pause/Resume">
                {status === "paused" ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                {status === "paused" ? "Resume" : "Pause"}
              </Button>
              <Button variant="destructive" className="gap-2" onClick={onStop} disabled={!busy} title="Stop training">
                <Square className="h-4 w-4" /> Stop
              </Button>
            </div>

            <div className="flex items-center justify-between">
              <Button variant="outline" className="gap-2" onClick={onDownload}>
                <Download className="h-4 w-4" /> Best Model
              </Button>
              <Button variant="ghost" className="gap-2" onClick={onReset}>
                <RefreshCw className="h-4 w-4" /> Reset
              </Button>
            </div>
          </div>
        </div>

        {/* Right */}
        <div className="xl:col-span-2 space-y-6">
          <div className="space-y-2">
            <Label>Training Progress</Label>
            <Progress value={clamp(progress, 0, 100)} className="h-3" />
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>
                Epoch: <span className="text-foreground">{epoch}</span>
              </span>
              <span>{clamp(progress, 0, 100).toFixed(1)}%</span>
            </div>
          </div>

          <Tabs defaultValue="logs" className="w-full">
            <TabsList className="grid grid-cols-4 w-full">
              <TabsTrigger value="logs">Logs</TabsTrigger>
              <TabsTrigger value="loss">Loss</TabsTrigger>
              <TabsTrigger value="acc">Accuracy</TabsTrigger>
              <TabsTrigger value="confusion">Confusion</TabsTrigger>
            </TabsList>

            <TabsContent value="logs" className="mt-4">
              <ChartCard title="Training Logs" subtitle="Most recent at bottom">
                <Textarea ref={logRef} className="h-64 resize-none text-xs font-mono" value={logs} readOnly />
              </ChartCard>
            </TabsContent>

            <TabsContent value="loss" className="mt-4">
              <ChartCard title="Loss" subtitle="loss vs valLoss per epoch">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 16, bottom: 0, left: -10 }}>
                      <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="loss" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="valLoss" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </ChartCard>
            </TabsContent>

            <TabsContent value="acc" className="mt-4">
              <ChartCard title="Accuracy" subtitle="accuracy vs valAccuracy per epoch">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 16, bottom: 0, left: -10 }}>
                      <XAxis dataKey="epoch" tick={{ fontSize: 12 }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="acc" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="valAcc" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </ChartCard>
            </TabsContent>

            <TabsContent value="confusion" className="mt-4">
              <ChartCard title="Confusion Matrix" subtitle="Confusion matrix and basic metrics">
                <div className="p-2">
                  {renderConfusion()}
                </div>
              </ChartCard>
            </TabsContent>
          </Tabs>
        </div>
      </CardContent>

      <CardFooter className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">Tip: keep this tab open to stream metrics live during training.</p>
        <p className="text-xs text-muted-foreground flex items-center gap-1">
          <Activity className="h-3 w-3" /> v2.1 • Live Monitor
        </p>
      </CardFooter>
    </Card>
  );
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <Card className="rounded-2xl">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{title}</CardTitle>
        {subtitle && <CardDescription>{subtitle}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function sanitizeMetric(m: MetricEvt): MetricEvt {
  const safe = { ...m };
  const finite = (v: unknown, fb = 0) => (typeof v === "number" && Number.isFinite(v) ? v : fb);
  safe.epoch = finite(safe.epoch, 0) as number;
  safe.loss = finite(safe.loss, 0) as number;
  safe.valLoss = finite(safe.valLoss, 0) as number;
  const norm = (v: number) => {
    if (!Number.isFinite(v)) return 0;
    if (v > 1) return clamp(v, 0, 100) / 100;
    if (v < 0) return 0;
    return v;
  };
  safe.accuracy = norm(finite(safe.accuracy, 0) as number);
  safe.valAccuracy = norm(finite(safe.valAccuracy, 0) as number);
  return safe;
}
