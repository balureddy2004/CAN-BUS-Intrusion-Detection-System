// src/components/ReplayControls.tsx
import React, { useEffect, useState } from "react";

type ReplayState = {
  running: boolean;
  speed: number;
  current_index?: number;
  total_frames?: number;
};

export default function ReplayControls() {
  // 🔹 Backend-reported state
  const [state, setState] = useState<ReplayState>({
    running: false,
    speed: 1,
  });

  // 🔹 UI-controlled values (IMPORTANT)
  const [uiSpeed, setUiSpeed] = useState<number>(1);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState<string>("demo");
  const [loading, setLoading] = useState(false);

  // ---------------------------
  // Fetch available datasets
  // ---------------------------
  async function fetchDatasets() {
    try {
      const res = await fetch("/api/replay/datasets");
      if (!res.ok) return;
      const data = await res.json();
      if (data?.datasets?.length) {
        setDatasets(data.datasets);
        if (!data.datasets.includes(dataset)) {
          setDataset(data.datasets[0]);
        }
      }
    } catch {}
  }

  // ---------------------------
  // Fetch backend replay state
  // ---------------------------
  async function fetchState() {
    try {
      const res = await fetch("/api/replay/state");
      if (!res.ok) return;
      const data = await res.json();
      if (data?.state) {
        setState(data.state);

        // 🔹 Only sync speed AFTER replay starts
        if (data.state.running) {
          setUiSpeed(data.state.speed);
        }
      }
    } catch {}
  }

  useEffect(() => {
    fetchDatasets();
    fetchState();
    const t = setInterval(fetchState, 1000);
    return () => clearInterval(t);
  }, []);

  // ---------------------------
  async function startReplay() {
    setLoading(true);
    await fetch("/api/replay/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        speed: uiSpeed, // ✅ use UI speed
      }),
    });
    setLoading(false);
    fetchState();
  }

  async function stopReplay() {
    await fetch("/api/replay/stop", { method: "POST" });
    fetchState();
  }

  async function applySpeed(speed: number) {
    setUiSpeed(speed);

    // Only tell backend if replay is running
    if (state.running) {
      await fetch("/api/replay/speed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed }),
      });
    }
  }

  const progress =
    typeof state.current_index === "number" &&
    typeof state.total_frames === "number" &&
    state.total_frames > 0
      ? Math.round((state.current_index / state.total_frames) * 100)
      : null;

  // ---------------------------
  return (
    <div className="p-4 rounded-2xl bg-white/5 shadow-sm">
      <h3 className="text-lg font-semibold text-center mb-3">
        ▶ CAN Replay Engine
      </h3>

      {/* Status */}
      <div className="text-center text-sm mb-2">
        Status:{" "}
        <b className={state.running ? "text-green-600" : "text-red-600"}>
          {state.running ? "RUNNING" : "STOPPED"}
        </b>{" "}
        · Speed: <b>{uiSpeed}×</b>
      </div>

      {/* Dataset selector */}
      <div className="flex justify-center mb-3">
        <select
          value={dataset}
          disabled={state.running}
          onChange={(e) => setDataset(e.target.value)}
          className="border rounded px-2 py-1 text-sm min-w-[220px]"
        >
          {datasets.map((ds) => (
            <option key={ds} value={ds}>
              {ds}
            </option>
          ))}
        </select>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3 mb-3">
        <button
          onClick={startReplay}
          disabled={loading || state.running}
          className="px-3 py-1 rounded bg-green-600 text-white text-sm disabled:opacity-50"
        >
          ▶ Start
        </button>

        <button
          onClick={stopReplay}
          disabled={!state.running}
          className="px-3 py-1 rounded bg-red-600 text-white text-sm disabled:opacity-50"
        >
          ⏹ Stop
        </button>
      </div>

      {/* Speed */}
      <div className="flex justify-center gap-2 text-xs mb-3">
        {[1, 5, 10, 50, 100, 1000, 5000].map((s) => (
          <button
            key={s}
            onClick={() => applySpeed(s)}
            className={`px-2 py-1 rounded border ${
              uiSpeed === s ? "bg-black text-white" : ""
            }`}
          >
            {s}×
          </button>
        ))}
      </div>

      {/* Progress */}
      <div className="mt-3">
        <div className="text-xs text-center mb-1">Replay Progress</div>

        {progress === null ? (
          <div className="text-xs text-center text-muted-foreground">—</div>
        ) : (
          <>
            <div className="w-full h-2 bg-gray-200 rounded overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-xs text-center mt-1 text-muted-foreground">
              {state.current_index} / {state.total_frames} frames ({progress}%)
            </div>
          </>
        )}
      </div>
    </div>
  );
}
