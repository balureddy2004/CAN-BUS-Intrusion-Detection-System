// src/components/TrainControls.tsx
import React, { useEffect, useState } from "react";

export default function TrainControls({ onStart }: { onStart?: (params: any) => void }) {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [dataset, setDataset] = useState<string>("");
  const [model, setModel] = useState<string>("RandomForest");
  const [epochs, setEpochs] = useState<number>(10);
  const [batchSize, setBatchSize] = useState<number>(256);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetch("/api/train/datasets")
      .then((r) => r.json())
      .then((j) => {
        // server returns { datasets: [{filename,...}, ...] } in some cases,
        // normalize to an array of strings
        if (Array.isArray(j.datasets)) {
          const names = j.datasets.map((d: any) => d.filename || d).filter(Boolean);
          setDatasets(names);
          if (!dataset && names.length) setDataset(names[0]);
        } else if (Array.isArray(j)) {
          setDatasets(j.map((x: any) => (typeof x === "string" ? x : x.filename)));
        }
      })
      .catch(() => { /* ignore */ });
  }, []);

  const start = async () => {
    setRunning(true);
    const payload = {
      model,
      epochs,
      batchSize,
      dataset: dataset || "ocs-lab",
    };
    try {
      const res = await fetch("/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const j = await res.json();
      console.log("train/start ->", j);
      if (onStart) onStart(payload);
    } catch (e) {
      console.error("start failed", e);
      alert("Failed to start training, check server logs.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="p-2 space-y-2">
      <div className="flex gap-2 items-center">
        <label className="text-sm">Dataset</label>
        <select value={dataset} onChange={(e) => setDataset(e.target.value)} className="border rounded p-1">
          <option value="ocs-lab">ocs-lab (preset)</option>
          {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
      </div>

      <div className="flex gap-2 items-center">
        <label className="text-sm">Model</label>
        <select value={model} onChange={(e) => setModel(e.target.value)} className="border rounded p-1">
          <option value="RandomForest">RandomForest</option>
          <option value="XGBoost" disabled>XGBoost (coming)</option>
          <option value="LightGBM" disabled>LightGBM (coming)</option>
        </select>
      </div>

      <div className="flex gap-2 items-center">
        <label className="text-sm">Epochs</label>
        <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} className="w-20 border rounded p-1" />
        <label className="text-sm">Batch</label>
        <input type="number" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} className="w-24 border rounded p-1" />
      </div>

      <div>
        <button onClick={start} disabled={running} className="px-3 py-1 rounded bg-blue-600 text-white">
          {running ? "Starting..." : "Start Training"}
        </button>
      </div>
    </div>
  );
}
