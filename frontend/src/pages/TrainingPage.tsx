// src/pages/TrainingPage.tsx
import React from "react";
import TrainingCard from "@/components/TrainingCard";
import LiveMonitorCard from "@/components/LiveMonitorCard";

const api = (p: string) => (p.startsWith("/") ? p : `/${p}`);

export default function TrainingPage() {
  return (
    <div className="relative w-full">
      <main className="relative z-0 p-6 w-full max-w-7xl mx-auto space-y-14">

        {/* ================= Model Training ================= */}
        <section className="space-y-3">
          <h1 className="text-2xl font-bold">Model Training</h1>
          <p className="text-sm text-muted-foreground max-w-3xl">
            Train intrusion detection models using CAN traffic datasets and
            monitor training progress in real time.
          </p>

          <div className="ids-panel p-4">
            <TrainingCard
              startUrl={api("train/start")}
              pauseUrl={api("train/pause")}
              stopUrl={api("train/stop")}
              resetUrl={api("train/reset")}
              downloadModelUrl={api("train/model")}
              wsUrl="/ws/train"
            />
          </div>
        </section>

        {/* ================= Live Training Monitor ================= */}
        <section className="space-y-3">
          <h2 className="text-xl font-semibold">Live Training Monitor</h2>
          <p className="text-sm text-muted-foreground max-w-3xl">
            Stream live training metrics such as loss, accuracy, and validation
            performance over WebSocket.
          </p>

          <div className="ids-panel p-4">
            <LiveMonitorCard
              wsUrl="/ws/train"
              statusPollUrl={api("train/status")}
            />
          </div>
        </section>

      </main>
    </div>
  );
}
