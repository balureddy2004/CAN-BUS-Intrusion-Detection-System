// src/pages/SettingsPage.tsx
import React from "react";
import EmailAlertsToggle from "@/components/EmailAlertsToggle";

export default function SettingsPage() {
  return (
    <div className="relative w-full">
      <main className="relative z-0 p-6 w-full max-w-7xl mx-auto space-y-14">

        <section className="space-y-3 max-w-3xl">
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-sm text-muted-foreground">
            Configure alerting behavior, notifications, and system preferences
            for the CAN IDS platform.
          </p>

          <div className="rounded-2xl border bg-gradient-to-b from-slate-950/60 to-slate-900/40 backdrop-blur p-4 shadow-xl">
            <EmailAlertsToggle />
          </div>
        </section>

      </main>
    </div>
  );
}
