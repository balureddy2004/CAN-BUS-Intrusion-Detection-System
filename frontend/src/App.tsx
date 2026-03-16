// src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import TopBar from "@/components/TopBar";
import Dashboard from "@/pages/Dashboard";
import TrainingPage from "@/pages/TrainingPage";
import SettingsPage from "@/pages/SettingsPage";
import { Toaster } from "sonner";
import "./index.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-foreground flex flex-col">
        <TopBar />

        {/* ✅ Minimal transition wrapper to smooth page switches */}
        <div className="flex-1 transition-opacity duration-150 ease-out">
          <Routes>
            {/* Default landing */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />

            {/* Pages */}
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </div>
      </div>

      <Toaster richColors position="top-right" closeButton />
    </BrowserRouter>
  );
}
