import React from "react";
import { Link, useLocation } from "react-router-dom";

export default function TopBar() {
  const { pathname } = useLocation();

  const isActive = (path: string) =>
    pathname === path
      ? "text-white"
      : "text-slate-400 hover:text-slate-200";

  return (
    <header className="sticky top-0 z-[100] pointer-events-auto border-b border-white/10 bg-slate-950/80 backdrop-blur">
      <div className="mx-auto max-w-7xl h-14 px-4 sm:px-6 flex items-center justify-between pointer-events-auto">
        {/* Brand */}
        <div className="flex items-center gap-2">
          <span className="inline-flex h-6 w-6 items-center justify-center rounded-md bg-indigo-600 text-white text-xs font-semibold">
            C
          </span>
          <span className="font-semibold text-white">CAN IDS</span>
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-4 text-sm">
          <Link
            to="/dashboard"
            className={`px-2 py-1 rounded-md transition ${isActive("/dashboard")}`}
          >
            Dashboard
          </Link>

          <Link
            to="/training"
            className={`px-2 py-1 rounded-md transition ${isActive("/training")}`}
          >
            Training
          </Link>

          <Link
            to="/settings"
            className={`px-2 py-1 rounded-md transition ${isActive("/settings")}`}
          >
            Settings
          </Link>
        </nav>
      </div>
    </header>
  );
}
