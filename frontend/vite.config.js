// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },

  server: {
    port: 5173,
    proxy: {
      // REST APIs - forward /api/* to backend at 127.0.0.1:5000
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
        secure: false,
        rewrite: (p) => p.replace(/^\/api/, "/api"),
      },

      // Training control endpoints (if you have routes starting with /train)
      "/train": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
        secure: false,
      },

      // WebSocket proxy - forwards ws://localhost:5173/ws/* -> ws://127.0.0.1:5000/ws/*
      "/ws": {
        target: "ws://127.0.0.1:5000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },

      // Optional: other websocket route prefixes
      "/socket": {
        target: "ws://127.0.0.1:5000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
