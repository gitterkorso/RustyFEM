import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri förväntar sig en fast port och vill se Rust-felen i terminalen.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5173, strictPort: true, watch: { ignored: ["**/src-tauri/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: { target: "es2022", sourcemap: false, chunkSizeWarningLimit: 1500 },
});
