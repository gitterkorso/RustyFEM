import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// Utan StrictMode: modellfönstret äger en WebGL-kontext som inte ska skapas två gånger.
createRoot(document.getElementById("root")!).render(<App />);
