import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// Without StrictMode: the viewport owns a WebGL context that must not be created twice.
createRoot(document.getElementById("root")!).render(<App />);
