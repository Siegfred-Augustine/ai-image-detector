import { useState } from "react";
import ForensicBenchmark from "./pages/Benchmark";
import Detector from "./pages/Detector";
import "./App.css";

type Tab = "detector" | "benchmark";

function App() {
  const [tab, setTab] = useState<Tab>("detector");

  const tabButtonStyle = (active: boolean): React.CSSProperties => ({
    fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
    padding: "10px 20px",
    border: "none",
    borderBottom: active ? "2px solid #8B85D6" : "2px solid transparent",
    background: "transparent",
    color: active ? "#E7E9ED" : "#5C6472",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    letterSpacing: 0.4,
  });

  return (
    <div style={{ minHeight: "100vh", background: "#12151A" }}>
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid #2A3038",
          background: "#1A1E25",
          padding: "0 20px",
        }}
      >
        <button
          style={tabButtonStyle(tab === "detector")}
          onClick={() => setTab("detector")}
        >
          DETECTOR
        </button>
        <button
          style={tabButtonStyle(tab === "benchmark")}
          onClick={() => setTab("benchmark")}
        >
          BENCHMARK
        </button>
      </div>
      {tab === "detector" ? <Detector /> : <ForensicBenchmark />}
    </div>
  );
}

export default App;
