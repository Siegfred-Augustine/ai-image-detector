import React, { useCallback, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/* CONFIG                                                              */
/* ------------------------------------------------------------------ */

// Where the FastAPI backend (serve/api.py) is running. Override at
// build time with a .env file (VITE_API_BASE=https://your-host) if
// deploying the frontend and backend separately.
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/* ------------------------------------------------------------------ */
/* TYPES                                                               */
/* ------------------------------------------------------------------ */

type PredictResult = {
  filename: string;
  label?: "Real" | "AI-generated";
  label_index?: 0 | 1;
  confidence?: number;
  probabilities?: { Real: number; "AI-generated": number };
  attention_weights?: Record<string, number> | null;
  ela_image_base64?: string | null;
  prnu_residual_image_base64?: string | null;
  wavelet_tau?: Record<string, number> | null;
  active_branches?: string[];
  checkpoint?: { path: string; epoch: number | null; seed: number | null };
  error?: string;
};

type UploadItem = {
  file: File;
  previewUrl: string;
  result?: PredictResult;
  status: "pending" | "loading" | "done" | "error";
};

/* ------------------------------------------------------------------ */
/* TOKENS (matches pages/Benchmark.tsx's design language)              */
/* ------------------------------------------------------------------ */

const C = {
  bg: "#12151A",
  surface: "#1A1E25",
  surface2: "#20252D",
  border: "#2A3038",
  borderSoft: "#232830",
  text: "#E7E9ED",
  textDim: "#9098A8",
  textFaint: "#5C6472",
  real: "#4FB3AC",
  ai: "#D89552",
  full: "#8B85D6",
  danger: "#D9645A",
  good: "#6FBE86",
};

const sans = { fontFamily: "'IBM Plex Sans', ui-sans-serif, sans-serif" };
const mono = { fontFamily: "'IBM Plex Mono', ui-monospace, monospace" };

const BRANCH_LABELS: Record<string, string> = {
  content: "Content",
  ela: "ELA",
  prnu: "PRNU",
};

const BRANCH_COLORS: Record<string, string> = {
  content: C.full,
  ela: C.ai,
  prnu: C.real,
};

/* ------------------------------------------------------------------ */
/* HELPERS                                                             */
/* ------------------------------------------------------------------ */

function pct(x: number | undefined): string {
  if (x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

/* ------------------------------------------------------------------ */
/* SUBCOMPONENTS                                                       */
/* ------------------------------------------------------------------ */

function LabelBadge({ label }: { label?: "Real" | "AI-generated" }) {
  const color = label === "AI-generated" ? C.danger : C.good;
  return (
    <span
      style={{
        ...mono,
        display: "inline-block",
        padding: "6px 14px",
        borderRadius: 6,
        fontSize: 14,
        fontWeight: 600,
        letterSpacing: 0.4,
        color,
        background: `${color}22`,
        border: `1px solid ${color}55`,
      }}
    >
      {label ?? "—"}
    </span>
  );
}

function ProbabilityBar({
  probabilities,
}: {
  probabilities?: { Real: number; "AI-generated": number };
}) {
  const real = probabilities?.Real ?? 0;
  const ai = probabilities?.["AI-generated"] ?? 0;
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 12,
          color: C.textDim,
          marginBottom: 4,
          ...mono,
        }}
      >
        <span>Real {pct(real)}</span>
        <span>AI-generated {pct(ai)}</span>
      </div>
      <div
        style={{
          display: "flex",
          height: 10,
          borderRadius: 5,
          overflow: "hidden",
          border: `1px solid ${C.border}`,
        }}
      >
        <div style={{ width: `${real * 100}%`, background: C.good }} />
        <div style={{ width: `${ai * 100}%`, background: C.danger }} />
      </div>
    </div>
  );
}

function AttentionBars({
  weights,
}: {
  weights?: Record<string, number> | null;
}) {
  if (!weights) {
    return (
      <div style={{ fontSize: 13, color: C.textFaint, ...sans }}>
        No fusion weights (single-branch model).
      </div>
    );
  }
  const entries = Object.entries(weights);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {entries.map(([branch, w]) => (
        <div key={branch}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 12,
              color: C.textDim,
              marginBottom: 3,
              ...mono,
            }}
          >
            <span>{BRANCH_LABELS[branch] ?? branch}</span>
            <span>{pct(w)}</span>
          </div>
          <div
            style={{
              height: 8,
              borderRadius: 4,
              background: C.surface2,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${Math.min(w, 1) * 100}%`,
                height: "100%",
                background: BRANCH_COLORS[branch] ?? C.full,
                borderRadius: 4,
                transition: "width 0.4s ease",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function StreamImage({
  title,
  base64,
  caption,
}: {
  title: string;
  base64?: string | null;
  caption: string;
}) {
  return (
    <div>
      <div
        style={{
          fontSize: 12,
          color: C.textDim,
          marginBottom: 6,
          ...mono,
          textTransform: "uppercase",
          letterSpacing: 0.6,
        }}
      >
        {title}
      </div>
      {base64 ? (
        <img
          src={`data:image/png;base64,${base64}`}
          alt={title}
          style={{
            width: "100%",
            borderRadius: 6,
            border: `1px solid ${C.border}`,
            display: "block",
            background: "#000",
          }}
        />
      ) : (
        <div
          style={{
            width: "100%",
            aspectRatio: "1 / 1",
            borderRadius: 6,
            border: `1px dashed ${C.border}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: C.textFaint,
            fontSize: 12,
            ...sans,
          }}
        >
          not available
        </div>
      )}
      <div style={{ fontSize: 11, color: C.textFaint, marginTop: 6, ...sans }}>
        {caption}
      </div>
    </div>
  );
}

function ResultCard({ item }: { item: UploadItem }) {
  const r = item.result;
  return (
    <div
      style={{
        background: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        padding: 20,
        display: "grid",
        gridTemplateColumns: "180px 1fr 1fr",
        gap: 20,
        alignItems: "start",
      }}
    >
      {/* Original preview + label */}
      <div>
        <img
          src={item.previewUrl}
          alt={item.file.name}
          style={{
            width: "100%",
            borderRadius: 6,
            border: `1px solid ${C.border}`,
            display: "block",
            marginBottom: 10,
          }}
        />
        <div
          style={{
            fontSize: 12,
            color: C.textFaint,
            wordBreak: "break-all",
            marginBottom: 10,
            ...mono,
          }}
        >
          {item.file.name}
        </div>

        {item.status === "loading" && (
          <div style={{ color: C.textDim, fontSize: 13, ...sans }}>
            Analyzing…
          </div>
        )}
        {item.status === "error" && (
          <div style={{ color: C.danger, fontSize: 13, ...sans }}>
            {r?.error ?? "Prediction failed."}
          </div>
        )}
        {item.status === "done" && r && !r.error && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <LabelBadge label={r.label} />
            <div style={{ fontSize: 12, color: C.textDim, ...sans }}>
              Confidence:{" "}
              <span style={{ color: C.text, ...mono }}>
                {pct(r.confidence)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Probabilities + attention */}
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <div>
          <div
            style={{
              fontSize: 12,
              color: C.textDim,
              marginBottom: 8,
              ...mono,
              textTransform: "uppercase",
              letterSpacing: 0.6,
            }}
          >
            Prediction
          </div>
          <ProbabilityBar probabilities={r?.probabilities} />
        </div>
        <div>
          <div
            style={{
              fontSize: 12,
              color: C.textDim,
              marginBottom: 8,
              ...mono,
              textTransform: "uppercase",
              letterSpacing: 0.6,
            }}
          >
            Branch attention weights
          </div>
          <AttentionBars weights={r?.attention_weights} />
        </div>
      </div>

      {/* Forensic stream visualizations */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <StreamImage
          title="ELA map"
          base64={r?.ela_image_base64}
          caption="Recompression error — brighter regions recompressed differently than the rest of the image."
        />
        <StreamImage
          title="PRNU residual"
          base64={r?.prnu_residual_image_base64}
          caption="Learned wavelet noise residual — texture the model's PRNU branch is keying on."
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* MAIN PAGE                                                           */
/* ------------------------------------------------------------------ */

export default function Detector() {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [apiStatus, setApiStatus] = useState<"unknown" | "ok" | "down">(
    "unknown",
  );
  const inputRef = useRef<HTMLInputElement>(null);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setApiStatus(data.status === "ok" ? "ok" : "down");
    } catch {
      setApiStatus("down");
    }
  }, []);

  React.useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const addFiles = useCallback((fileList: FileList | File[]) => {
    const files = Array.from(fileList).filter((f) =>
      f.type.startsWith("image/"),
    );
    const newItems: UploadItem[] = files.map((file) => ({
      file,
      previewUrl: URL.createObjectURL(file),
      status: "pending",
    }));
    setItems((prev) => [...prev, ...newItems]);
  }, []);

  const analyzeAll = useCallback(async () => {
    const pending = items.filter(
      (it) => it.status === "pending" || it.status === "error",
    );
    if (pending.length === 0) return;

    setItems((prev) =>
      prev.map((it) =>
        it.status === "pending" || it.status === "error"
          ? { ...it, status: "loading" }
          : it,
      ),
    );

    const formData = new FormData();
    pending.forEach((it) => formData.append("files", it.file, it.file.name));

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server returned ${res.status}`);
      }
      const data: { results: PredictResult[] } = await res.json();

      setItems((prev) => {
        const pendingNames = pending.map((it) => it.file.name);
        let resultIdx = 0;
        return prev.map((it) => {
          if (!pendingNames.includes(it.file.name) || it.status !== "loading")
            return it;
          const result = data.results[resultIdx++];
          return { ...it, result, status: result?.error ? "error" : "done" };
        });
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : "Request failed";
      setItems((prev) =>
        prev.map((it) =>
          it.status === "loading"
            ? {
                ...it,
                status: "error",
                result: { filename: it.file.name, error: message },
              }
            : it,
        ),
      );
    }
  }, [items]);

  const clearAll = useCallback(() => {
    items.forEach((it) => URL.revokeObjectURL(it.previewUrl));
    setItems([]);
  }, [items]);

  const hasPending = items.some(
    (it) => it.status === "pending" || it.status === "error",
  );

  return (
    <div
      style={{
        background: C.bg,
        minHeight: "100vh",
        padding: "28px 20px 60px",
        ...sans,
      }}
    >
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 24,
          }}
        >
          <div>
            <h1
              style={{
                color: C.text,
                fontSize: 24,
                fontWeight: 600,
                margin: 0,
              }}
            >
              AI Image Detector
            </h1>
            <div style={{ color: C.textDim, fontSize: 14, marginTop: 4 }}>
              Full Model — ELA + PRNU + Content, attention-weighted fusion
            </div>
          </div>
          <div
            style={{
              ...mono,
              fontSize: 12,
              padding: "4px 10px",
              borderRadius: 5,
              color:
                apiStatus === "ok"
                  ? C.good
                  : apiStatus === "down"
                    ? C.danger
                    : C.textDim,
              background:
                apiStatus === "ok"
                  ? `${C.good}18`
                  : apiStatus === "down"
                    ? `${C.danger}18`
                    : C.surface2,
              border: `1px solid ${C.border}`,
            }}
          >
            API:{" "}
            {apiStatus === "ok"
              ? "connected"
              : apiStatus === "down"
                ? "unreachable"
                : "checking…"}
          </div>
        </div>

        {apiStatus === "down" && (
          <div
            style={{
              background: `${C.danger}14`,
              border: `1px solid ${C.danger}55`,
              borderRadius: 8,
              padding: "12px 16px",
              color: C.textDim,
              fontSize: 13,
              marginBottom: 20,
            }}
          >
            Can't reach the backend at <code style={mono}>{API_BASE}</code>.
            Start it with{" "}
            <code style={mono}>uvicorn serve.api:app --reload --port 8000</code>{" "}
            from the project root (train the Full Model first if you haven't:{" "}
            <code style={mono}>
              python -m training.train --config config/full.yaml --seed 42
            </code>
            ).
          </div>
        )}

        {/* Upload dropzone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files) addFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          style={{
            border: `2px dashed ${isDragging ? C.full : C.border}`,
            borderRadius: 10,
            padding: "36px 20px",
            textAlign: "center",
            cursor: "pointer",
            background: isDragging ? `${C.full}0d` : C.surface,
            transition: "border-color 0.2s, background 0.2s",
            marginBottom: 20,
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            style={{ display: "none" }}
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
          <div style={{ color: C.text, fontSize: 15, marginBottom: 4 }}>
            Drop images here, or click to browse
          </div>
          <div style={{ color: C.textFaint, fontSize: 13 }}>
            JPG / PNG, one or more files
          </div>
        </div>

        {items.length > 0 && (
          <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
            <button
              onClick={analyzeAll}
              disabled={!hasPending || apiStatus !== "ok"}
              style={{
                ...mono,
                padding: "8px 18px",
                borderRadius: 6,
                border: "none",
                background:
                  hasPending && apiStatus === "ok" ? C.full : C.surface2,
                color: hasPending && apiStatus === "ok" ? "#fff" : C.textFaint,
                fontSize: 13,
                fontWeight: 600,
                cursor:
                  hasPending && apiStatus === "ok" ? "pointer" : "not-allowed",
              }}
            >
              Analyze{" "}
              {items.filter(
                (it) => it.status === "pending" || it.status === "error",
              ).length || ""}
            </button>
            <button
              onClick={clearAll}
              style={{
                ...mono,
                padding: "8px 18px",
                borderRadius: 6,
                border: `1px solid ${C.border}`,
                background: "transparent",
                color: C.textDim,
                fontSize: 13,
                cursor: "pointer",
              }}
            >
              Clear all
            </button>
            <button
              onClick={checkHealth}
              style={{
                ...mono,
                padding: "8px 18px",
                borderRadius: 6,
                border: `1px solid ${C.border}`,
                background: "transparent",
                color: C.textDim,
                fontSize: 13,
                cursor: "pointer",
              }}
            >
              Re-check API
            </button>
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {items.map((item, i) => (
            <ResultCard key={`${item.file.name}-${i}`} item={item} />
          ))}
        </div>

        {items.length === 0 && (
          <div
            style={{
              textAlign: "center",
              color: C.textFaint,
              fontSize: 13,
              padding: "40px 0",
            }}
          >
            No images yet — upload one to see the model's prediction,
            confidence, per-branch attention weights, and the extracted ELA /
            PRNU maps.
          </div>
        )}
      </div>
    </div>
  );
}
