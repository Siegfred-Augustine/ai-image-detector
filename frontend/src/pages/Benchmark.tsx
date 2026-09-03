import React, { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

/* ------------------------------------------------------------------ */
/* TYPES                                                              */
/* ------------------------------------------------------------------ */

type RunResult = {
  seed: number;
  precision: number;
  recall: number;
  f1: number;
};

type ConfusionMatrix = {
  tp: number;
  fn: number;
  fp: number;
  tn: number;
};

type ModelGroup = "flagship" | "sop1" | "ablation";

type ModelData = {
  id: string;
  name: string;
  sub: string;
  group: ModelGroup;
  color: string;
  runs: RunResult[];
  confusion: ConfusionMatrix;
  pVsFull: number | null;
  mcnemarVsFull: number | null;
};

type ChipProps = {
  children: React.ReactNode;
  color: string;
  dim?: boolean;
};

type CMCellProps = {
  label: string;
  value: number;
  total: number;
  color: string;
  note: string;
};

/* ------------------------------------------------------------------ */
/* TOKENS                                                             */
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

const mono = {
  fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
};

const sans = {
  fontFamily: "'IBM Plex Sans', ui-sans-serif, sans-serif",
};

/* ------------------------------------------------------------------ */
/* MOCK DATA                                                          */
/* Temporary demo data — replace with real evaluation results later. */
/* ------------------------------------------------------------------ */

const MODEL_DATA: ModelData[] = [
  {
    id: "full",
    name: "Full Model",
    sub: "ELA + PRNU + Content, attention fusion",
    group: "flagship",
    color: C.full,
    runs: [
      { seed: 1, precision: 0.941, recall: 0.937, f1: 0.939 },
      { seed: 2, precision: 0.935, recall: 0.944, f1: 0.939 },
      { seed: 3, precision: 0.948, recall: 0.929, f1: 0.938 },
      { seed: 4, precision: 0.939, recall: 0.941, f1: 0.940 },
      { seed: 5, precision: 0.933, recall: 0.938, f1: 0.935 },
    ],
    confusion: {
      tp: 1408,
      fn: 92,
      fp: 87,
      tn: 1413,
    },
    pVsFull: null,
    mcnemarVsFull: null,
  },

  {
    id: "prnu_only",
    name: "PRNU-only CNN",
    sub: "SOP 1a — wavelet residual branch alone",
    group: "sop1",
    color: C.ai,
    runs: [
      { seed: 1, precision: 0.881, recall: 0.902, f1: 0.891 },
      { seed: 2, precision: 0.874, recall: 0.896, f1: 0.885 },
      { seed: 3, precision: 0.889, recall: 0.887, f1: 0.888 },
      { seed: 4, precision: 0.877, recall: 0.905, f1: 0.891 },
      { seed: 5, precision: 0.883, recall: 0.891, f1: 0.887 },
    ],
    confusion: {
      tp: 1350,
      fn: 150,
      fp: 178,
      tn: 1322,
    },
    pVsFull: 0.0021,
    mcnemarVsFull: 0.008,
  },

  {
    id: "ela_only",
    name: "ELA-only CNN",
    sub: "SOP 1b — compression-artifact branch alone",
    group: "sop1",
    color: C.ai,
    runs: [
      { seed: 1, precision: 0.858, recall: 0.849, f1: 0.853 },
      { seed: 2, precision: 0.851, recall: 0.861, f1: 0.856 },
      { seed: 3, precision: 0.863, recall: 0.844, f1: 0.853 },
      { seed: 4, precision: 0.847, recall: 0.858, f1: 0.852 },
      { seed: 5, precision: 0.855, recall: 0.852, f1: 0.853 },
    ],
    confusion: {
      tp: 1281,
      fn: 219,
      fp: 226,
      tn: 1274,
    },
    pVsFull: 0.0006,
    mcnemarVsFull: 0.001,
  },

  {
    id: "content_only",
    name: "Content-only CNN",
    sub: "SOP 1c — structural / GAN-artifact branch alone",
    group: "sop1",
    color: C.ai,
    runs: [
      { seed: 1, precision: 0.902, recall: 0.911, f1: 0.906 },
      { seed: 2, precision: 0.897, recall: 0.905, f1: 0.901 },
      { seed: 3, precision: 0.908, recall: 0.898, f1: 0.903 },
      { seed: 4, precision: 0.895, recall: 0.913, f1: 0.904 },
      { seed: 5, precision: 0.900, recall: 0.902, f1: 0.901 },
    ],
    confusion: {
      tp: 1367,
      fn: 133,
      fp: 150,
      tn: 1350,
    },
    pVsFull: 0.014,
    mcnemarVsFull: 0.041,
  },

  {
    id: "no_prnu",
    name: "Full Model − PRNU",
    sub: "SOP 2 — ELA + Content only",
    group: "ablation",
    color: C.danger,
    runs: [
      { seed: 1, precision: 0.903, recall: 0.888, f1: 0.895 },
      { seed: 2, precision: 0.897, recall: 0.895, f1: 0.896 },
      { seed: 3, precision: 0.911, recall: 0.879, f1: 0.895 },
      { seed: 4, precision: 0.894, recall: 0.901, f1: 0.897 },
      { seed: 5, precision: 0.900, recall: 0.884, f1: 0.892 },
    ],
    confusion: {
      tp: 1332,
      fn: 168,
      fp: 145,
      tn: 1355,
    },
    pVsFull: 0.0009,
    mcnemarVsFull: 0.003,
  },

  {
    id: "no_ela",
    name: "Full Model − ELA",
    sub: "SOP 3 — PRNU + Content only",
    group: "ablation",
    color: C.good,
    runs: [
      { seed: 1, precision: 0.928, recall: 0.921, f1: 0.924 },
      { seed: 2, precision: 0.922, recall: 0.930, f1: 0.926 },
      { seed: 3, precision: 0.933, recall: 0.917, f1: 0.925 },
      { seed: 4, precision: 0.919, recall: 0.928, f1: 0.923 },
      { seed: 5, precision: 0.925, recall: 0.922, f1: 0.923 },
    ],
    confusion: {
      tp: 1391,
      fn: 109,
      fp: 105,
      tn: 1395,
    },
    pVsFull: 0.061,
    mcnemarVsFull: 0.12,
  },

  {
    id: "no_content",
    name: "Full Model − Content",
    sub: "SOP 4 — ELA + PRNU only",
    group: "ablation",
    color: C.danger,
    runs: [
      { seed: 1, precision: 0.912, recall: 0.905, f1: 0.908 },
      { seed: 2, precision: 0.906, recall: 0.913, f1: 0.909 },
      { seed: 3, precision: 0.918, recall: 0.899, f1: 0.908 },
      { seed: 4, precision: 0.903, recall: 0.916, f1: 0.909 },
      { seed: 5, precision: 0.909, recall: 0.907, f1: 0.908 },
    ],
    confusion: {
      tp: 1364,
      fn: 136,
      fp: 128,
      tn: 1372,
    },
    pVsFull: 0.003,
    mcnemarVsFull: 0.017,
  },
];

/* ------------------------------------------------------------------ */
/* GROUPS                                                             */
/* ------------------------------------------------------------------ */

const GROUPS: { id: ModelGroup; label: string }[] = [
  {
    id: "flagship",
    label: "Flagship",
  },
  {
    id: "sop1",
    label: "SOP 1 — Single-stream",
  },
  {
    id: "ablation",
    label: "SOP 2–4 — Component ablation",
  },
];

/* ------------------------------------------------------------------ */
/* STAT HELPERS                                                       */
/* ------------------------------------------------------------------ */

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;

  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[]): number {
  if (arr.length <= 1) return 0;

  const m = mean(arr);

  return Math.sqrt(
    arr.reduce((sum, value) => sum + (value - m) ** 2, 0) /
      (arr.length - 1)
  );
}

function metricStats(
  model: ModelData,
  key: "precision" | "recall" | "f1"
) {
  const vals = model.runs.map((run) => run[key]);

  return {
    mean: mean(vals),
    std: std(vals),
  };
}

function fmtPct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

function fmtPVal(p: number | null | undefined): string {
  if (p === null || p === undefined) {
    return "—";
  }

  return p < 0.001 ? "p<0.001" : `p=${p.toFixed(3)}`;
}

/* ------------------------------------------------------------------ */
/* PRIMITIVES                                                         */
/* ------------------------------------------------------------------ */

function Chip({ children, color, dim }: ChipProps) {
  return (
    <span
      style={{
        ...mono,
        fontSize: 11,
        letterSpacing: "0.02em",
        padding: "3px 8px",
        borderRadius: 4,
        border: `1px solid ${
          dim ? C.borderSoft : `${color}55`
        }`,
        color: dim ? C.textFaint : color,
        background: dim ? "transparent" : `${color}14`,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
}

function SigBadge({ p }: { p: number | null }) {
  if (p === null || p === undefined) {
    return (
      <Chip color={C.textFaint} dim>
        reference
      </Chip>
    );
  }

  const sig = p < 0.05;

  return (
    <Chip color={sig ? C.danger : C.textDim}>
      {sig ? "significant · " : "n.s. · "}
      {fmtPVal(p)}
    </Chip>
  );
}

/* ------------------------------------------------------------------ */
/* MAIN APP                                                           */
/* ------------------------------------------------------------------ */

export default function ForensicBenchmark() {
  const [activeGroup, setActiveGroup] =
    useState<ModelGroup>("sop1");

  const [inspectId, setInspectId] =
    useState<string>("full");

  const visibleModels = useMemo(() => {
    if (activeGroup === "sop1") {
      return MODEL_DATA.filter(
        (model) =>
          model.group === "flagship" ||
          model.group === "sop1"
      );
    }

    if (activeGroup === "ablation") {
      return MODEL_DATA.filter(
        (model) =>
          model.group === "flagship" ||
          model.group === "ablation"
      );
    }

    return MODEL_DATA.filter(
      (model) => model.group === "flagship"
    );
  }, [activeGroup]);

  const chartData = visibleModels.map((model) => {
    const f1 = metricStats(model, "f1");

    return {
      name: model.name,
      f1Mean: Number((f1.mean * 100).toFixed(2)),
      f1Std: Number((f1.std * 100).toFixed(2)),
      color: model.color,
      id: model.id,
    };
  });

  const inspectModel =
    MODEL_DATA.find((model) => model.id === inspectId) ??
    MODEL_DATA[0];

  const cm = inspectModel.confusion;

  const total =
    cm.tp +
    cm.fn +
    cm.fp +
    cm.tn;

  return (
    <div
      style={{
        ...sans,
        background: C.bg,
        color: C.text,
        minHeight: "100%",
        padding: "28px 20px 60px",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        * {
          box-sizing: border-box;
        }

        ::selection {
          background: ${C.full}55;
        }

        table {
          border-collapse: collapse;
          width: 100%;
        }

        th,
        td {
          text-align: left;
          padding: 10px 12px;
        }

        tbody tr:hover {
          background: ${C.surface2};
        }

        .scroll-x {
          overflow-x: auto;
        }

        button,
        select {
          font-family: inherit;
        }
      `}</style>

      <div
        style={{
          maxWidth: 1080,
          margin: "0 auto",
        }}
      >
        {/* ---------------- HEADER ---------------- */}

        <div style={{ marginBottom: 28 }}>
          <div
            style={{
              ...mono,
              fontSize: 12,
              color: C.textFaint,
              marginBottom: 8,
              letterSpacing: "0.02em",
            }}
          >
            AI-FACE FORENSICS · MODEL COMPARISON
          </div>

          <h1
            style={{
              fontSize: 28,
              fontWeight: 600,
              margin: 0,
              lineHeight: 1.25,
            }}
          >
            Multi-stream CNN benchmark
          </h1>

          <p
            style={{
              color: C.textDim,
              marginTop: 8,
              maxWidth: 620,
              lineHeight: 1.55,
              fontSize: 14,
            }}
          >
            Comparing the full attention-fused model against
            single-stream and branch-ablated variants defined
            in the handoff spec, on the FFHQ / AI-Face test
            split (1,500 real + 1,500 AI-generated, 5 runs
            per model).
          </p>
        </div>

        {/* ---------------- MOCK DATA NOTICE ---------------- */}

        <div
          style={{
            background: C.surface,
            border: `1px solid ${C.borderSoft}`,
            borderLeft: `3px solid ${C.ai}`,
            borderRadius: 6,
            padding: "12px 16px",
            marginBottom: 28,
            fontSize: 13,
            color: C.textDim,
            lineHeight: 1.5,
          }}
        >
          <strong style={{ color: C.text }}>
            Placeholder data.
          </strong>{" "}
          All numbers below are mock values so the dashboard
          has something to render — swap in real results by
          editing the{" "}
          <code
            style={{
              ...mono,
              color: C.ai,
            }}
          >
            MODEL_DATA
          </code>{" "}
          array at the top of the file.
          <br />
          Several hyperparameters (patch size, branch filter
          counts, learning rate, batch size, FC dimensions)
          are marked <em>not specified</em> in the source spec
          and still need to be fixed before real runs can
          produce these numbers.
        </div>

        {/* ---------------- GROUP TABS ---------------- */}

        <div
          style={{
            display: "flex",
            gap: 4,
            marginBottom: 20,
            borderBottom: `1px solid ${C.borderSoft}`,
            overflowX: "auto",
          }}
        >
          {GROUPS.filter(
            (group) => group.id !== "flagship"
          ).map((group) => {
            const active =
              activeGroup === group.id;

            return (
              <button
                key={group.id}
                onClick={() =>
                  setActiveGroup(group.id)
                }
                style={{
                  ...sans,
                  background: "transparent",
                  border: "none",
                  borderBottom: active
                    ? `2px solid ${C.full}`
                    : "2px solid transparent",
                  color: active
                    ? C.text
                    : C.textDim,
                  fontSize: 14,
                  fontWeight: 500,
                  padding: "10px 4px",
                  marginRight: 24,
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                {group.label}
              </button>
            );
          })}
        </div>

        {/* ---------------- COMPARISON TABLE ---------------- */}

        <div
          className="scroll-x"
          style={{ marginBottom: 32 }}
        >
          <table style={{ fontSize: 13.5 }}>
            <thead>
              <tr
                style={{
                  borderBottom: `1px solid ${C.border}`,
                  color: C.textFaint,
                  ...mono,
                  fontSize: 11,
                }}
              >
                <th>MODEL</th>
                <th>PRECISION</th>
                <th>RECALL</th>
                <th>F1</th>
                <th>PAIRED T-TEST</th>
                <th>MCNEMAR</th>
              </tr>
            </thead>

            <tbody>
              {visibleModels.map((model) => {
                const precision = metricStats(
                  model,
                  "precision"
                );

                const recall = metricStats(
                  model,
                  "recall"
                );

                const f1 = metricStats(
                  model,
                  "f1"
                );

                return (
                  <tr
                    key={model.id}
                    onClick={() =>
                      setInspectId(model.id)
                    }
                    style={{
                      borderBottom: `1px solid ${C.borderSoft}`,
                      cursor: "pointer",
                      background:
                        inspectId === model.id
                          ? C.surface2
                          : "transparent",
                    }}
                  >
                    <td>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                        }}
                      >
                        <span
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: 2,
                            background: model.color,
                            flexShrink: 0,
                          }}
                        />

                        <div>
                          <div
                            style={{
                              fontWeight: 500,
                            }}
                          >
                            {model.name}
                          </div>

                          <div
                            style={{
                              fontSize: 11.5,
                              color: C.textFaint,
                            }}
                          >
                            {model.sub}
                          </div>
                        </div>
                      </div>
                    </td>

                    <td style={mono}>
                      {fmtPct(precision.mean)}{" "}
                      <span
                        style={{
                          color: C.textFaint,
                        }}
                      >
                        ±{fmtPct(precision.std)}
                      </span>
                    </td>

                    <td style={mono}>
                      {fmtPct(recall.mean)}{" "}
                      <span
                        style={{
                          color: C.textFaint,
                        }}
                      >
                        ±{fmtPct(recall.std)}
                      </span>
                    </td>

                    <td
                      style={{
                        ...mono,
                        fontWeight: 600,
                      }}
                    >
                      {fmtPct(f1.mean)}
                    </td>

                    <td>
                      <SigBadge
                        p={model.pVsFull}
                      />
                    </td>

                    <td>
                      <SigBadge
                        p={model.mcnemarVsFull}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* ---------------- F1 CHART ---------------- */}

        <div
          style={{
            background: C.surface,
            border: `1px solid ${C.borderSoft}`,
            borderRadius: 8,
            padding: "20px 20px 8px",
            marginBottom: 32,
          }}
        >
          <div
            style={{
              fontSize: 13,
              fontWeight: 600,
              marginBottom: 4,
            }}
          >
            Mean F1-score by model
          </div>

          <div
            style={{
              fontSize: 12,
              color: C.textFaint,
              marginBottom: 12,
            }}
          >
            Averaged across 5 seeded runs · dashed line marks
            the Full Model's mean
          </div>

          <ResponsiveContainer
            width="100%"
            height={280}
          >
            <BarChart
              data={chartData}
              margin={{
                top: 8,
                right: 12,
                left: -12,
                bottom: 8,
              }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={C.borderSoft}
                vertical={false}
              />

              <XAxis
                dataKey="name"
                tick={{
                  fill: C.textFaint,
                  fontSize: 11,
                }}
                interval={0}
                angle={-18}
                textAnchor="end"
                height={70}
                axisLine={{
                  stroke: C.borderSoft,
                }}
                tickLine={false}
              />

              <YAxis
                domain={[75, 100]}
                tick={{
                  fill: C.textFaint,
                  fontSize: 11,
                }}
                axisLine={{
                  stroke: C.borderSoft,
                }}
                tickLine={false}
                unit="%"
              />

              <Tooltip
                cursor={{
                  fill: C.surface2,
                }}
                contentStyle={{
                  background: C.surface2,
                  border: `1px solid ${C.border}`,
                  borderRadius: 6,
                  fontSize: 12,
                }}
                labelStyle={{
                  color: C.text,
                }}
                formatter={(value, key) => {
                  if (key === "f1Mean") {
                    return [
                      `${value}%`,
                      "F1",
                    ];
                  }

                  return [value, key];
                }}
              />

              <ReferenceLine
                y={
                  chartData.find(
                    (data) => data.id === "full"
                  )?.f1Mean
                }
                stroke={C.full}
                strokeDasharray="4 4"
              />

              <Bar
                dataKey="f1Mean"
                radius={[
                  4,
                  4,
                  0,
                  0,
                ]}
              >
                {chartData.map((data) => (
                  <Cell
                    key={data.id}
                    fill={data.color}
                    fillOpacity={
                      data.id === inspectId
                        ? 1
                        : 0.65
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* ---------------- CONFUSION MATRIX ---------------- */}

        <div
          style={{
            background: C.surface,
            border: `1px solid ${C.borderSoft}`,
            borderRadius: 8,
            padding: 20,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-start",
              flexWrap: "wrap",
              gap: 12,
              marginBottom: 16,
            }}
          >
            <div>
              <div
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                Confusion matrix —{" "}
                {inspectModel.name}
              </div>

              <div
                style={{
                  fontSize: 12,
                  color: C.textFaint,
                }}
              >
                Pooled over{" "}
                {total.toLocaleString()} test images ·
                click a row above to inspect a different
                model
              </div>
            </div>

            <select
              value={inspectId}
              onChange={(event) =>
                setInspectId(
                  event.target.value
                )
              }
              style={{
                background: C.surface2,
                color: C.text,
                border: `1px solid ${C.border}`,
                borderRadius: 6,
                padding: "6px 10px",
                fontSize: 12.5,
                ...sans,
              }}
            >
              {MODEL_DATA.map((model) => (
                <option
                  key={model.id}
                  value={model.id}
                >
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "minmax(0,1fr) minmax(0,1fr)",
              gap: 10,
              maxWidth: 460,
            }}
          >
            <CMCell
              label="TP — AI → AI"
              value={cm.tp}
              total={total}
              color={C.good}
              note="correctly flagged"
            />

            <CMCell
              label="FP — Real → AI"
              value={cm.fp}
              total={total}
              color={C.textDim}
              note="false alarm"
            />

            <CMCell
              label="FN — AI → Real"
              value={cm.fn}
              total={total}
              color={C.danger}
              note="dangerous: AI accepted as real"
            />

            <CMCell
              label="TN — Real → Real"
              value={cm.tn}
              total={total}
              color={C.good}
              note="correctly cleared"
            />
          </div>

          <div
            style={{
              marginTop: 16,
              fontSize: 12,
              color: C.textFaint,
              lineHeight: 1.5,
            }}
          >
            The spec flags{" "}
            <strong
              style={{
                color: C.danger,
              }}
            >
              false negatives
            </strong>{" "}
            (an AI-generated image accepted as real) as the
            highest-stakes forensic failure mode — worth
            watching even when overall F1 looks competitive.
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* CONFUSION MATRIX CELL                                              */
/* ------------------------------------------------------------------ */

function CMCell({
  label,
  value,
  total,
  color,
  note,
}: CMCellProps) {
  const pct =
    total > 0
      ? ((value / total) * 100).toFixed(1)
      : "0.0";

  return (
    <div
      style={{
        background: C.surface2,
        border: `1px solid ${C.borderSoft}`,
        borderRadius: 6,
        padding: "14px 16px",
      }}
    >
      <div
        style={{
          ...mono,
          fontSize: 11,
          color: C.textFaint,
          marginBottom: 6,
        }}
      >
        {label}
      </div>

      <div
        style={{
          ...mono,
          fontSize: 24,
          fontWeight: 600,
          color,
        }}
      >
        {value.toLocaleString()}
      </div>

      <div
        style={{
          fontSize: 11.5,
          color: C.textFaint,
          marginTop: 4,
        }}
      >
        {pct}% of set · {note}
      </div>
    </div>
  );
}
