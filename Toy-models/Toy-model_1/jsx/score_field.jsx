import { useState, useEffect, useRef, useCallback } from "react";

// ─── Math ──────────────────────────────────────────────────────────────────────
const inv2x2 = ([[a, b], [c, d]]) => {
  const det = a * d - b * c + 1e-12;
  return [[d / det, -b / det], [-c / det, a / det]];
};
const det2x2 = ([[a, b], [c, d]]) => a * d - b * c;
const chol2 = ([[a, b], [, d]]) => {
  const l00 = Math.sqrt(Math.max(a, 1e-8));
  const l10 = b / l00;
  const l11 = Math.sqrt(Math.max(d - l10 * l10, 1e-8));
  return [[l00, 0], [l10, l11]];
};

const gaussian2D = (x1, x2, mu, Sigma) => {
  const iS = inv2x2(Sigma);
  const d0 = x1 - mu[0], d1 = x2 - mu[1];
  const q = d0 * (iS[0][0] * d0 + iS[0][1] * d1) + d1 * (iS[1][0] * d0 + iS[1][1] * d1);
  const det = det2x2(Sigma); if (det <= 0) return 0;
  return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
};

// ∇ log N(x; mu, Sigma) = -Sigma^{-1} (x - mu)
const scoreGaussian = (x1, x2, mu, Sigma) => {
  const iS = inv2x2(Sigma);
  const d0 = x1 - mu[0], d1 = x2 - mu[1];
  return [-(iS[0][0] * d0 + iS[0][1] * d1), -(iS[1][0] * d0 + iS[1][1] * d1)];
};

const chi2ppf2 = (p) => -2 * Math.log(1 - p + 1e-12);

const ouParams = (mu0, S0, t) => {
  const a = Math.exp(-t), b = Math.exp(-2 * t);
  return {
    mu: [a * mu0[0], a * mu0[1]],
    Sigma: [[b * S0[0][0] + (1 - b), b * S0[0][1]], [b * S0[1][0], b * S0[1][1] + (1 - b)]],
  };
};

// ─── Mixture definition ────────────────────────────────────────────────────────
const COMPONENTS_DEFAULT = [
  { w: 0.55, mu: [1.2, 0.8], Sigma: [[0.25, 0.18], [0.18, 0.60]] },
  { w: 0.45, mu: [2.2, 2.8], Sigma: [[0.35, -0.15], [-0.15, 0.25]] },
];

// Score of mixture: ∇ log p_t(x) = Σ_k [w_k p_k(x) / p(x)] ∇ log p_k(x)
// This is the "posterior-weighted average" of component scores
const mixturePdfAndScore = (x1, x2, comps, t) => {
  let ptotal = 0, s0 = 0, s1 = 0;
  const cps = comps.map(({ mu: mu0, Sigma: S0, w }) => {
    const { mu, Sigma } = ouParams(mu0, S0, t);
    const p = w * gaussian2D(x1, x2, mu, Sigma);
    const [g0, g1] = scoreGaussian(x1, x2, mu, Sigma);
    return { p, g0, g1 };
  });
  const totalW = comps.reduce((s, c) => s + c.w, 0);
  cps.forEach(({ p, g0, g1 }) => {
    ptotal += p / totalW;
    s0 += p * g0;
    s1 += p * g1;
  });
  if (ptotal < 1e-15) return { p: 0, score: [0, 0] };
  return { p: ptotal, score: [s0 / ptotal, s1 / ptotal] };
};

const ellipsePoints = (mu, Sigma, mass = 0.8, n = 160) => {
  const r = Math.sqrt(chi2ppf2(mass));
  const L = chol2(Sigma);
  return Array.from({ length: n + 1 }, (_, i) => {
    const a = (2 * Math.PI * i) / n;
    const cx = r * Math.cos(a), cy = r * Math.sin(a);
    return [mu[0] + L[0][0] * cx, mu[1] + L[1][0] * cx + L[1][1] * cy];
  });
};

const ELLIPSE_COLORS = ["#ff4d6d", "#00d4ff"];

// ─── Plasma-like palette for density ──────────────────────────────────────────
const PLASMA = [
  [13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137],
  [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]
];
const plasmaColor = (v) => {
  const idx = Math.min(v * (PLASMA.length - 1), PLASMA.length - 1.001);
  const lo = Math.floor(idx), hi = lo + 1, f = idx - lo;
  return PLASMA[lo].map((c, i) => Math.round(c * (1 - f) + PLASMA[hi][i] * f));
};

const T_MAX = 2.5;
const DENSITY_GRID = 70;
const ARROW_GRID = 18; // number of arrows per axis

export default function ScoreFieldViz() {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const tRef = useRef(0.3);
  const dirRef = useRef(1);
  const [t, setT] = useState(0.3);
  const [playing, setPlaying] = useState(false);
  const [showDensity, setShowDensity] = useState(true);
  const [showEllipses, setShowEllipses] = useState(true);
  const [showScore, setShowScore] = useState(true);
  const [showStreamlines, setShowStreamlines] = useState(false);
  const [components] = useState(COMPONENTS_DEFAULT);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  const SIZE = 520;
  const xmin = -2.5, xmax = 5.5, ymin = -2.5, ymax = 5.5;

  const toCanvas = useCallback((x, y) => [
    ((x - xmin) / (xmax - xmin)) * SIZE,
    SIZE - ((y - ymin) / (ymax - ymin)) * SIZE,
  ], []);

  const draw = useCallback((t) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, SIZE * dpr, SIZE * dpr);
    ctx.save();
    ctx.scale(dpr, dpr);

    // BG
    ctx.fillStyle = "#050508";
    ctx.fillRect(0, 0, SIZE, SIZE);

    // Subtle grid
    ctx.strokeStyle = "rgba(255,255,255,0.035)";
    ctx.lineWidth = 0.5;
    for (let gx = Math.ceil(xmin); gx <= Math.floor(xmax); gx++) {
      const [cx] = toCanvas(gx, 0); ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, SIZE); ctx.stroke();
    }
    for (let gy = Math.ceil(ymin); gy <= Math.floor(ymax); gy++) {
      const [, cy] = toCanvas(0, gy); ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(SIZE, cy); ctx.stroke();
    }

    // Density heatmap
    if (showDensity) {
      const vals = new Float32Array(DENSITY_GRID * DENSITY_GRID);
      let maxV = 0;
      for (let iy = 0; iy < DENSITY_GRID; iy++) {
        for (let ix = 0; ix < DENSITY_GRID; ix++) {
          const x1 = xmin + (ix / (DENSITY_GRID - 1)) * (xmax - xmin);
          const x2 = ymin + (iy / (DENSITY_GRID - 1)) * (ymax - ymin);
          const { p } = mixturePdfAndScore(x1, x2, components, t);
          vals[iy * DENSITY_GRID + ix] = p;
          if (p > maxV) maxV = p;
        }
      }
      const img = new ImageData(DENSITY_GRID, DENSITY_GRID);
      for (let i = 0; i < DENSITY_GRID * DENSITY_GRID; i++) {
        const v = vals[i] / (maxV + 1e-12);
        const [r, g, b] = plasmaColor(v);
        img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = Math.round(180 * Math.pow(v, 0.5));
      }
      const off = document.createElement("canvas"); off.width = DENSITY_GRID; off.height = DENSITY_GRID;
      off.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.drawImage(off, 0, 0, SIZE, SIZE);
    }

    // Streamlines (reverse-time trajectories through the score field)
    if (showStreamlines) {
      const nLines = 20;
      const steps = 60;
      const dt = 0.04;
      ctx.lineWidth = 0.8;
      for (let li = 0; li < nLines; li++) {
        // Start from a noisy region
        const angle = (li / nLines) * 2 * Math.PI;
        const radius = 1.5 + Math.sin(li * 1.7) * 0.8;
        let px = 1.8 + radius * Math.cos(angle);
        let py = 1.8 + radius * Math.sin(angle);

        ctx.beginPath();
        const [sx0, sy0] = toCanvas(px, py);
        ctx.moveTo(sx0, sy0);

        let alpha = 0.7;
        for (let step = 0; step < steps; step++) {
          const { score: [dx, dy], p } = mixturePdfAndScore(px, py, components, t);
          if (p < 1e-8) break;
          px += dx * dt;
          py += dy * dt;
          const [cx, cy] = toCanvas(px, py);
          ctx.lineTo(cx, cy);
        }
        ctx.strokeStyle = `rgba(180,230,255,0.18)`;
        ctx.stroke();
      }
    }

    // Score arrows
    if (showScore) {
      const cellX = (xmax - xmin) / ARROW_GRID;
      const cellY = (ymax - ymin) / ARROW_GRID;
      let maxMag = 0;
      const arrows = [];

      for (let iy = 0; iy <= ARROW_GRID; iy++) {
        for (let ix = 0; ix <= ARROW_GRID; ix++) {
          const x1 = xmin + ix * cellX;
          const x2 = ymin + iy * cellY;
          const { p, score: [s0, s1] } = mixturePdfAndScore(x1, x2, components, t);
          const mag = Math.sqrt(s0 * s0 + s1 * s1);
          if (mag > maxMag) maxMag = mag;
          arrows.push({ x1, x2, s0, s1, mag, p });
        }
      }

      const arrowScale = 0.28 * Math.min(cellX, cellY) / (maxMag + 1e-12) * ARROW_GRID;

      arrows.forEach(({ x1, x2, s0, s1, mag, p }) => {
        if (mag < 1e-4) return;
        const normMag = mag / (maxMag + 1e-12);
        const [cx, cy] = toCanvas(x1, x2);
        const ex = x1 + s0 * arrowScale;
        const ey = x2 + s1 * arrowScale;
        const [ecx, ecy] = toCanvas(ex, ey);

        // Color: direction-based (hue = atan2) + brightness = magnitude
        const hue = ((Math.atan2(s1, s0) / Math.PI) * 180 + 360) % 360;
        const lightness = 40 + normMag * 50;
        const alpha = 0.3 + normMag * 0.65;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ecx, ecy);
        ctx.strokeStyle = `hsla(${hue}, 80%, ${lightness}%, ${alpha})`;
        ctx.lineWidth = 0.8 + normMag * 1.2;
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(ecy - cy, ecx - cx);
        const headLen = 3 + normMag * 4;
        ctx.beginPath();
        ctx.moveTo(ecx, ecy);
        ctx.lineTo(ecx - headLen * Math.cos(angle - 0.4), ecy - headLen * Math.sin(angle - 0.4));
        ctx.lineTo(ecx - headLen * Math.cos(angle + 0.4), ecy - headLen * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = `hsla(${hue}, 80%, ${lightness}%, ${alpha})`;
        ctx.fill();
      });
    }

    // Ellipses
    if (showEllipses) {
      components.forEach(({ mu: mu0, Sigma: S0 }, k) => {
        const { mu, Sigma } = ouParams(mu0, S0, t);
        const pts = ellipsePoints(mu, Sigma);
        ctx.beginPath();
        pts.forEach(([x, y], i) => {
          const [cx, cy] = toCanvas(x, y);
          i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
        });
        ctx.closePath();
        ctx.strokeStyle = ELLIPSE_COLORS[k];
        ctx.lineWidth = 2;
        ctx.shadowColor = ELLIPSE_COLORS[k]; ctx.shadowBlur = 10;
        ctx.stroke(); ctx.shadowBlur = 0;

        const [mx, my] = toCanvas(mu[0], mu[1]);
        ctx.beginPath(); ctx.arc(mx, my, 4, 0, Math.PI * 2);
        ctx.fillStyle = ELLIPSE_COLORS[k]; ctx.fill();
      });
    }

    // Axes
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 0.8;
    const [ox, oy] = toCanvas(0, 0);
    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(SIZE, oy); ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = `11px "IBM Plex Mono", monospace`;
    ctx.fillText("x₁", SIZE - 18, oy - 8);
    ctx.fillText("x₂", ox + 8, 14);

    ctx.restore();
  }, [toCanvas, components, showDensity, showEllipses, showScore, showStreamlines, dpr]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = SIZE * dpr; canvas.height = SIZE * dpr;
    canvas.style.width = `${SIZE}px`; canvas.style.height = `${SIZE}px`;
  }, [dpr]);

  useEffect(() => { draw(t); }, [t, draw]);

  useEffect(() => {
    if (!playing) { cancelAnimationFrame(animRef.current); return; }
    const step = () => {
      tRef.current += dirRef.current * 0.018;
      if (tRef.current <= 0) { tRef.current = 0; dirRef.current = 1; }
      if (tRef.current >= T_MAX) { tRef.current = T_MAX; dirRef.current = -1; }
      setT(tRef.current);
      animRef.current = requestAnimationFrame(step);
    };
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing]);

  const Toggle = ({ label, value, onChange, color = "#00d4ff" }) => (
    <button onClick={() => onChange(!value)} style={{
      background: value ? `rgba(${color === "#ff4d6d" ? "255,77,109" : color === "#ffd166" ? "255,209,102" : "0,212,255"},0.12)` : "rgba(255,255,255,0.03)",
      border: `1px solid ${value ? color + "55" : "rgba(255,255,255,0.08)"}`,
      color: value ? color : "#555", padding: "5px 12px", borderRadius: 3,
      cursor: "pointer", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase",
      fontFamily: "IBM Plex Mono, monospace", transition: "all 0.15s",
    }}>{label}</button>
  );

  const scoreExplainer = t < 0.4
    ? "Strong, sharp arrows — score points firmly toward dense modes"
    : t < 1.2
    ? "Mixed regime — arrows weigh both components, slight ambiguity"
    : "Score ≈ −x — Gaussian limit, arrows point toward origin";

  return (
    <div style={{
      background: "#050508", minHeight: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", padding: "28px 12px",
      fontFamily: "'IBM Plex Mono', monospace", color: "#e0e0e0",
    }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&display=swap');`}</style>

      <div style={{ fontSize: 10, letterSpacing: "0.25em", color: "#444", marginBottom: 6, textTransform: "uppercase" }}>
        Score-Based Generative Modeling · Key Concept
      </div>
      <h1 style={{ fontSize: 18, fontWeight: 600, margin: "0 0 4px", color: "#fff" }}>
        Score Function Visualizer
      </h1>
      <div style={{ fontSize: 12, color: "#555", marginBottom: 20, fontStyle: "italic" }}>
        ∇<sub>x</sub> log p<sub>t</sub>(x) — the field that drives reverse diffusion
      </div>

      <div style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
        {/* Canvas */}
        <div>
          <div style={{
            position: "relative", borderRadius: 4, overflow: "hidden",
            boxShadow: "0 0 60px rgba(180,50,255,0.07), 0 0 0 1px rgba(255,255,255,0.05)",
          }}>
            <canvas ref={canvasRef} />
            <div style={{
              position: "absolute", top: 10, left: 10, right: 10,
              fontSize: 10, color: "rgba(255,255,255,0.4)",
              background: "rgba(0,0,0,0.55)", padding: "5px 10px", borderRadius: 3,
            }}>
              t = {t.toFixed(3)} · {scoreExplainer}
            </div>
          </div>

          {/* Time */}
          <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 12, width: SIZE }}>
            <button onClick={() => setPlaying(p => !p)} style={{
              background: playing ? "rgba(255,77,109,0.12)" : "rgba(180,100,255,0.1)",
              border: `1px solid ${playing ? "rgba(255,77,109,0.35)" : "rgba(180,100,255,0.3)"}`,
              color: playing ? "#ff4d6d" : "#c47cff", padding: "7px 16px", borderRadius: 3,
              cursor: "pointer", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase",
              fontFamily: "inherit", whiteSpace: "nowrap",
            }}>{playing ? "⏸ Pause" : "▶ Play"}</button>
            <input type="range" min={0} max={T_MAX} step={0.01} value={t}
              onChange={e => { const v = parseFloat(e.target.value); tRef.current = v; setT(v); }}
              style={{ flex: 1, accentColor: "#c47cff", cursor: "pointer" }} />
            <span style={{ fontSize: 11, color: "#555", minWidth: 32 }}>t={t.toFixed(2)}</span>
          </div>

          {/* Toggles */}
          <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Toggle label="Density" value={showDensity} onChange={setShowDensity} color="#ffd166" />
            <Toggle label="Ellipses" value={showEllipses} onChange={setShowEllipses} color="#00d4ff" />
            <Toggle label="Score ∇log p" value={showScore} onChange={setShowScore} color="#c47cff" />
            <Toggle label="Streamlines" value={showStreamlines} onChange={setShowStreamlines} color="#7fff6a" />
          </div>
        </div>

        {/* Theory panel */}
        <div style={{
          width: 240, background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6,
          padding: "16px 14px", fontSize: 11, lineHeight: 1.8, color: "#666",
        }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 10 }}>
            What you're seeing
          </div>

          <div style={{ marginBottom: 14 }}>
            <span style={{ color: "#c47cff" }}>Arrows</span> = score field s(x,t) = ∇ log p_t(x)
            <br />Each arrow shows which direction probability density <em>increases</em> from that point.
          </div>

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 12, marginBottom: 14 }}>
            <div style={{ color: "#888", marginBottom: 6 }}>For a Gaussian mixture:</div>
            <div style={{ color: "#aaa", fontFamily: "monospace", fontSize: 10 }}>
              s(x,t) = Σ_k r_k(x,t) · s_k(x,t)
            </div>
            <div style={{ fontSize: 10, color: "#555", marginTop: 6 }}>
              where r_k = posterior responsibility of component k, and s_k = −Σ_k⁻¹(x−μ_k) is each component's score.
            </div>
          </div>

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 12, marginBottom: 14 }}>
            <div style={{ color: "#888", marginBottom: 6 }}>Arrow color = direction</div>
            <div style={{ fontSize: 10, color: "#555" }}>
              Hue encodes the angle of the score vector (like a phase map). Brightness = magnitude.
            </div>
          </div>

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 12, marginBottom: 14 }}>
            <div style={{ color: "#888", marginBottom: 6 }}>At t → ∞:</div>
            <div style={{ fontSize: 10, color: "#555" }}>
              p_t(x) → N(0,I), so s(x,t) → −x. Arrows point radially inward from all directions toward origin.
            </div>
          </div>

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 12 }}>
            <div style={{ color: "#888", marginBottom: 6 }}>Reverse SDE:</div>
            <div style={{ color: "#aaa", fontFamily: "monospace", fontSize: 10 }}>
              dX = [X + 2s(X,t)]dt + √2 dW̄
            </div>
            <div style={{ fontSize: 10, color: "#555", marginTop: 6 }}>
              The score is exactly what the reverse-time SDE uses to denoise. Streamlines trace these paths.
            </div>
          </div>

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 12, marginTop: 4 }}>
            <div style={{ fontSize: 9, color: "#444", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>Live values</div>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: "#555" }}>
              e⁻ᵗ = {Math.exp(-t).toFixed(4)}<br />
              e⁻²ᵗ = {Math.exp(-2 * t).toFixed(4)}<br />
              noise = {(1 - Math.exp(-2 * t)).toFixed(4)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
