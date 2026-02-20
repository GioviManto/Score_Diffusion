import { useState, useEffect, useRef, useCallback } from "react";

// ─── Math utilities ────────────────────────────────────────────────────────────

const inv2x2 = ([[a, b], [c, d]]) => {
  const det = a * d - b * c;
  return [[(d / det), (-b / det)], [(-c / det), (a / det)]];
};

const det2x2 = ([[a, b], [c, d]]) => a * d - b * c;

const cholesky2x2 = ([[a, b], [, d]]) => {
  // L such that L L^T = [[a,b],[b,d]]
  const l00 = Math.sqrt(a);
  const l10 = b / l00;
  const l11 = Math.sqrt(Math.max(0, d - l10 * l10));
  return [[l00, 0], [l10, l11]];
};

const gaussianPdf2D = (x1, x2, mu, Sigma) => {
  const iS = inv2x2(Sigma);
  const d0 = x1 - mu[0], d1 = x2 - mu[1];
  const quad = d0 * (iS[0][0] * d0 + iS[0][1] * d1) + d1 * (iS[1][0] * d0 + iS[1][1] * d1);
  const denom = 2 * Math.PI * Math.sqrt(Math.max(0, det2x2(Sigma)));
  return Math.exp(-0.5 * quad) / denom;
};

const gaussian1D = (x, mu, sigma2) => {
  const s = Math.sqrt(sigma2);
  return Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
};

// Chi2 ppf for df=2 and mass p: r^2 = -2*ln(1-p)
const chi2ppf2 = (p) => -2 * Math.log(1 - p);

// ─── Problem definition ────────────────────────────────────────────────────────

const W = [0.55, 0.45];
const MU0 = [[1.2, 0.8], [2.2, 2.8]];
const SIGMA0 = [[[0.25, 0.18], [0.18, 0.60]], [[0.35, -0.15], [-0.15, 0.25]]];
const I2 = [[1, 0], [0, 1]];

const ouParams = (mu0k, S0k, t) => {
  const a = Math.exp(-t), b = Math.exp(-2 * t);
  const mu = [a * mu0k[0], a * mu0k[1]];
  const Sigma = [
    [b * S0k[0][0] + (1 - b) * I2[0][0], b * S0k[0][1]],
    [b * S0k[1][0], b * S0k[1][1] + (1 - b) * I2[1][1]],
  ];
  return { mu, Sigma };
};

const mixtureParams = (t) =>
  [0, 1].map(k => ouParams(MU0[k], SIGMA0[k], t));

const mixturePdf = (x1, x2, t) => {
  const comps = mixtureParams(t);
  return comps.reduce((s, { mu, Sigma }, k) => s + W[k] * gaussianPdf2D(x1, x2, mu, Sigma), 0);
};

const marginalX = (x, t) => {
  const comps = mixtureParams(t);
  return comps.reduce((s, { mu, Sigma }, k) => s + W[k] * gaussian1D(x, mu[0], Sigma[0][0]), 0);
};
const marginalY = (y, t) => {
  const comps = mixtureParams(t);
  return comps.reduce((s, { mu, Sigma }, k) => s + W[k] * gaussian1D(y, mu[1], Sigma[1][1]), 0);
};

// Ellipse points for 80% mass
const ellipsePoints = (mu, Sigma, n = 200) => {
  const r = Math.sqrt(chi2ppf2(0.80));
  const L = cholesky2x2(Sigma);
  return Array.from({ length: n + 1 }, (_, i) => {
    const a = (2 * Math.PI * i) / n;
    const cx = r * Math.cos(a), cy = r * Math.sin(a);
    return [mu[0] + L[0][0] * cx + L[0][1] * cy, mu[1] + L[1][0] * cx + L[1][1] * cy];
  });
};

// ─── Canvas renderer ───────────────────────────────────────────────────────────

const GRID = 90;   // resolution of density grid
const T_FINAL = 2.4;
const ELLIPSE_COLORS = ["#ff4d6d", "#00d4ff"];

const useDevicePixelRatio = () => {
  const [dpr, setDpr] = useState(window.devicePixelRatio || 1);
  useEffect(() => {
    const mq = window.matchMedia(`(resolution: ${dpr}dppx)`);
    const handler = () => setDpr(window.devicePixelRatio || 1);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [dpr]);
  return dpr;
};

// Pre-compute density on grid and cache ImageData
const renderDensity = (ctx, t, toCanvas, W_px, H_px) => {
  // Domain: x in [-2, 4.5], y in [-1, 4.5]
  const xmin = -2, xmax = 4.5, ymin = -1, ymax = 4.5;
  const img = ctx.createImageData(GRID, GRID);

  let maxVal = 0;
  const vals = new Float32Array(GRID * GRID);
  for (let iy = 0; iy < GRID; iy++) {
    for (let ix = 0; ix < GRID; ix++) {
      const x1 = xmin + (ix / (GRID - 1)) * (xmax - xmin);
      const x2 = ymin + (iy / (GRID - 1)) * (ymax - ymin);
      const v = mixturePdf(x1, x2, t);
      vals[iy * GRID + ix] = v;
      if (v > maxVal) maxVal = v;
    }
  }

  // Viridis-like palette interpolation
  const viridis = [
    [68, 1, 84], [72, 40, 120], [62, 83, 160], [49, 120, 167],
    [38, 154, 166], [53, 183, 121], [130, 205, 57], [230, 229, 28]
  ];
  const getColor = (v) => {
    const n = v / (maxVal + 1e-12);
    const idx = Math.min(n * (viridis.length - 1), viridis.length - 1.001);
    const lo = Math.floor(idx), hi = lo + 1;
    const f = idx - lo;
    return viridis[lo].map((c, i) => c * (1 - f) + viridis[hi][i] * f);
  };

  for (let iy = 0; iy < GRID; iy++) {
    for (let ix = 0; ix < GRID; ix++) {
      const v = vals[iy * GRID + ix];
      const [r, g, b] = getColor(v);
      const px = (iy * GRID + ix) * 4;
      img.data[px] = r; img.data[px + 1] = g; img.data[px + 2] = b;
      img.data[px + 3] = 220;
    }
  }

  // Draw scaled
  const offscreen = document.createElement("canvas");
  offscreen.width = GRID; offscreen.height = GRID;
  offscreen.getContext("2d").putImageData(img, 0, 0);

  const [cx0, cy0] = toCanvas(xmin, ymax);
  const [cx1, cy1] = toCanvas(xmax, ymin);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(offscreen, cx0, cy0, cx1 - cx0, cy1 - cy0);
};

export default function OUDiffusionViz() {
  const dpr = useDevicePixelRatio();
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const [tau, setTau] = useState(T_FINAL);
  const [playing, setPlaying] = useState(false);
  const tauRef = useRef(T_FINAL);
  const dirRef = useRef(-1); // -1 = reverse time (noise → data)

  // Canvas size
  const SIZE = 520;
  const MARG = 80; // px strip for marginals
  const W_px = SIZE, H_px = SIZE;

  // Domain
  const xmin = -2, xmax = 4.5, ymin = -1, ymax = 4.5;

  const toCanvas = useCallback((x, y) => {
    const cx = MARG + ((x - xmin) / (xmax - xmin)) * (W_px - MARG);
    const cy = (H_px - MARG) - ((y - ymin) / (ymax - ymin)) * (H_px - MARG);
    return [cx, cy];
  }, []);

  const draw = useCallback((t) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W_px * dpr, H_px * dpr);
    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = "#0a0a14";
    ctx.fillRect(0, 0, W_px, H_px);

    // Grid lines (faint)
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let gx = Math.ceil(xmin); gx <= Math.floor(xmax); gx++) {
      const [cx] = toCanvas(gx, 0);
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H_px - MARG); ctx.stroke();
    }
    for (let gy = Math.ceil(ymin); gy <= Math.floor(ymax); gy++) {
      const [, cy] = toCanvas(0, gy);
      ctx.beginPath(); ctx.moveTo(MARG, cy); ctx.lineTo(W_px, cy); ctx.stroke();
    }

    // Density heatmap
    renderDensity(ctx, t, toCanvas, W_px, H_px);

    // Ellipses
    const comps = mixtureParams(t);
    comps.forEach(({ mu, Sigma }, k) => {
      const pts = ellipsePoints(mu, Sigma);
      ctx.beginPath();
      pts.forEach(([x, y], i) => {
        const [cx, cy] = toCanvas(x, y);
        i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      });
      ctx.closePath();
      ctx.strokeStyle = ELLIPSE_COLORS[k];
      ctx.lineWidth = 2.5;
      ctx.shadowColor = ELLIPSE_COLORS[k];
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Center dot
      const [cx, cy] = toCanvas(mu[0], mu[1]);
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.fillStyle = ELLIPSE_COLORS[k];
      ctx.fill();
    });

    // Marginal: x1 (bottom strip)
    const nMarg = 120;
    const xs = Array.from({ length: nMarg }, (_, i) => xmin + (i / (nMarg - 1)) * (xmax - xmin));
    const px = xs.map(x => marginalX(x, t));
    const maxPx = Math.max(...px) + 1e-12;
    ctx.beginPath();
    xs.forEach((x, i) => {
      const [cx] = toCanvas(x, 0);
      const cy = H_px - MARG + MARG * 0.85 * (1 - px[i] / maxPx);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(255,255,255,0.4)";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Baseline for x marginal
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(MARG, H_px - MARG);
    ctx.lineTo(W_px, H_px - MARG);
    ctx.stroke();

    // Marginal: x2 (left strip)
    const ys = Array.from({ length: nMarg }, (_, i) => ymin + (i / (nMarg - 1)) * (ymax - ymin));
    const py = ys.map(y => marginalY(y, t));
    const maxPy = Math.max(...py) + 1e-12;
    ctx.beginPath();
    ys.forEach((y, i) => {
      const [, cy] = toCanvas(0, y);
      const cx = MARG - MARG * 0.85 * (1 - py[i] / maxPy);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(255,255,255,0.4)";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Baseline for y marginal
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(MARG, 0);
    ctx.lineTo(MARG, H_px - MARG);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "13px 'IBM Plex Mono', monospace";
    ctx.fillText("x₁", W_px - 20, H_px - MARG + 15);
    ctx.save();
    ctx.translate(14, 14);
    ctx.rotate(Math.PI / 2);
    ctx.fillText("x₂", 0, 0);
    ctx.restore();

    ctx.restore();
  }, [toCanvas, dpr]);

  // Setup canvas dimensions
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = W_px * dpr;
    canvas.height = H_px * dpr;
    canvas.style.width = `${W_px}px`;
    canvas.style.height = `${H_px}px`;
  }, [dpr]);

  // Draw on tau change
  useEffect(() => {
    draw(tau);
  }, [tau, draw]);

  // Animation loop
  useEffect(() => {
    if (!playing) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    const speed = 0.018;
    const step = () => {
      tauRef.current += dirRef.current * speed;
      if (tauRef.current <= 0) { tauRef.current = 0; dirRef.current = 1; }
      if (tauRef.current >= T_FINAL) { tauRef.current = T_FINAL; dirRef.current = -1; }
      setTau(tauRef.current);
      animRef.current = requestAnimationFrame(step);
    };
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing]);

  const handleSlider = (e) => {
    const v = parseFloat(e.target.value);
    tauRef.current = v;
    setTau(v);
  };

  // τ = T - t: slider shows reverse time (0=pure noise at t=T, T=data at t=0)
  // But our t IS tau here: t=T_FINAL is pure noise, t=0 is data
  const tDisplay = tau.toFixed(3);
  const phase = tau > 1.5 ? "pure noise" : tau > 0.6 ? "emerging structure" : "structured data";

  return (
    <div style={{
      background: "#0a0a14",
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "32px 16px",
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      color: "#e0e0e0",
    }}>
      {/* Google Font */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Crimson+Pro:ital,wght@0,300;1,300&display=swap');`}</style>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 28, maxWidth: 560 }}>
        <div style={{ fontSize: 11, letterSpacing: "0.25em", color: "#555", marginBottom: 6, textTransform: "uppercase" }}>
          Score-Based Generative Modeling · Analytical Lab
        </div>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: "0 0 8px", color: "#fff", letterSpacing: "-0.01em" }}>
          OU Forward Diffusion on a Gaussian Mixture
        </h1>
        <p style={{ fontSize: 13, color: "#666", margin: 0, fontFamily: "'Crimson Pro', serif", fontStyle: "italic", fontWeight: 300 }}>
          Reverse-time view: noise collapses into multimodal structure as τ → 0
        </p>
      </div>

      {/* Canvas */}
      <div style={{
        position: "relative",
        borderRadius: 4,
        overflow: "hidden",
        boxShadow: "0 0 60px rgba(0,212,255,0.08), 0 0 0 1px rgba(255,255,255,0.06)",
      }}>
        <canvas ref={canvasRef} />

        {/* τ overlay */}
        <div style={{
          position: "absolute", top: 12, right: 14,
          fontSize: 12, color: "rgba(255,255,255,0.45)",
          background: "rgba(0,0,0,0.5)", padding: "4px 10px", borderRadius: 3,
          letterSpacing: "0.05em",
        }}>
          t = {tDisplay} &nbsp;·&nbsp; <span style={{ color: tau > 1.5 ? "#ff4d6d" : tau > 0.6 ? "#ffd166" : "#00d4ff" }}>{phase}</span>
        </div>
      </div>

      {/* Math box */}
      <div style={{
        marginTop: 20,
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: 4,
        padding: "14px 20px",
        maxWidth: 560,
        width: "100%",
        fontSize: 12,
        color: "#888",
        lineHeight: 1.9,
      }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 24px" }}>
          {[
            ["Forward SDE", "X_t = e⁻ᵗ X₀ + √(1−e⁻²ᵗ) Z"],
            ["Mean decay", "μₖ(t) = e⁻ᵗ μₖ(0)"],
            ["Cov. evolution", "Σₖ(t) = e⁻²ᵗ Σₖ(0) + (1−e⁻²ᵗ) I"],
            ["Marginal", "p_t(x) = Σₖ wₖ 𝒩(x; μₖ(t), Σₖ(t))"],
          ].map(([label, eq]) => (
            <div key={label}>
              <span style={{ color: "#555", fontSize: 10, display: "block", textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</span>
              <span style={{ color: "#aaa", fontFamily: "monospace" }}>{eq}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div style={{ marginTop: 18, display: "flex", alignItems: "center", gap: 16, maxWidth: 560, width: "100%" }}>
        <button
          onClick={() => setPlaying(p => !p)}
          style={{
            background: playing ? "rgba(255,77,109,0.15)" : "rgba(0,212,255,0.12)",
            border: `1px solid ${playing ? "rgba(255,77,109,0.4)" : "rgba(0,212,255,0.3)"}`,
            color: playing ? "#ff4d6d" : "#00d4ff",
            padding: "8px 20px",
            borderRadius: 3,
            cursor: "pointer",
            fontSize: 12,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            fontFamily: "inherit",
            whiteSpace: "nowrap",
          }}
        >
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>

        <input
          type="range"
          min={0}
          max={T_FINAL}
          step={0.01}
          value={tau}
          onChange={handleSlider}
          style={{ flex: 1, accentColor: "#00d4ff", cursor: "pointer" }}
        />

        <div style={{ fontSize: 11, color: "#555", whiteSpace: "nowrap" }}>
          <span style={{ color: "#777" }}>t =</span> {tDisplay}
        </div>
      </div>

      {/* Legend */}
      <div style={{ marginTop: 14, display: "flex", gap: 24, fontSize: 11, color: "#555" }}>
        {[["#ff4d6d", "Component 1  (w=0.55)"], ["#00d4ff", "Component 2  (w=0.45)"], ["rgba(255,255,255,0.6)", "Marginals p(x₁), p(x₂)"]].map(([color, label]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 20, height: 2, background: color, borderRadius: 1 }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
