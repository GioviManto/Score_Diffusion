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
  const det = det2x2(Sigma);
  if (det <= 0) return 0;
  return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
};
const gaussian1D = (x, mu, v) => Math.exp(-0.5 * ((x - mu) ** 2) / v) / Math.sqrt(2 * Math.PI * v);
const chi2ppf2 = (p) => -2 * Math.log(1 - p + 1e-12);

const ouParams = (mu0, S0, t) => {
  const a = Math.exp(-t), b = Math.exp(-2 * t);
  return {
    mu: [a * mu0[0], a * mu0[1]],
    Sigma: [[b * S0[0][0] + (1 - b), b * S0[0][1]], [b * S0[1][0], b * S0[1][1] + (1 - b)]],
  };
};

const buildMixtureParams = (params, t) =>
  params.components.map(c => ouParams(c.mu, c.Sigma, t));

const mixturePdf = (x1, x2, params, t) => {
  const comps = buildMixtureParams(params, t);
  return comps.reduce((s, { mu, Sigma }, k) =>
    s + params.components[k].w * gaussian2D(x1, x2, mu, Sigma), 0);
};

const ellipsePoints = (mu, Sigma, mass = 0.8, n = 180) => {
  const r = Math.sqrt(chi2ppf2(mass));
  const L = chol2(Sigma);
  return Array.from({ length: n + 1 }, (_, i) => {
    const a = (2 * Math.PI * i) / n;
    const cx = r * Math.cos(a), cy = r * Math.sin(a);
    return [mu[0] + L[0][0] * cx, mu[1] + L[1][0] * cx + L[1][1] * cy];
  });
};

// ─── Viridis palette ───────────────────────────────────────────────────────────
const VIRIDIS = [
  [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142],
  [33, 144, 141], [53, 183, 121], [143, 211, 74], [252, 231, 37]
];
const viridisColor = (v) => {
  const idx = Math.min(v * (VIRIDIS.length - 1), VIRIDIS.length - 1.001);
  const lo = Math.floor(idx), hi = lo + 1, f = idx - lo;
  return VIRIDIS[lo].map((c, i) => Math.round(c * (1 - f) + VIRIDIS[hi][i] * f));
};

const ELLIPSE_COLORS = ["#ff4d6d", "#00d4ff", "#7fff6a", "#ffb347"];
const COMP_BG = ["rgba(255,77,109,0.08)", "rgba(0,212,255,0.08)", "rgba(127,255,106,0.08)", "rgba(255,179,71,0.08)"];

// ─── Slider component ──────────────────────────────────────────────────────────
const Sl = ({ label, value, min, max, step, onChange, color = "#00d4ff", fmt = (v) => v.toFixed(2) }) => (
  <div style={{ marginBottom: 7 }}>
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#666", marginBottom: 3, fontFamily: "IBM Plex Mono, monospace" }}>
      <span style={{ color: "#888" }}>{label}</span>
      <span style={{ color }}>{fmt(value)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      style={{ width: "100%", accentColor: color, cursor: "pointer", height: 3 }}
    />
  </div>
);

// ─── Default state ─────────────────────────────────────────────────────────────
const DEFAULT = {
  components: [
    { w: 0.55, mu: [1.2, 0.8], Sigma: [[0.25, 0.18], [0.18, 0.60]] },
    { w: 0.45, mu: [2.2, 2.8], Sigma: [[0.35, -0.15], [-0.15, 0.25]] },
  ]
};

const GRID = 80;
const T_MAX = 2.5;

export default function OUDiffusionLab() {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const tRef = useRef(T_MAX);
  const dirRef = useRef(-1);
  const [t, setT] = useState(T_MAX);
  const [playing, setPlaying] = useState(false);
  const [params, setParams] = useState(DEFAULT);
  const [activeComp, setActiveComp] = useState(0);
  const [numComps, setNumComps] = useState(2);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  const SIZE = 480;
  const MARG = 72;
  const xmin = -3, xmax = 5, ymin = -2.5, ymax = 5.5;

  const toCanvas = useCallback((x, y) => [
    MARG + ((x - xmin) / (xmax - xmin)) * (SIZE - MARG),
    (SIZE - MARG) - ((y - ymin) / (ymax - ymin)) * (SIZE - MARG),
  ], []);

  const draw = useCallback((t) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, SIZE * dpr, SIZE * dpr);
    ctx.save();
    ctx.scale(dpr, dpr);

    // BG
    ctx.fillStyle = "#080810";
    ctx.fillRect(0, 0, SIZE, SIZE);

    // Clip density region
    ctx.save();
    ctx.beginPath();
    ctx.rect(MARG, 0, SIZE - MARG, SIZE - MARG);
    ctx.clip();

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 0.5;
    for (let gx = Math.ceil(xmin); gx <= Math.floor(xmax); gx++) {
      const [cx] = toCanvas(gx, 0); ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, SIZE - MARG); ctx.stroke();
    }
    for (let gy = Math.ceil(ymin); gy <= Math.floor(ymax); gy++) {
      const [, cy] = toCanvas(0, gy); ctx.beginPath(); ctx.moveTo(MARG, cy); ctx.lineTo(SIZE, cy); ctx.stroke();
    }

    // Density heatmap
    const usedComps = params.components.slice(0, numComps);
    const visParams = { components: usedComps };
    const totalW = usedComps.reduce((s, c) => s + c.w, 0);
    const normParams = {
      components: usedComps.map(c => ({ ...c, w: c.w / totalW }))
    };

    const vals = new Float32Array(GRID * GRID);
    let maxV = 0;
    for (let iy = 0; iy < GRID; iy++) {
      for (let ix = 0; ix < GRID; ix++) {
        const x1 = xmin + (ix / (GRID - 1)) * (xmax - xmin);
        const x2 = ymin + (iy / (GRID - 1)) * (ymax - ymin);
        const v = mixturePdf(x1, x2, normParams, t);
        vals[iy * GRID + ix] = v;
        if (v > maxV) maxV = v;
      }
    }
    const img = new ImageData(GRID, GRID);
    for (let i = 0; i < GRID * GRID; i++) {
      const [r, g, b] = viridisColor(vals[i] / (maxV + 1e-12));
      img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 210;
    }
    const off = document.createElement("canvas"); off.width = GRID; off.height = GRID;
    off.getContext("2d").putImageData(img, 0, 0);
    const [cx0, cy0] = toCanvas(xmin, ymax);
    const [cx1, cy1] = toCanvas(xmax, ymin);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, cx0, cy0, cx1 - cx0, cy1 - cy0);

    // Ellipses
    normParams.components.forEach(({ mu: mu0, Sigma: Sigma0, w }, k) => {
      const { mu, Sigma } = ouParams(mu0, Sigma0, t);
      const pts = ellipsePoints(mu, Sigma);
      ctx.beginPath();
      pts.forEach(([x, y], i) => {
        const [cx, cy] = toCanvas(x, y);
        i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      });
      ctx.closePath();
      const col = ELLIPSE_COLORS[k];
      ctx.strokeStyle = col;
      ctx.lineWidth = k === activeComp ? 3 : 1.8;
      ctx.setLineDash(k === activeComp ? [] : []);
      ctx.shadowColor = col; ctx.shadowBlur = k === activeComp ? 12 : 4;
      ctx.stroke(); ctx.shadowBlur = 0;

      // Mean dot
      const [mx, my] = toCanvas(mu[0], mu[1]);
      ctx.beginPath(); ctx.arc(mx, my, k === activeComp ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.fill();

      // Weight label
      const [ox0] = toCanvas(mu0[0], mu0[1]);
      const [, oy0] = toCanvas(mu0[0], mu0[1]);
      // small w label near center
      ctx.font = `bold 10px "IBM Plex Mono", monospace`;
      ctx.fillStyle = col;
      ctx.fillText(`w=${(w / totalW).toFixed(2)}`, mx + 7, my - 5);
    });

    ctx.restore();

    // Marginal x1 (bottom)
    const nM = 100;
    const xs = Array.from({ length: nM }, (_, i) => xmin + (i / (nM - 1)) * (xmax - xmin));
    const px = xs.map(x => normParams.components.reduce((s, { mu: mu0, Sigma: Sigma0, w }, k) => {
      const { mu, Sigma } = ouParams(mu0, Sigma0, t);
      return s + w * gaussian1D(x, mu[0], Sigma[0][0]);
    }, 0));
    const maxPx = Math.max(...px) + 1e-12;
    ctx.beginPath();
    xs.forEach((x, i) => {
      const [cx] = toCanvas(x, 0);
      const cy = SIZE - MARG + (MARG - 6) * (1 - px[i] / maxPx);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 1.8;
    ctx.shadowColor = "white"; ctx.shadowBlur = 5; ctx.stroke(); ctx.shadowBlur = 0;

    // Baseline
    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(MARG, SIZE - MARG); ctx.lineTo(SIZE, SIZE - MARG); ctx.stroke();

    // Marginal x2 (left)
    const ys = Array.from({ length: nM }, (_, i) => ymin + (i / (nM - 1)) * (ymax - ymin));
    const py = ys.map(y => normParams.components.reduce((s, { mu: mu0, Sigma: Sigma0, w }, k) => {
      const { mu, Sigma } = ouParams(mu0, Sigma0, t);
      return s + w * gaussian1D(y, mu[1], Sigma[1][1]);
    }, 0));
    const maxPy = Math.max(...py) + 1e-12;
    ctx.beginPath();
    ys.forEach((y, i) => {
      const [, cy] = toCanvas(0, y);
      const cx = MARG - (MARG - 6) * (1 - py[i] / maxPy);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 1.8;
    ctx.shadowColor = "white"; ctx.shadowBlur = 5; ctx.stroke(); ctx.shadowBlur = 0;

    ctx.strokeStyle = "rgba(255,255,255,0.15)"; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(MARG, 0); ctx.lineTo(MARG, SIZE - MARG); ctx.stroke();

    // Axis labels
    ctx.fillStyle = "rgba(255,255,255,0.4)"; ctx.font = `12px "IBM Plex Mono", monospace`;
    ctx.fillText("x₁", SIZE - 18, SIZE - MARG + 16);
    ctx.save(); ctx.translate(12, 12); ctx.rotate(Math.PI / 2);
    ctx.fillText("x₂", 0, 0); ctx.restore();

    // Border
    ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.lineWidth = 1;
    ctx.strokeRect(MARG, 0, SIZE - MARG, SIZE - MARG);

    ctx.restore();
  }, [toCanvas, params, numComps, activeComp, dpr]);

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
      tRef.current += dirRef.current * 0.022;
      if (tRef.current <= 0) { tRef.current = 0; dirRef.current = 1; }
      if (tRef.current >= T_MAX) { tRef.current = T_MAX; dirRef.current = -1; }
      setT(tRef.current);
      animRef.current = requestAnimationFrame(step);
    };
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing]);

  const setComp = (k, fn) => setParams(p => {
    const comps = p.components.map((c, i) => i === k ? fn(c) : c);
    return { components: comps };
  });

  const c = params.components[activeComp];
  const totalW = params.components.slice(0, numComps).reduce((s, c) => s + c.w, 0);

  // Covariance: Sigma = [[sx^2, rho*sx*sy],[rho*sx*sy, sy^2]]
  // We parameterize via sx, sy, rho
  const sx = Math.sqrt(c.Sigma[0][0]);
  const sy = Math.sqrt(c.Sigma[1][1]);
  const rho = c.Sigma[0][1] / (sx * sy + 1e-12);

  const setSigmaFromRho = (newRho) => {
    const off = newRho * sx * sy;
    setComp(activeComp, c => ({ ...c, Sigma: [[c.Sigma[0][0], off], [off, c.Sigma[1][1]]] }));
  };
  const setSx = (v) => {
    const off = rho * v * sy;
    setComp(activeComp, c => ({ ...c, Sigma: [[v * v, off], [off, c.Sigma[1][1]]] }));
  };
  const setSy = (v) => {
    const off = rho * sx * v;
    setComp(activeComp, c => ({ ...c, Sigma: [[c.Sigma[0][0], off], [off, v * v]] }));
  };

  const phase = t > 1.8 ? "pure noise" : t > 0.7 ? "transition" : "structured";
  const phaseColor = t > 1.8 ? "#ff4d6d" : t > 0.7 ? "#ffd166" : "#00d4ff";

  const panelW = 240;

  return (
    <div style={{
      background: "#080810", minHeight: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", padding: "24px 12px",
      fontFamily: "'IBM Plex Mono', monospace", color: "#e0e0e0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&display=swap');
        input[type=range]{height:3px;border-radius:2px}
        .comp-tab{transition:all 0.15s;cursor:pointer;padding:6px 12px;border-radius:3px;font-size:11px;font-family:'IBM Plex Mono',monospace;border:1px solid transparent;}
        .comp-tab:hover{opacity:0.9}
      `}</style>

      <div style={{ fontSize: 10, letterSpacing: "0.25em", color: "#444", marginBottom: 6, textTransform: "uppercase" }}>
        OU Diffusion · Parameter Lab
      </div>
      <h1 style={{ fontSize: 18, fontWeight: 600, margin: "0 0 20px", color: "#fff", letterSpacing: "-0.01em" }}>
        Gaussian Mixture Diffusion Explorer
      </h1>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>

        {/* Canvas */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{
            position: "relative", borderRadius: 4, overflow: "hidden",
            boxShadow: "0 0 50px rgba(0,212,255,0.06), 0 0 0 1px rgba(255,255,255,0.05)",
          }}>
            <canvas ref={canvasRef} />
            <div style={{
              position: "absolute", top: 10, right: 10, fontSize: 11,
              background: "rgba(0,0,0,0.6)", padding: "3px 10px", borderRadius: 3,
              color: "rgba(255,255,255,0.4)",
            }}>
              t = {t.toFixed(3)} · <span style={{ color: phaseColor }}>{phase}</span>
            </div>
          </div>

          {/* Time control */}
          <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 12, width: SIZE }}>
            <button onClick={() => setPlaying(p => !p)} style={{
              background: playing ? "rgba(255,77,109,0.12)" : "rgba(0,212,255,0.1)",
              border: `1px solid ${playing ? "rgba(255,77,109,0.35)" : "rgba(0,212,255,0.25)"}`,
              color: playing ? "#ff4d6d" : "#00d4ff", padding: "7px 16px", borderRadius: 3,
              cursor: "pointer", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase",
              fontFamily: "inherit", whiteSpace: "nowrap",
            }}>{playing ? "⏸ Pause" : "▶ Play"}</button>
            <input type="range" min={0} max={T_MAX} step={0.01} value={t}
              onChange={e => { const v = parseFloat(e.target.value); tRef.current = v; setT(v); }}
              style={{ flex: 1, accentColor: "#00d4ff", cursor: "pointer" }} />
            <span style={{ fontSize: 11, color: "#555", minWidth: 32 }}>t={t.toFixed(2)}</span>
          </div>

          {/* Legend */}
          <div style={{ marginTop: 8, display: "flex", gap: 16, fontSize: 10, color: "#555" }}>
            {params.components.slice(0, numComps).map((_, k) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 16, height: 2, background: ELLIPSE_COLORS[k], borderRadius: 1 }} />
                <span style={{ color: ELLIPSE_COLORS[k] }}>Comp {k + 1}</span>
              </div>
            ))}
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <div style={{ width: 16, height: 2, background: "rgba(255,255,255,0.6)", borderRadius: 1 }} />
              <span>Marginals</span>
            </div>
          </div>
        </div>

        {/* Panel */}
        <div style={{
          width: panelW, background: "rgba(255,255,255,0.025)",
          border: "1px solid rgba(255,255,255,0.07)", borderRadius: 6,
          padding: "16px 14px", display: "flex", flexDirection: "column", gap: 12,
        }}>
          {/* Num components */}
          <div>
            <div style={{ fontSize: 9, color: "#555", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 8 }}>Mixture</div>
            <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
              {[1, 2, 3, 4].map(n => (
                <button key={n} onClick={() => { setNumComps(n); if (activeComp >= n) setActiveComp(n - 1); }}
                  style={{
                    flex: 1, padding: "5px 0", background: numComps === n ? "rgba(0,212,255,0.15)" : "rgba(255,255,255,0.04)",
                    border: `1px solid ${numComps === n ? "rgba(0,212,255,0.4)" : "rgba(255,255,255,0.08)"}`,
                    color: numComps === n ? "#00d4ff" : "#666", borderRadius: 3, cursor: "pointer",
                    fontSize: 11, fontFamily: "inherit",
                  }}>{n}K</button>
              ))}
            </div>

            {/* Component tabs */}
            <div style={{ display: "flex", gap: 4 }}>
              {Array.from({ length: numComps }, (_, k) => (
                <button key={k} className="comp-tab"
                  onClick={() => setActiveComp(k)}
                  style={{
                    flex: 1,
                    background: activeComp === k ? COMP_BG[k] : "rgba(255,255,255,0.02)",
                    border: `1px solid ${activeComp === k ? ELLIPSE_COLORS[k] + "66" : "rgba(255,255,255,0.07)"}`,
                    color: activeComp === k ? ELLIPSE_COLORS[k] : "#555",
                  }}>C{k + 1}</button>
              ))}
            </div>
          </div>

          <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 12 }}>
            <div style={{ fontSize: 9, color: "#555", letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 10 }}>
              Component {activeComp + 1}
              <span style={{ color: ELLIPSE_COLORS[activeComp], marginLeft: 8 }}>●</span>
            </div>

            <Sl label="weight w" value={c.w} min={0.05} max={2} step={0.05}
              color={ELLIPSE_COLORS[activeComp]}
              onChange={v => setComp(activeComp, c => ({ ...c, w: v }))} />

            <div style={{ fontSize: 9, color: "#444", letterSpacing: "0.1em", textTransform: "uppercase", margin: "10px 0 6px" }}>Mean μ</div>
            <Sl label="μ₁ (x-axis)" value={c.mu[0]} min={-2} max={4} step={0.05}
              color={ELLIPSE_COLORS[activeComp]}
              onChange={v => setComp(activeComp, c => ({ ...c, mu: [v, c.mu[1]] }))} />
            <Sl label="μ₂ (y-axis)" value={c.mu[1]} min={-2} max={4} step={0.05}
              color={ELLIPSE_COLORS[activeComp]}
              onChange={v => setComp(activeComp, c => ({ ...c, mu: [c.mu[0], v] }))} />

            <div style={{ fontSize: 9, color: "#444", letterSpacing: "0.1em", textTransform: "uppercase", margin: "10px 0 6px" }}>Covariance Σ</div>
            <Sl label="σ₁ (x spread)" value={sx} min={0.1} max={1.5} step={0.02}
              color={ELLIPSE_COLORS[activeComp]}
              onChange={setSx} />
            <Sl label="σ₂ (y spread)" value={sy} min={0.1} max={1.5} step={0.02}
              color={ELLIPSE_COLORS[activeComp]}
              onChange={setSy} />
            <Sl label="ρ (correlation)" value={rho} min={-0.95} max={0.95} step={0.02}
              color={ELLIPSE_COLORS[activeComp]}
              onChange={setSigmaFromRho} />
          </div>

          {/* Math info */}
          <div style={{
            borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 10,
            fontSize: 10, color: "#555", lineHeight: 1.8,
          }}>
            <div style={{ marginBottom: 4, fontSize: 9, color: "#444", letterSpacing: "0.1em", textTransform: "uppercase" }}>At current t</div>
            {params.components.slice(0, numComps).map((comp, k) => {
              const { mu: mu_t, Sigma: S_t } = ouParams(comp.mu, comp.Sigma, t);
              return (
                <div key={k} style={{ color: ELLIPSE_COLORS[k], marginBottom: 4, fontSize: 10 }}>
                  μ{k + 1} = [{mu_t[0].toFixed(2)}, {mu_t[1].toFixed(2)}]<br />
                  <span style={{ color: "#555" }}>
                    det(Σ{k + 1}) = {det2x2(S_t).toFixed(3)}
                  </span>
                </div>
              );
            })}
            <div style={{ marginTop: 8, fontSize: 9, color: "#444" }}>
              e⁻ᵗ = {Math.exp(-t).toFixed(3)} · e⁻²ᵗ = {Math.exp(-2 * t).toFixed(3)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
