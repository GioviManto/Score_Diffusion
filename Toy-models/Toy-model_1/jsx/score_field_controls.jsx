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
  const q = d0*(iS[0][0]*d0 + iS[0][1]*d1) + d1*(iS[1][0]*d0 + iS[1][1]*d1);
  const d = det2x2(Sigma); if (d <= 0) return 0;
  return Math.exp(-0.5*q) / (2*Math.PI*Math.sqrt(d));
};
// ∇ log N(x; μ, Σ) = −Σ⁻¹(x − μ)
const scoreGaussian = (x1, x2, mu, Sigma) => {
  const iS = inv2x2(Sigma);
  const d0 = x1-mu[0], d1 = x2-mu[1];
  return [-(iS[0][0]*d0 + iS[0][1]*d1), -(iS[1][0]*d0 + iS[1][1]*d1)];
};
const chi2ppf2 = p => -2*Math.log(1-p+1e-12);

const ouParams = (mu0, S0, t) => {
  const a = Math.exp(-t), b = Math.exp(-2*t);
  return {
    mu: [a*mu0[0], a*mu0[1]],
    Sigma: [[b*S0[0][0]+(1-b), b*S0[0][1]], [b*S0[1][0], b*S0[1][1]+(1-b)]],
  };
};

// ∇ log p_t(x) = Σ_k r_k(x,t) · ∇ log p_k,t(x)  where r_k = w_k p_k / Σ w_j p_j
const mixturePdfAndScore = (x1, x2, comps, numC, t) => {
  const used = comps.slice(0, numC);
  const totalW = used.reduce((s,c) => s+c.w, 0);
  let ptotal=0, s0=0, s1=0;
  used.forEach(({mu:mu0, Sigma:S0, w}) => {
    const {mu, Sigma} = ouParams(mu0, S0, t);
    const p = (w/totalW) * gaussian2D(x1, x2, mu, Sigma);
    const [g0,g1] = scoreGaussian(x1, x2, mu, Sigma);
    ptotal += p; s0 += p*g0; s1 += p*g1;
  });
  if (ptotal < 1e-15) return {p:0, score:[0,0]};
  return {p:ptotal, score:[s0/ptotal, s1/ptotal]};
};

const ellipsePoints = (mu, Sigma, mass=0.8, n=180) => {
  const r = Math.sqrt(chi2ppf2(mass));
  const L = chol2(Sigma);
  return Array.from({length:n+1},(_,i)=>{
    const a=(2*Math.PI*i)/n, cx=r*Math.cos(a), cy=r*Math.sin(a);
    return [mu[0]+L[0][0]*cx, mu[1]+L[1][0]*cx+L[1][1]*cy];
  });
};

// ─── Palettes ──────────────────────────────────────────────────────────────────
const PLASMA = [
  [13,8,135],[84,2,163],[139,10,165],[185,50,137],
  [219,92,104],[244,136,73],[254,188,43],[240,249,33]
];
const plasmaColor = v => {
  const idx = Math.min(v*(PLASMA.length-1), PLASMA.length-1.001);
  const lo=Math.floor(idx), hi=lo+1, f=idx-lo;
  return PLASMA[lo].map((c,i)=>Math.round(c*(1-f)+PLASMA[hi][i]*f));
};

const ELLIPSE_COLORS = ["#ff4d6d","#00d4ff","#7fff6a","#ffb347"];
const COMP_BG = [
  "rgba(255,77,109,0.08)","rgba(0,212,255,0.08)",
  "rgba(127,255,106,0.08)","rgba(255,179,71,0.08)"
];

const DEFAULT_COMPS = [
  {w:0.55, mu:[1.2,0.8],  Sigma:[[0.25,0.18],[0.18,0.60]]},
  {w:0.45, mu:[2.2,2.8],  Sigma:[[0.35,-0.15],[-0.15,0.25]]},
  {w:0.40, mu:[-1.0,2.0], Sigma:[[0.20,0.05],[0.05,0.30]]},
  {w:0.35, mu:[3.0,0.5],  Sigma:[[0.40,-0.10],[-0.10,0.20]]},
];

const T_MAX = 2.5;
const DENSITY_GRID = 72;
const ARROW_GRID = 17;

// ─── Reusable slider ───────────────────────────────────────────────────────────
const Sl = ({label, value, min, max, step, onChange, color="#00d4ff", fmt=v=>v.toFixed(2)}) => (
  <div style={{marginBottom:7}}>
    <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3,fontFamily:"IBM Plex Mono,monospace"}}>
      <span style={{color:"#777"}}>{label}</span>
      <span style={{color}}>{fmt(value)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e=>onChange(parseFloat(e.target.value))}
      style={{width:"100%",accentColor:color,cursor:"pointer",height:3}}/>
  </div>
);

const Toggle = ({label, value, onChange, color="#00d4ff"}) => (
  <button onClick={()=>onChange(!value)} style={{
    background: value ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.02)",
    border:`1px solid ${value ? color+"88" : "rgba(255,255,255,0.08)"}`,
    color: value ? color : "#444", padding:"5px 10px", borderRadius:3,
    cursor:"pointer", fontSize:10, letterSpacing:"0.07em", textTransform:"uppercase",
    fontFamily:"IBM Plex Mono,monospace", transition:"all 0.15s", whiteSpace:"nowrap",
  }}>{label}</button>
);

export default function ScoreFieldViz() {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);
  const tRef      = useRef(0.4);
  const dirRef    = useRef(1);

  const [t,           setT]           = useState(0.4);
  const [playing,     setPlaying]     = useState(false);
  const [showDensity,     setShowDensity]     = useState(true);
  const [showEllipses,    setShowEllipses]    = useState(true);
  const [showScore,       setShowScore]       = useState(true);
  const [showStreamlines, setShowStreamlines] = useState(false);

  const [comps,   setComps]   = useState(DEFAULT_COMPS);
  const [numC,    setNumC]    = useState(2);
  const [activeK, setActiveK] = useState(0);

  const dpr  = Math.min(window.devicePixelRatio||1, 2);
  const SIZE = 500;
  const xmin=-2.5, xmax=5.5, ymin=-2.5, ymax=5.5;

  const toCanvas = useCallback((x,y)=>[
    ((x-xmin)/(xmax-xmin))*SIZE,
    SIZE-((y-ymin)/(ymax-ymin))*SIZE,
  ],[]);

  const draw = useCallback((t)=>{
    const canvas = canvasRef.current; if(!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0,0,SIZE*dpr,SIZE*dpr);
    ctx.save(); ctx.scale(dpr,dpr);

    ctx.fillStyle="#050508"; ctx.fillRect(0,0,SIZE,SIZE);

    // subtle grid
    ctx.strokeStyle="rgba(255,255,255,0.03)"; ctx.lineWidth=0.5;
    for(let gx=Math.ceil(xmin);gx<=Math.floor(xmax);gx++){
      const [cx]=toCanvas(gx,0); ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,SIZE);ctx.stroke();
    }
    for(let gy=Math.ceil(ymin);gy<=Math.floor(ymax);gy++){
      const[,cy]=toCanvas(0,gy); ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(SIZE,cy);ctx.stroke();
    }

    // density heatmap
    if(showDensity){
      const vals=new Float32Array(DENSITY_GRID*DENSITY_GRID); let maxV=0;
      for(let iy=0;iy<DENSITY_GRID;iy++) for(let ix=0;ix<DENSITY_GRID;ix++){
        const x1=xmin+(ix/(DENSITY_GRID-1))*(xmax-xmin);
        const x2=ymin+(iy/(DENSITY_GRID-1))*(ymax-ymin);
        const {p}=mixturePdfAndScore(x1,x2,comps,numC,t);
        vals[iy*DENSITY_GRID+ix]=p; if(p>maxV) maxV=p;
      }
      const img=new ImageData(DENSITY_GRID,DENSITY_GRID);
      for(let i=0;i<DENSITY_GRID*DENSITY_GRID;i++){
        const v=vals[i]/(maxV+1e-12);
        const[r,g,b]=plasmaColor(v);
        img.data[i*4]=r; img.data[i*4+1]=g; img.data[i*4+2]=b;
        img.data[i*4+3]=Math.round(190*Math.pow(v,0.45));
      }
      const off=document.createElement("canvas"); off.width=DENSITY_GRID; off.height=DENSITY_GRID;
      off.getContext("2d").putImageData(img,0,0);
      ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality="high";
      ctx.drawImage(off,0,0,SIZE,SIZE);
    }

    // streamlines — gradient ascent paths through the score field
    if(showStreamlines){
      const seeds=[];
      for(let iy=0;iy<6;iy++) for(let ix=0;ix<6;ix++){
        seeds.push([xmin+0.5+(ix/5)*(xmax-xmin-1), ymin+0.5+(iy/5)*(ymax-ymin-1)]);
      }
      seeds.forEach(([sx0,sy0])=>{
        let px=sx0, py=sy0;
        ctx.beginPath();
        const[cx0,cy0]=toCanvas(px,py); ctx.moveTo(cx0,cy0);
        for(let step=0;step<80;step++){
          const{score:[dx,dy],p}=mixturePdfAndScore(px,py,comps,numC,t);
          if(p<1e-9) break;
          px+=dx*0.05; py+=dy*0.05;
          if(px<xmin||px>xmax||py<ymin||py>ymax) break;
          const[cx,cy]=toCanvas(px,py); ctx.lineTo(cx,cy);
        }
        ctx.strokeStyle="rgba(160,220,255,0.14)"; ctx.lineWidth=0.9; ctx.stroke();
      });
    }

    // score vector field arrows
    if(showScore){
      const cellX=(xmax-xmin)/ARROW_GRID, cellY=(ymax-ymin)/ARROW_GRID;
      let maxMag=0;
      const arrows=[];
      for(let iy=0;iy<=ARROW_GRID;iy++) for(let ix=0;ix<=ARROW_GRID;ix++){
        const x1=xmin+ix*cellX, x2=ymin+iy*cellY;
        const{p,score:[s0,s1]}=mixturePdfAndScore(x1,x2,comps,numC,t);
        const mag=Math.sqrt(s0*s0+s1*s1);
        if(mag>maxMag) maxMag=mag;
        arrows.push({x1,x2,s0,s1,mag,p});
      }
      const arrowScale=0.27*Math.min(cellX,cellY)/(maxMag+1e-12)*ARROW_GRID;
      arrows.forEach(({x1,x2,s0,s1,mag})=>{
        if(mag<1e-4) return;
        const nm=mag/(maxMag+1e-12);
        const[cx,cy]=toCanvas(x1,x2);
        const ex=x1+s0*arrowScale, ey=x2+s1*arrowScale;
        const[ecx,ecy]=toCanvas(ex,ey);
        const hue=((Math.atan2(s1,s0)/Math.PI)*180+360)%360;
        const light=38+nm*55, alpha=0.25+nm*0.70;
        ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(ecx,ecy);
        ctx.strokeStyle=`hsla(${hue},75%,${light}%,${alpha})`;
        ctx.lineWidth=0.7+nm*1.3; ctx.stroke();
        const ang=Math.atan2(ecy-cy,ecx-cx), hl=3+nm*4.5;
        ctx.beginPath();
        ctx.moveTo(ecx,ecy);
        ctx.lineTo(ecx-hl*Math.cos(ang-0.42),ecy-hl*Math.sin(ang-0.42));
        ctx.lineTo(ecx-hl*Math.cos(ang+0.42),ecy-hl*Math.sin(ang+0.42));
        ctx.closePath();
        ctx.fillStyle=`hsla(${hue},75%,${light}%,${alpha})`; ctx.fill();
      });
    }

    // ellipses
    if(showEllipses){
      comps.slice(0,numC).forEach(({mu:mu0,Sigma:S0},k)=>{
        const{mu,Sigma}=ouParams(mu0,S0,t);
        const pts=ellipsePoints(mu,Sigma);
        ctx.beginPath();
        pts.forEach(([x,y],i)=>{
          const[cx,cy]=toCanvas(x,y); i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
        });
        ctx.closePath();
        ctx.strokeStyle=ELLIPSE_COLORS[k];
        ctx.lineWidth=k===activeK?2.8:1.6;
        ctx.shadowColor=ELLIPSE_COLORS[k]; ctx.shadowBlur=k===activeK?12:5;
        ctx.stroke(); ctx.shadowBlur=0;
        const[mx,my]=toCanvas(mu[0],mu[1]);
        ctx.beginPath();ctx.arc(mx,my,k===activeK?5:3,0,Math.PI*2);
        ctx.fillStyle=ELLIPSE_COLORS[k]; ctx.fill();
      });
    }

    // axes
    ctx.strokeStyle="rgba(255,255,255,0.18)"; ctx.lineWidth=0.8;
    const[ox,oy]=toCanvas(0,0);
    ctx.beginPath();ctx.moveTo(ox,0);ctx.lineTo(ox,SIZE);ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,oy);ctx.lineTo(SIZE,oy);ctx.stroke();
    ctx.fillStyle="rgba(255,255,255,0.3)"; ctx.font=`11px "IBM Plex Mono",monospace`;
    ctx.fillText("x₁",SIZE-18,oy-8); ctx.fillText("x₂",ox+8,14);

    ctx.restore();
  },[toCanvas, comps, numC, activeK, showDensity, showEllipses, showScore, showStreamlines, dpr]);

  useEffect(()=>{
    const c=canvasRef.current; if(!c) return;
    c.width=SIZE*dpr; c.height=SIZE*dpr;
    c.style.width=`${SIZE}px`; c.style.height=`${SIZE}px`;
  },[dpr]);

  useEffect(()=>{ draw(t); },[t,draw]);

  useEffect(()=>{
    if(!playing){cancelAnimationFrame(animRef.current);return;}
    const step=()=>{
      tRef.current+=dirRef.current*0.018;
      if(tRef.current<=0){tRef.current=0;dirRef.current=1;}
      if(tRef.current>=T_MAX){tRef.current=T_MAX;dirRef.current=-1;}
      setT(tRef.current); animRef.current=requestAnimationFrame(step);
    };
    animRef.current=requestAnimationFrame(step);
    return()=>cancelAnimationFrame(animRef.current);
  },[playing]);

  // ── helpers to mutate active component ────────────────────────────────────────
  const setComp = (k,fn) => setComps(prev=>prev.map((c,i)=>i===k?fn(c):c));
  const c   = comps[activeK];
  const sx  = Math.sqrt(c.Sigma[0][0]);
  const sy  = Math.sqrt(c.Sigma[1][1]);
  const rho = c.Sigma[0][1]/(sx*sy+1e-12);

  const setSx  = v => { const off=rho*v*sy;  setComp(activeK,c=>({...c,Sigma:[[v*v,off],[off,c.Sigma[1][1]]]})); };
  const setSy  = v => { const off=rho*sx*v;  setComp(activeK,c=>({...c,Sigma:[[c.Sigma[0][0],off],[off,v*v]]]})); };
  const setRho = v => { const off=v*sx*sy;   setComp(activeK,c=>({...c,Sigma:[[c.Sigma[0][0],off],[off,c.Sigma[1][1]]]})); };

  const scoreLabel = t<0.4
    ? "Sharp arrows — score locked onto modes"
    : t<1.3 ? "Bimodal tug-of-war — responsibilities split"
    : "Score ≈ −x — collapsing toward Gaussian";

  const panelW = 228;

  return (
    <div style={{
      background:"#050508", minHeight:"100vh", display:"flex", flexDirection:"column",
      alignItems:"center", justifyContent:"center", padding:"20px 10px",
      fontFamily:"'IBM Plex Mono',monospace", color:"#e0e0e0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&display=swap');
        input[type=range]{height:3px;border-radius:2px}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:2px}
      `}</style>

      <div style={{fontSize:9,letterSpacing:"0.25em",color:"#3a3a55",marginBottom:5,textTransform:"uppercase"}}>
        Score-Based Generative Modeling
      </div>
      <h1 style={{fontSize:17,fontWeight:600,margin:"0 0 3px",color:"#fff",letterSpacing:"-0.01em"}}>
        Score Function &amp; Parameter Explorer
      </h1>
      <p style={{fontSize:11,color:"#444",margin:"0 0 18px",fontStyle:"italic"}}>
        ∇<sub>x</sub> log p<sub>t</sub>(x) · edit the mixture · watch the field reshape live
      </p>

      <div style={{display:"flex",gap:14,alignItems:"flex-start",flexWrap:"wrap",justifyContent:"center"}}>

        {/* ── Canvas column ── */}
        <div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
          <div style={{
            position:"relative",borderRadius:4,overflow:"hidden",
            boxShadow:"0 0 60px rgba(150,50,255,0.08),0 0 0 1px rgba(255,255,255,0.05)",
          }}>
            <canvas ref={canvasRef}/>
            <div style={{
              position:"absolute",top:9,left:9,right:9,
              fontSize:10,color:"rgba(255,255,255,0.38)",
              background:"rgba(0,0,0,0.6)",padding:"4px 10px",borderRadius:3,
            }}>
              t = {t.toFixed(3)} · <span style={{color:"#c47cff"}}>{scoreLabel}</span>
            </div>
          </div>

          {/* time controls */}
          <div style={{marginTop:11,display:"flex",alignItems:"center",gap:10,width:SIZE}}>
            <button onClick={()=>setPlaying(p=>!p)} style={{
              background:playing?"rgba(255,77,109,0.12)":"rgba(180,100,255,0.1)",
              border:`1px solid ${playing?"rgba(255,77,109,0.35)":"rgba(180,100,255,0.3)"}`,
              color:playing?"#ff4d6d":"#c47cff", padding:"6px 14px", borderRadius:3,
              cursor:"pointer",fontSize:10,letterSpacing:"0.08em",textTransform:"uppercase",
              fontFamily:"inherit",whiteSpace:"nowrap",
            }}>{playing?"⏸ Pause":"▶ Play"}</button>
            <input type="range" min={0} max={T_MAX} step={0.01} value={t}
              onChange={e=>{const v=parseFloat(e.target.value);tRef.current=v;setT(v);}}
              style={{flex:1,accentColor:"#c47cff",cursor:"pointer"}}/>
            <span style={{fontSize:10,color:"#444",minWidth:30}}>t={t.toFixed(2)}</span>
          </div>

          {/* layer toggles */}
          <div style={{marginTop:8,display:"flex",gap:6,flexWrap:"wrap",width:SIZE}}>
            <Toggle label="Density"     value={showDensity}     onChange={setShowDensity}     color="#ffd166"/>
            <Toggle label="Ellipses"    value={showEllipses}    onChange={setShowEllipses}    color="#00d4ff"/>
            <Toggle label="∇log p"      value={showScore}       onChange={setShowScore}       color="#c47cff"/>
            <Toggle label="Streamlines" value={showStreamlines} onChange={setShowStreamlines} color="#7fff6a"/>
          </div>

          {/* live math strip */}
          <div style={{
            marginTop:9,width:SIZE,background:"rgba(255,255,255,0.02)",
            border:"1px solid rgba(255,255,255,0.05)",borderRadius:4,
            padding:"8px 14px",display:"flex",gap:20,fontSize:10,color:"#555",flexWrap:"wrap",
          }}>
            {[
              ["e⁻ᵗ",   Math.exp(-t).toFixed(4)],
              ["e⁻²ᵗ",  Math.exp(-2*t).toFixed(4)],
              ["noise", (1-Math.exp(-2*t)).toFixed(4)],
              ["signal",Math.exp(-2*t).toFixed(4)],
            ].map(([k,v])=>(
              <div key={k}><span style={{color:"#3a3a55"}}>{k} = </span><span style={{color:"#777"}}>{v}</span></div>
            ))}
          </div>
        </div>

        {/* ── Parameter panel ── */}
        <div style={{
          width:panelW, background:"rgba(255,255,255,0.025)",
          border:"1px solid rgba(255,255,255,0.07)", borderRadius:6,
          padding:"14px 13px", display:"flex", flexDirection:"column", gap:11,
          maxHeight:612, overflowY:"auto",
        }}>

          {/* K selector */}
          <div>
            <div style={{fontSize:9,color:"#3a3a55",letterSpacing:"0.15em",textTransform:"uppercase",marginBottom:7}}>
              Components
            </div>
            <div style={{display:"flex",gap:5,marginBottom:9}}>
              {[1,2,3,4].map(n=>(
                <button key={n} onClick={()=>{setNumC(n);if(activeK>=n)setActiveK(n-1);}} style={{
                  flex:1, padding:"5px 0",
                  background:numC===n?"rgba(196,124,255,0.15)":"rgba(255,255,255,0.03)",
                  border:`1px solid ${numC===n?"rgba(196,124,255,0.45)":"rgba(255,255,255,0.07)"}`,
                  color:numC===n?"#c47cff":"#555", borderRadius:3, cursor:"pointer",
                  fontSize:11,fontFamily:"inherit",
                }}>{n}K</button>
              ))}
            </div>

            {/* component tabs */}
            <div style={{display:"flex",gap:4}}>
              {Array.from({length:numC},(_,k)=>(
                <button key={k} onClick={()=>setActiveK(k)} style={{
                  flex:1, padding:"5px 0", borderRadius:3, cursor:"pointer",
                  fontSize:10, fontFamily:"inherit",
                  background:activeK===k?COMP_BG[k]:"rgba(255,255,255,0.02)",
                  border:`1px solid ${activeK===k?ELLIPSE_COLORS[k]+"66":"rgba(255,255,255,0.07)"}`,
                  color:activeK===k?ELLIPSE_COLORS[k]:"#555",
                }}>C{k+1}</button>
              ))}
            </div>
          </div>

          {/* per-component sliders */}
          <div style={{borderTop:"1px solid rgba(255,255,255,0.06)",paddingTop:11}}>
            <div style={{
              fontSize:9,color:ELLIPSE_COLORS[activeK],letterSpacing:"0.12em",
              textTransform:"uppercase",marginBottom:9,
            }}>
              ● Component {activeK+1}
            </div>

            <Sl label="weight w" value={c.w} min={0.05} max={2} step={0.05}
              color={ELLIPSE_COLORS[activeK]}
              onChange={v=>setComp(activeK,c=>({...c,w:v}))}/>

            <div style={{fontSize:9,color:"#3a3a55",letterSpacing:"0.1em",textTransform:"uppercase",margin:"9px 0 6px"}}>
              Mean μ
            </div>
            <Sl label="μ₁  (x-axis)" value={c.mu[0]} min={-2} max={4.5} step={0.05}
              color={ELLIPSE_COLORS[activeK]}
              onChange={v=>setComp(activeK,c=>({...c,mu:[v,c.mu[1]]}))}/>
            <Sl label="μ₂  (y-axis)" value={c.mu[1]} min={-2} max={4.5} step={0.05}
              color={ELLIPSE_COLORS[activeK]}
              onChange={v=>setComp(activeK,c=>({...c,mu:[c.mu[0],v]}))}/>

            <div style={{fontSize:9,color:"#3a3a55",letterSpacing:"0.1em",textTransform:"uppercase",margin:"9px 0 6px"}}>
              Covariance Σ
            </div>
            <Sl label="σ₁  (x spread)" value={sx} min={0.1} max={1.6} step={0.02}
              color={ELLIPSE_COLORS[activeK]} onChange={setSx}/>
            <Sl label="σ₂  (y spread)" value={sy} min={0.1} max={1.6} step={0.02}
              color={ELLIPSE_COLORS[activeK]} onChange={setSy}/>
            <Sl label="ρ   (correlation)" value={rho} min={-0.95} max={0.95} step={0.02}
              color={ELLIPSE_COLORS[activeK]} onChange={setRho}/>
          </div>

          {/* live component readout */}
          <div style={{borderTop:"1px solid rgba(255,255,255,0.06)",paddingTop:10,fontSize:10,color:"#555",lineHeight:1.85}}>
            <div style={{fontSize:9,color:"#3a3a55",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:7}}>
              At current t
            </div>
            {comps.slice(0,numC).map((comp,k)=>{
              const{mu:mt,Sigma:St}=ouParams(comp.mu,comp.Sigma,t);
              return(
                <div key={k} style={{marginBottom:5}}>
                  <span style={{color:ELLIPSE_COLORS[k]}}>C{k+1}</span>
                  <span style={{color:"#444"}}> μ=[{mt[0].toFixed(2)},{mt[1].toFixed(2)}]</span><br/>
                  <span style={{color:"#333",fontSize:9}}>det(Σ)={det2x2(St).toFixed(3)}</span>
                </div>
              );
            })}
          </div>

          {/* theory notes */}
          <div style={{borderTop:"1px solid rgba(255,255,255,0.06)",paddingTop:10,fontSize:10,color:"#555",lineHeight:1.8}}>
            <div style={{fontSize:9,color:"#3a3a55",letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:7}}>
              Key identity
            </div>
            <div style={{color:"#666",fontFamily:"monospace",fontSize:9,lineHeight:2}}>
              s(x,t) = Σ_k r_k · s_k(x,t)<br/>
              r_k = w_k p_k / Σ w_j p_j<br/>
              s_k = −Σ_k⁻¹(x−μ_k(t))
            </div>
            <div style={{marginTop:8,color:"#3a3a55",fontSize:9,lineHeight:1.7}}>
              r_k is the <em style={{color:"#555"}}>posterior responsibility</em> — how much component k claims point x. Score = responsibility-weighted average of component scores.
            </div>
            <div style={{marginTop:9,color:"#555",fontFamily:"monospace",fontSize:9,lineHeight:2}}>
              Reverse SDE:<br/>
              dX=[X+2s(X,t)]dt+√2 dW̄
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
