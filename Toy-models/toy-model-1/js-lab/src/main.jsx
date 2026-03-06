import { useState } from 'react'
import ReactDOM from 'react-dom/client'
import OUDiffusionViz from './OUDiffusion.jsx'
import OUDiffusionLab from './OUDiffusionControls.jsx'
import ScoreFieldViz from './ScoreField.jsx'

const TABS = [
  { id: 'ou',       label: 'OU Diffusion',         Component: OUDiffusionViz  },
  { id: 'oulab',    label: 'Diffusion + Controls',  Component: OUDiffusionLab  },
  { id: 'score',    label: 'Score Field',           Component: ScoreFieldViz   },
]

function App() {
  const [active, setActive] = useState('ou')
  const { Component } = TABS.find(t => t.id === active)

  return (
    <div style={{ fontFamily: "'IBM Plex Mono', monospace" }}>
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
        background: 'rgba(8,8,16,0.95)', backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        display: 'flex', alignItems: 'center', gap: 4, padding: '0 20px', height: 44,
      }}>
        <span style={{ fontSize: 10, color: '#333', letterSpacing: '0.2em', marginRight: 16, textTransform: 'uppercase' }}>
          Diffusion Lab
        </span>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setActive(t.id)} style={{
            background: active === t.id ? 'rgba(0,212,255,0.12)' : 'transparent',
            border: `1px solid ${active === t.id ? 'rgba(0,212,255,0.35)' : 'transparent'}`,
            color: active === t.id ? '#00d4ff' : '#555',
            padding: '5px 14px', borderRadius: 3, cursor: 'pointer',
            fontSize: 11, fontFamily: 'inherit', letterSpacing: '0.05em',
            transition: 'all 0.15s',
          }}>{t.label}</button>
        ))}
      </nav>
      <div style={{ paddingTop: 44 }}>
        <Component />
      </div>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />)
