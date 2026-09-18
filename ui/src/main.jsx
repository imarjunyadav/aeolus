import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import './styles.css'

const VESSEL = import.meta.env.VITE_VESSEL_API_URL || '/vessel-api'

function MapView({ route }) {
  React.useEffect(() => {
    const map = L.map('map', { zoomControl: true }).setView([-70, 0], 3)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors' }).addTo(map)
    if (route?.waypoints?.length) {
      const points = route.waypoints.map(w => [w.position.latitude, w.position.longitude])
      L.polyline(points, { weight: 5 }).addTo(map)
      L.marker(points[0]).addTo(map).bindPopup('Vessel')
      L.marker(points.at(-1)).addTo(map).bindPopup('Destination')
      map.fitBounds(points, { padding: [30, 30] })
    }
    return () => map.remove()
  }, [route])
  return <div id="map" />
}

function App() {
  const [form, setForm] = useState({ lat: -68.5, lon: -170.2, destLat: -66, destLon: -164, speed: 10 })
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('Ready')

  const run = async () => {
    setStatus('Syncing forecast…')
    const payload = { vessel_position: { latitude: +form.lat, longitude: +form.lon }, destination: { latitude: +form.destLat, longitude: +form.destLon }, forecast_horizons_hours: [6,12,24,48,72] }
    const sync = await fetch(`${VESSEL}/sync`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) })
    if (!sync.ok) throw new Error(await sync.text())
    setStatus('Computing safe route…')
    const navPayload = { vessel_id: 'DEMO-VESSEL', timestamp: new Date().toISOString(), latitude: +form.lat, longitude: +form.lon, speed_knots: +form.speed, heading_deg: 90, destination: payload.destination, context: { vessel_id: 'DEMO-VESSEL', max_ice_concentration: 0.85, min_iceberg_clearance_km: 5 } }
    const nav = await fetch(`${VESSEL}/navigate`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(navPayload) })
    if (!nav.ok) throw new Error(await nav.text())
    setResult(await nav.json())
    setStatus('Live decision-support cycle complete')
  }

  return <main>
    <header><div><span className="eyebrow">ANTARCTIC DECISION SUPPORT</span><h1>AEOLUS</h1></div><span className="status">● {status}</span></header>
    <section className="grid">
      <aside className="panel controls">
        <h2>Voyage</h2>
        {[["lat","Vessel latitude"],["lon","Vessel longitude"],["destLat","Destination latitude"],["destLon","Destination longitude"],["speed","Speed (knots)"]].map(([k,l]) => <label key={k}>{l}<input type="number" step="any" value={form[k]} onChange={e=>setForm({...form,[k]:e.target.value})}/></label>)}
        <button onClick={()=>run().catch(e=>setStatus(`Error: ${e.message}`))}>Sync + Plan Route</button>
        <p className="hint">The MVP uses real-provider forecasts when the cloud is configured for REAL mode, and deterministic safety-constrained routing onboard.</p>
      </aside>
      <section className="panel map-panel"><MapView route={result?.recommended_route}/></section>
      <aside className="panel metrics">
        <h2>Navigation</h2>
        {result?.no_safe_route ? <div className="danger">NO SAFE ROUTE<br/><small>{result.no_safe_route_reason}</small></div> : result?.recommended_route ? <>
          <div className="metric"><span>Risk</span><strong>{result.recommended_route.risk_level}</strong></div>
          <div className="metric"><span>ETA</span><strong>{result.recommended_route.estimated_eta_hours.toFixed(1)} h</strong></div>
          <div className="metric"><span>Distance</span><strong>{result.recommended_route.total_distance_nm.toFixed(1)} nm</strong></div>
          <div className="metric"><span>Relative fuel</span><strong>{result.recommended_route.estimated_fuel_relative.toFixed(1)}</strong></div>
          <div className="metric"><span>Confidence</span><strong>{(result.forecast_freshness.current_confidence*100).toFixed(0)}%</strong></div>
        </> : <p className="hint">Run a planning cycle to populate the decision panel.</p>}
        {result?.active_warnings?.length > 0 && <div className="warnings"><h3>Warnings</h3>{result.active_warnings.map(w=><div key={w.warning_id}>⚠ {w.message}</div>)}</div>}
      </aside>
    </section>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
