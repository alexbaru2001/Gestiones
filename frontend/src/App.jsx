import { useState } from 'react'

export function App() {
  const [status, setStatus] = useState('')

  const checkHealth = async () => {
    const res = await fetch('http://localhost:8000/health')
    const data = await res.json()
    setStatus(data.status)
  }

  return (
    <main style={{ fontFamily: 'sans-serif', padding: 24 }}>
      <h1>Gestiones</h1>
      <p>Frontend mínimo para consumir backend FastAPI.</p>
      <button onClick={checkHealth}>Comprobar health backend</button>
      {status && <p>Estado backend: {status}</p>}
    </main>
  )
}
