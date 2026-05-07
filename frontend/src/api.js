export const API_URL = (import.meta.env?.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export async function requestJson(path, options = {}, fallbackMessage = 'No se pudo completar la operación') {
  let response
  try {
    response = await fetch(`${API_URL}${path}`, options)
  } catch (error) {
    throw new Error(`No se pudo conectar con el backend: ${error.message}`)
  }

  const data = await readJson(response)
  if (!response.ok) {
    throw new Error(data?.detail || fallbackMessage)
  }
  return data
}

async function readJson(response) {
  const text = await response.text()
  if (!text) return null

  try {
    return JSON.parse(text)
  } catch {
    if (!response.ok) return { detail: text }
    throw new Error('El backend devolvió una respuesta no válida')
  }
}
