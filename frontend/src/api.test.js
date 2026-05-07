import assert from 'node:assert/strict'
import test from 'node:test'

import { API_URL, requestJson } from './api.js'

test('API_URL defaults to local backend without trailing slash', () => {
  assert.equal(API_URL, 'http://localhost:8000')
})

test('requestJson returns parsed json on success', async () => {
  const calls = []
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options })
    return new Response(JSON.stringify({ status: 'ok' }), { status: 200 })
  }

  const data = await requestJson('/health', { method: 'GET' })

  assert.deepEqual(data, { status: 'ok' })
  assert.equal(calls[0].url, 'http://localhost:8000/health')
  assert.equal(calls[0].options.method, 'GET')
})

test('requestJson uses backend detail on error responses', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Falló algo' }), { status: 400 })

  await assert.rejects(() => requestJson('/bad'), /Falló algo/)
})

test('requestJson reports connection errors clearly', async () => {
  globalThis.fetch = async () => {
    throw new Error('ECONNREFUSED')
  }

  await assert.rejects(() => requestJson('/health'), /No se pudo conectar con el backend: ECONNREFUSED/)
})
