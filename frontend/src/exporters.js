export function downloadJson(filename, data) {
  downloadText(filename, JSON.stringify(data, null, 2), 'application/json')
}

export function downloadText(filename, text, type = 'text/plain;charset=utf-8') {
  const blob = new Blob([text], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export function toCsv(records) {
  if (!records.length) return ''
  const columns = Array.from(
    records.reduce((keys, row) => {
      Object.keys(row).forEach((key) => keys.add(key))
      return keys
    }, new Set()),
  )
  return [columns.map(csvEscape).join(','), ...records.map((row) => columns.map((column) => csvEscape(row[column])).join(','))].join('\n')
}

function csvEscape(value) {
  if (value === null || value === undefined) return ''
  const text = String(value)
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}
