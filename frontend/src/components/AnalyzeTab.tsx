import { useState, FormEvent } from 'react'

interface AnalyzeFormData {
  file: string
  preset: string
  report?: string
  json_output: boolean
  pretty: boolean
}

interface AnalyzeResult {
  status: string
  error?: string
  [key: string]: any
}

export default function AnalyzeTab() {
  const [formData, setFormData] = useState<AnalyzeFormData>({
    file: '',
    preset: 'dj_safe',
    report: '',
    json_output: false,
    pretty: true,
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const payload: any = {
        file: formData.file,
        preset: formData.preset,
        json_output: formData.json_output,
        pretty: formData.pretty,
      }
      if (formData.report) {
        payload.report = formData.report
      }

      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await response.json()
      setLoading(false)

      if (response.ok) {
        setResult(data)
      } else {
        setError(data.error || 'Unknown error occurred')
      }
    } catch (err) {
      setLoading(false)
      setError(err instanceof Error ? err.message : 'Network error occurred')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Audio File <span className="text-red-500">*</span>
          <span className="block text-xs font-normal text-gray-500 mt-1">
            Filename (relative to input/ directory or absolute path)
          </span>
        </label>
        <input
          type="text"
          required
          value={formData.file}
          onChange={(e) => setFormData({ ...formData, file: e.target.value })}
          placeholder="example.wav"
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
        />
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Preset <span className="text-gray-500 text-xs font-normal">(optional, default: dj_safe)</span>
        </label>
        <select
          value={formData.preset}
          onChange={(e) => setFormData({ ...formData, preset: e.target.value })}
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
        >
          <option value="dj_safe">dj_safe</option>
          <option value="dj_safe_strict">dj_safe_strict</option>
          <option value="dj_safe_lenient">dj_safe_lenient</option>
          <option value="more_motion">more_motion</option>
          <option value="no_vocals">no_vocals</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Report Path <span className="text-gray-500 text-xs font-normal">(optional)</span>
          <span className="block text-xs font-normal text-gray-500 mt-1">
            Output path for JSON report
          </span>
        </label>
        <input
          type="text"
          value={formData.report}
          onChange={(e) => setFormData({ ...formData, report: e.target.value })}
          placeholder="report.json"
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
        />
      </div>

      <div className="flex items-center space-x-2">
        <input
          type="checkbox"
          id="json_output"
          checked={formData.json_output}
          onChange={(e) => setFormData({ ...formData, json_output: e.target.checked })}
          className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
        />
        <label htmlFor="json_output" className="text-sm text-gray-700">
          JSON Output <span className="text-gray-500 text-xs">(optional)</span>
        </label>
      </div>

      <div className="flex items-center space-x-2">
        <input
          type="checkbox"
          id="pretty"
          checked={formData.pretty}
          onChange={(e) => setFormData({ ...formData, pretty: e.target.checked })}
          className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
        />
        <label htmlFor="pretty" className="text-sm text-gray-700">
          Pretty Print <span className="text-gray-500 text-xs">(optional, default: true)</span>
        </label>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>

      {loading && (
        <div className="text-center py-4">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
          <p className="mt-2 text-gray-600">Analyzing audio file...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="bg-green-50 border border-green-200 text-green-800 rounded-lg p-4">
          <strong>Success!</strong>
          <pre className="mt-2 text-xs bg-white p-3 rounded overflow-x-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </form>
  )
}
