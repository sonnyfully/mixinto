import { useState, FormEvent } from 'react'

interface ExtendFormData {
  file: string
  output?: string
  bars?: number
  seconds?: number
  preset: string
  backend: string
  seed: string
  report?: string
  json_output: boolean
  pretty: boolean
  overwrite: boolean
  dry_run: boolean
}

interface ExtendResult {
  status: string
  error?: string
  refused?: boolean
  refusal_reason?: string
  [key: string]: any
}

export default function ExtendTab() {
  const [formData, setFormData] = useState<ExtendFormData>({
    file: '',
    output: '',
    bars: undefined,
    seconds: undefined,
    preset: 'dj_safe',
    backend: 'baseline',
    seed: '0',
    report: '',
    json_output: false,
    pretty: true,
    overwrite: false,
    dry_run: false,
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ExtendResult | null>(null)
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
        backend: formData.backend,
        seed: formData.seed,
        json_output: formData.json_output,
        pretty: formData.pretty,
        overwrite: formData.overwrite,
        dry_run: formData.dry_run,
      }
      if (formData.output) payload.output = formData.output
      if (formData.bars) payload.bars = formData.bars
      if (formData.seconds) payload.seconds = formData.seconds
      if (formData.report) payload.report = formData.report

      const response = await fetch('/api/extend', {
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
        if (data.status === 'refused') {
          setError(data.refusal_reason || 'Extension was refused')
        }
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
          Output File <span className="text-gray-500 text-xs font-normal">(optional)</span>
          <span className="block text-xs font-normal text-gray-500 mt-1">
            Output filename (defaults to output/{'{input}'}_extended.wav)
          </span>
        </label>
        <input
          type="text"
          value={formData.output}
          onChange={(e) => setFormData({ ...formData, output: e.target.value })}
          placeholder="output.wav"
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
        />
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Bars <span className="text-gray-500 text-xs font-normal">(optional, use bars OR seconds)</span>
          <span className="block text-xs font-normal text-gray-500 mt-1">
            Number of bars to extend the intro by
          </span>
        </label>
        <input
          type="number"
          min="1"
          value={formData.bars || ''}
          onChange={(e) => setFormData({ ...formData, bars: e.target.value ? parseInt(e.target.value) : undefined })}
          placeholder="4"
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
        />
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Seconds <span className="text-gray-500 text-xs font-normal">(optional, use bars OR seconds)</span>
          <span className="block text-xs font-normal text-gray-500 mt-1">
            Number of seconds to extend the intro by
          </span>
        </label>
        <input
          type="number"
          step="0.1"
          min="0"
          value={formData.seconds || ''}
          onChange={(e) => setFormData({ ...formData, seconds: e.target.value ? parseFloat(e.target.value) : undefined })}
          placeholder="16.0"
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
          Backend <span className="text-gray-500 text-xs font-normal">(optional, default: baseline)</span>
        </label>
        <select
          value={formData.backend}
          onChange={(e) => setFormData({ ...formData, backend: e.target.value })}
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
        >
          <option value="baseline">baseline</option>
          <option value="loop">loop</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Seed <span className="text-gray-500 text-xs font-normal">(optional, default: 0)</span>
          <span className="block text-xs font-normal text-gray-500 mt-1">
            Random seed for deterministic generation (use 'random' for random seed)
          </span>
        </label>
        <input
          type="text"
          value={formData.seed}
          onChange={(e) => setFormData({ ...formData, seed: e.target.value })}
          placeholder="0"
          className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500"
        />
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

      <div className="space-y-3">
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

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="overwrite"
            checked={formData.overwrite}
            onChange={(e) => setFormData({ ...formData, overwrite: e.target.checked })}
            className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
          />
          <label htmlFor="overwrite" className="text-sm text-gray-700">
            Overwrite <span className="text-gray-500 text-xs">(optional)</span>
          </label>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="dry_run"
            checked={formData.dry_run}
            onChange={(e) => setFormData({ ...formData, dry_run: e.target.checked })}
            className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
          />
          <label htmlFor="dry_run" className="text-sm text-gray-700">
            Dry Run <span className="text-gray-500 text-xs">(optional)</span>
          </label>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
      >
        {loading ? 'Extending...' : 'Extend'}
      </button>

      {loading && (
        <div className="text-center py-4">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
          <p className="mt-2 text-gray-600">Extending audio file...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 rounded-lg p-4">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className={`border rounded-lg p-4 ${
          result.status === 'refused' || result.error
            ? 'bg-red-50 border-red-200 text-red-800'
            : 'bg-green-50 border-green-200 text-green-800'
        }`}>
          <strong>
            {result.status === 'refused' ? 'Extension Refused' : result.error ? 'Error' : 'Success!'}
          </strong>
          {result.refusal_reason && (
            <p className="mt-2">Reason: {result.refusal_reason}</p>
          )}
          <pre className="mt-2 text-xs bg-white p-3 rounded overflow-x-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </form>
  )
}
