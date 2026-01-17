import { useState } from 'react'
import AnalyzeTab from './components/AnalyzeTab'
import ExtendTab from './components/ExtendTab'

function App() {
  const [activeTab, setActiveTab] = useState<'analyze' | 'extend'>('analyze')

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-indigo-600 p-4">
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-8 text-center">
          <h1 className="text-4xl font-bold mb-2">🎵 MixInto</h1>
          <p className="text-purple-100 text-lg">
            Extend track intros into longer, mix-safe lead-ins
          </p>
        </div>

        {/* Tabs */}
        <div className="flex bg-gray-100 border-b-2 border-gray-200">
          <button
            onClick={() => setActiveTab('analyze')}
            className={`flex-1 px-6 py-4 text-center font-semibold transition-colors ${
              activeTab === 'analyze'
                ? 'bg-white text-purple-600 border-b-3 border-purple-600'
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            Analyze
          </button>
          <button
            onClick={() => setActiveTab('extend')}
            className={`flex-1 px-6 py-4 text-center font-semibold transition-colors ${
              activeTab === 'extend'
                ? 'bg-white text-purple-600 border-b-3 border-purple-600'
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            Extend
          </button>
        </div>

        {/* Content */}
        <div className="p-8">
          {activeTab === 'analyze' ? <AnalyzeTab /> : <ExtendTab />}
        </div>
      </div>
    </div>
  )
}

export default App
