import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, UploadCloud, Sparkles, Loader2, MessageCircle, ShieldCheck, Zap } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const languages = ['English', 'Spanish', 'Russian', 'Portuguese']

function GradientCard({ title, subtitle, icon: Icon, children }) {
  return (
    <div className="glass">
      <div className="card-heading">
        <div className="icon-circle">
          <Icon size={18} />
        </div>
        <div>
          <p className="eyebrow">{subtitle}</p>
          <h3>{title}</h3>
        </div>
      </div>
      {children}
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <p className="eyebrow">{label}</p>
      <p className="metric-value">{value}</p>
    </div>
  )
}

function jsonPreview(data) {
  try {
    return JSON.stringify(data, null, 2)
  } catch (err) {
    return 'Unable to format response.'
  }
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [language, setLanguage] = useState('Spanish')
  const [customPrompt, setCustomPrompt] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [chatQuestion, setChatQuestion] = useState('')
  const [chatAnswer, setChatAnswer] = useState('')
  const [error, setError] = useState('')

  const canChat = useMemo(() => Boolean(analysis?.extractedText?.length), [analysis])

  const uploadFile = async () => {
    if (!selectedFile) {
      setError('Selecciona un archivo para empezar.')
      return
    }
    setError('')
    setIsUploading(true)
    setChatAnswer('')
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('language', language)
      if (customPrompt.trim()) formData.append('customPrompt', customPrompt)

      const response = await axios.post(`${API_BASE}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setAnalysis(response.data)
    } catch (err) {
      const message = err.response?.data?.detail || 'No pudimos procesar el documento.'
      setError(message)
    } finally {
      setIsUploading(false)
    }
  }

  const askQuestion = async () => {
    if (!chatQuestion.trim() || !canChat) return
    setChatAnswer('')
    try {
      const response = await axios.post(`${API_BASE}/api/chat`, {
        question: chatQuestion,
        extractedText: analysis.extractedText,
      })
      setChatAnswer(response.data.answer)
    } catch (err) {
      const message = err.response?.data?.detail || 'No pudimos obtener la respuesta.'
      setError(message)
    }
  }

  useEffect(() => {
    if (!error) return
    const timer = setTimeout(() => setError(''), 4000)
    return () => clearTimeout(timer)
  }, [error])

  return (
    <div className="page">
      <div className="hero">
        <div>
          <p className="eyebrow">IDP React</p>
          <h1>Procesa contratos con un front-end inspirado en InsightBot</h1>
          <p className="lede">
            Carga PDFs o imágenes, deja que Azure Form Recognizer y OpenAI extraigan la información clave y conversa con el resultado
            en un lienzo moderno, rápido y lleno de micro-animaciones.
          </p>
          <div className="hero-badges">
            <span className="pill"><Sparkles size={16} /> UX animada</span>
            <span className="pill"><ShieldCheck size={16} /> Datos seguros</span>
            <span className="pill"><Zap size={16} /> API + React</span>
          </div>
        </div>
        <div className="hero-card">
          <div className="upload-area" onClick={() => document.getElementById('file-input').click()}>
            <input
              id="file-input"
              type="file"
              className="hidden"
              accept="application/pdf,image/*"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
            />
            <UploadCloud size={32} />
            <p>{selectedFile ? selectedFile.name : 'Arrastra o selecciona un archivo'}</p>
            <small>PDF, JPG, PNG soportados</small>
          </div>
          <div className="controls">
            <label className="field">
              <span>Idioma de extracción</span>
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                {languages.map((lang) => (
                  <option key={lang}>{lang}</option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Prompt personalizado (opcional)</span>
              <textarea
                placeholder="Añade instrucciones adicionales para la extracción"
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
              />
            </label>
            <button className="primary" onClick={uploadFile} disabled={isUploading}>
              {isUploading ? <Loader2 className="spin" size={18} /> : 'Analizar documento'}
            </button>
          </div>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="grid">
        <GradientCard title="Resultados" subtitle="JSON estructurado" icon={FileText}>
          <div className="code-block">
            {analysis?.openaiResults ? <pre>{jsonPreview(analysis.openaiResults)}</pre> : <p className="muted">El resultado aparecerá aquí.</p>}
          </div>
        </GradientCard>

        <GradientCard title="Métricas" subtitle="Latencia en vivo" icon={Sparkles}>
          <div className="metrics-row">
            <Metric label="Azure Form Recognizer" value={analysis ? `${analysis.metrics.textract_duration.toFixed(2)}s` : '--'} />
            <Metric label="OpenAI" value={analysis ? `${analysis.metrics.openai_duration.toFixed(2)}s` : '--'} />
            <Metric label="Páginas" value={analysis ? analysis.metrics.page_count : '--'} />
          </div>
        </GradientCard>

        <GradientCard title="Texto OCR" subtitle="Vista previa" icon={MessageCircle}>
          <div className="scrollable">
            {analysis?.extractedText?.length ? (
              <ul>
                {analysis.extractedText.slice(0, 8).map((item, idx) => (
                  <li key={`${item.text}-${idx}`}>
                    <span className="pill">Pág. {item.page}</span>
                    <p>{item.text}</p>
                  </li>
                ))}
                {analysis.extractedText.length > 8 && <p className="muted">...y más líneas procesadas.</p>}
              </ul>
            ) : (
              <p className="muted">El texto extraído se mostrará aquí.</p>
            )}
          </div>
        </GradientCard>
      </div>

      <div className="chat">
        <div className="chat-header">
          <div className="icon-circle">
            <MessageCircle size={18} />
          </div>
          <div>
            <p className="eyebrow">Chat inteligente</p>
            <h3>Conversar sobre el documento</h3>
          </div>
        </div>
        <div className="chat-body">
          <input
            type="text"
            placeholder={canChat ? 'Pregúntale algo al documento...' : 'Carga un documento para habilitar el chat'}
            value={chatQuestion}
            onChange={(e) => setChatQuestion(e.target.value)}
            disabled={!canChat}
          />
          <button className="secondary" onClick={askQuestion} disabled={!canChat}>
            Preguntar
          </button>
        </div>
        <AnimatePresence>
          {chatAnswer && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="answer"
            >
              {chatAnswer}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
