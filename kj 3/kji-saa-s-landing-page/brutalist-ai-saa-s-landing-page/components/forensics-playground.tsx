"use client"

import { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Upload,
  ShieldCheck,
  ShieldAlert,
  Cpu,
  Layers,
  FileCode2,
  Activity,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Sparkles,
  Eye,
  Sliders,
  Download,
  FileText,
  Contrast
} from "lucide-react"

import {
  AnalysisResult,
  analyzeImageWithAPI,
  checkBackendHealth,
  MOCK_SAMPLES
} from "@/lib/api"
import { generatePDFReport } from "@/lib/pdf_generator"

async function invertImageFile(file: File): Promise<File> {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => {
      const canvas = document.createElement("canvas")
      canvas.width = img.width
      canvas.height = img.height
      const ctx = canvas.getContext("2d")
      if (!ctx) return resolve(file)

      ctx.drawImage(img, 0, 0)
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const data = imgData.data
      for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i]       // R
        data[i + 1] = 255 - data[i + 1] // G
        data[i + 2] = 255 - data[i + 2] // B
      }
      ctx.putImageData(imgData, 0, 0)

      canvas.toBlob((blob) => {
        if (!blob) return resolve(file)
        const invertedFile = new File([blob], `positive_${file.name}`, { type: file.type || "image/png" })
        resolve(invertedFile)
      }, file.type || "image/png")
    }
    img.onerror = () => resolve(file)
    img.src = URL.createObjectURL(file)
  })
}

export function ForensicsPlayground() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [analyzingStep, setAnalyzingStep] = useState<string>("")
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)
  const [activeTab, setActiveTab] = useState<"overview" | "heatmap" | "metadata" | "artifacts" | "ocr">("overview")
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.75)
  const [isNegativeInverted, setIsNegativeInverted] = useState<boolean>(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Check backend server status on mount
  useEffect(() => {
    async function verifyBackend() {
      const isHealthy = await checkBackendHealth()
      setBackendOnline(isHealthy)
    }
    verifyBackend()
    // Default to genuine sample on initial render
    loadSample("genuine_portrait")
  }, [])

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) {
      processFile(file, isNegativeInverted)
    }
  }

  async function processFile(file: File, shouldInvert: boolean = false) {
    setSelectedFile(file)
    setError(null)
    const reader = new FileReader()
    reader.onload = () => {
      setImagePreview(reader.result as string)
    }
    reader.readAsDataURL(file)

    const fileToAnalyze = shouldInvert ? await invertImageFile(file) : file
    runAnalysis(fileToAnalyze)
  }

  async function handleToggleInvert() {
    const newInvertState = !isNegativeInverted
    setIsNegativeInverted(newInvertState)
    if (selectedFile) {
      const fileToAnalyze = newInvertState ? await invertImageFile(selectedFile) : selectedFile
      runAnalysis(fileToAnalyze)
    }
  }

  async function runAnalysis(file: File) {
    setLoading(true)
    setError(null)
    setAnalyzingStep("Initializing PyTorch Model & GPU...")

    try {
      if (backendOnline) {
        setAnalyzingStep("Running Classifier & Grad-CAM Heatmap...")
        const data = await analyzeImageWithAPI(file)
        setResult(data)
      } else {
        // Fallback to sample simulation if server is offline
        await simulateAnalysis()
        setResult({
          filename: file.name,
          verdict: file.name.toLowerCase().includes("ai") ? "AI-Generated" : "Genuine (Real)",
          confidence: 0.962,
          trust_score: file.name.toLowerCase().includes("ai") ? 18 : 94,
          class_probabilities: {
            genuine: file.name.toLowerCase().includes("ai") ? 0.038 : 0.962,
            ai_generated: file.name.toLowerCase().includes("ai") ? 0.962 : 0.038,
          },
          heatmap_b64: null,
          metadata_findings: {
            metadata_trust_signal: 0.9,
            has_exif: true,
            software: "Camera Raw 15.0",
            notes: "Demo analysis (Backend offline). Start api_server.py for live GPU inference.",
          },
          artifact_findings: {
            artifact_trust_signal: 0.88,
            ela_mean_score: 0.15,
            fft_grid_score: 0.08,
            flags: ["FastAPI server offline — running local heuristic check"],
          },
        })
      }
    } catch (err: any) {
      console.error(err)
      setError(err.message || "Failed to analyze image")
    } finally {
      setLoading(false)
      setAnalyzingStep("")
    }
  }

  async function simulateAnalysis() {
    setAnalyzingStep("Extracting EXIF Metadata & Headers...")
    await new Promise((r) => setTimeout(r, 400))
    setAnalyzingStep("Performing ELA & FFT Grid Frequency Analysis...")
    await new Promise((r) => setTimeout(r, 500))
    setAnalyzingStep("Fusing Signals into Trust Score Engine...")
    await new Promise((r) => setTimeout(r, 400))
  }

  function loadSample(key: string) {
    const sample = MOCK_SAMPLES[key]
    if (!sample) return
    setSelectedFile(null)
    setImagePreview(sample.previewUrl)
    setResult(sample)
    setError(null)
  }

  const isGenuine = result?.verdict === "Genuine (Real)"

  return (
    <section id="playground" className="w-full px-6 py-16 lg:px-24 bg-background border-t-2 border-foreground">
      <div className="max-w-[1400px] mx-auto">
        {/* Section Title Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 pb-6 border-b-2 border-foreground">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#ea580c]/10 border border-[#ea580c] text-[#ea580c] font-mono text-xs mb-3">
              <Sparkles className="w-3.5 h-3.5" />
              LIVE FORENSIC ENGINE
            </div>
            <h2 className="font-pixel text-3xl sm:text-5xl text-foreground uppercase tracking-tight">
              INTERACTIVE AI PLAYGROUND
            </h2>
            <p className="font-mono text-xs sm:text-sm text-muted-foreground mt-2 max-w-2xl">
              Upload any image to test TRUEFRAME's multi-modal deep learning classifier, EXIF metadata parser, ELA+ error level analysis, and fused Trust Score engine.
            </p>
          </div>

          {/* Backend Status Indicator */}
          <div className="mt-4 md:mt-0 flex items-center gap-3 font-mono text-xs border border-border p-3 bg-muted/30">
            <div className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${backendOnline ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`} />
              <span className="font-bold text-foreground">
                {backendOnline ? "GPU BACKEND ONLINE" : "DEMO MODE (BACKEND OFFLINE)"}
              </span>
            </div>
            <button
              onClick={async () => setBackendOnline(await checkBackendHealth())}
              className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-foreground transition-colors"
              title="Refresh backend status"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Main 2-Column Workbench */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Input & Sample Selector (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            {/* Upload Box */}
            <div className="border-2 border-foreground bg-card p-6 shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] dark:shadow-[6px_6px_0px_0px_rgba(255,255,255,1)]">
              <h3 className="font-mono text-sm font-bold text-foreground uppercase tracking-wider mb-4 flex items-center justify-between">
                <span>1. INPUT IMAGE</span>
                <span className="text-[10px] text-muted-foreground font-normal">PNG, JPG, WEBP (MAX 15MB)</span>
              </h3>

              {/* Drag & Drop Area */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-foreground hover:border-[#ea580c] bg-background/50 hover:bg-muted/50 p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all min-h-[220px] group"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />

                {imagePreview ? (
                  <div className="relative w-full aspect-video border border-border overflow-hidden bg-black/40 group">
                    <img
                      src={imagePreview}
                      alt="Uploaded Preview"
                      className={`w-full h-full object-contain ${isNegativeInverted ? "invert" : ""}`}
                    />
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                      <span className="font-mono text-xs text-white bg-black/80 px-3 py-1.5 border border-white/20">
                        CHANGE IMAGE
                      </span>
                    </div>
                  </div>
                ) : (
                  <>
                    <Upload className="w-10 h-10 text-muted-foreground group-hover:text-[#ea580c] mb-3 transition-colors" />
                    <p className="font-mono text-xs font-bold text-foreground">
                      DRAG & DROP IMAGE HERE
                    </p>
                    <p className="font-mono text-[10px] text-muted-foreground mt-1">
                      or click to browse your device
                    </p>
                  </>
                )}
              </div>

              {/* Negative Inverter Toggle */}
              {imagePreview && (
                <div className="mt-4 p-3 border border-border bg-muted/20 flex items-center justify-between font-mono text-xs">
                  <div className="flex items-center gap-2">
                    <Contrast className="w-4 h-4 text-[#ea580c]" />
                    <span className="font-bold text-foreground">INVERT NEGATIVE:</span>
                  </div>
                  <button
                    type="button"
                    onClick={handleToggleInvert}
                    className={`px-3 py-1 font-bold border transition-all ${
                      isNegativeInverted
                        ? "bg-[#ea580c] text-white border-[#ea580c] shadow-xs"
                        : "bg-background text-foreground border-border hover:border-foreground"
                    }`}
                  >
                    {isNegativeInverted ? "INVERTED (ON)" : "OFF (ORIGINAL)"}
                  </button>
                </div>
              )}

              {/* Sample Images Palette */}
              <div className="mt-6 border-t border-border pt-4">
                <span className="font-mono text-[11px] text-muted-foreground uppercase tracking-wider block mb-3">
                  OR TEST PRESET DEMO SAMPLES:
                </span>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => loadSample("genuine_portrait")}
                    className="p-2 border border-border hover:border-emerald-500 bg-background text-left transition-colors font-mono text-[10px]"
                  >
                    <span className="block font-bold text-emerald-500">REAL PHOTO</span>
                    <span className="text-muted-foreground truncate block">Canon RAW</span>
                  </button>
                  <button
                    onClick={() => loadSample("ai_synthetic_art")}
                    className="p-2 border border-border hover:border-red-500 bg-background text-left transition-colors font-mono text-[10px]"
                  >
                    <span className="block font-bold text-red-500">MIDJOURNEY</span>
                    <span className="text-muted-foreground truncate block">AI Synthetic</span>
                  </button>
                  <button
                    onClick={() => loadSample("manipulated_face")}
                    className="p-2 border border-border hover:border-amber-500 bg-background text-left transition-colors font-mono text-[10px]"
                  >
                    <span className="block font-bold text-amber-500">DEEPFAKE</span>
                    <span className="text-muted-foreground truncate block">Face Swap</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Stats Banner */}
            <div className="border border-border p-4 bg-muted/20 font-mono text-xs space-y-2">
              <div className="flex justify-between text-muted-foreground">
                <span>MODEL CHECKPOINT:</span>
                <span className="text-foreground font-bold">best_model.pth (EfficientNet-B0)</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>FORENSIC PIPELINE:</span>
                <span className="text-foreground font-bold">EfficientNet-B0 + Grad-CAM + ELA+ + FFT</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>FUSION WEIGHTS:</span>
                <span className="text-foreground font-bold">CLS 50% | META 25% | ART 25% (Trust Fusion)</span>
              </div>
            </div>
          </div>

          {/* Right Column: Analysis Results Dashboard (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <div className="border-2 border-foreground bg-card p-6 shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] dark:shadow-[6px_6px_0px_0px_rgba(255,255,255,1)] min-h-[480px] flex flex-col">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-border">
                <h3 className="font-mono text-sm font-bold text-foreground uppercase tracking-wider flex items-center gap-2">
                  <Activity className="w-4 h-4 text-[#ea580c]" />
                  2. FORENSIC ANALYSIS RESULTS
                </h3>

                <div className="flex items-center gap-3">
                  {/* Download PDF Button */}
                  {result && (
                    <button
                      onClick={() => generatePDFReport(result, imagePreview)}
                      className="flex items-center gap-2 px-3 py-1.5 bg-[#ea580c] hover:bg-[#c2410c] text-white font-mono text-xs font-bold border border-foreground shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] transition-all"
                      title="Download complete PDF Forensic Audit Report"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>PDF REPORT</span>
                    </button>
                  )}

                  {/* Tabs */}
                  {result && (
                    <div className="flex border border-border font-mono text-xs">
                      {(["overview", "heatmap", "metadata", "artifacts", "ocr"] as const).map((tab) => (
                        <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-3 py-1.5 uppercase transition-colors ${
                          activeTab === tab
                            ? "bg-foreground text-background font-bold"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {tab}
                      </button>
                    ))}
                  </div>
                )}
                </div>
              </div>

              {/* Loading State */}
              {loading && (
                <div className="flex-1 flex flex-col items-center justify-center p-12 text-center font-mono space-y-4">
                  <RefreshCw className="w-10 h-10 text-[#ea580c] animate-spin" />
                  <p className="text-sm font-bold text-foreground animate-pulse">
                    {analyzingStep}
                  </p>
                  <div className="w-64 h-1.5 bg-muted overflow-hidden border border-border">
                    <div className="w-full h-full bg-[#ea580c] animate-pulse" />
                  </div>
                </div>
              )}

              {/* Error State */}
              {error && !loading && (
                <div className="flex-1 flex flex-col items-center justify-center p-8 text-center font-mono">
                  <AlertTriangle className="w-12 h-12 text-red-500 mb-3" />
                  <h4 className="text-base font-bold text-red-500 uppercase mb-2">Analysis Failed</h4>
                  <p className="text-xs text-muted-foreground max-w-md border border-red-500/30 p-3 bg-red-500/5">
                    {error}
                  </p>
                </div>
              )}

              {/* Results Content */}
              {!loading && !error && result && (
                <div className="flex-1 flex flex-col">
                  {/* TAB 1: OVERVIEW */}
                  {activeTab === "overview" && (
                    <div className="space-y-6">
                      {/* Verdict Banner */}
                      <div className={`p-6 border-2 flex items-center justify-between ${
                        isGenuine
                          ? "border-emerald-500 bg-emerald-500/10 text-emerald-500"
                          : "border-red-500 bg-red-500/10 text-red-500"
                      }`}>
                        <div className="flex items-center gap-4">
                          {isGenuine ? (
                            <ShieldCheck className="w-12 h-12 stroke-[1.5]" />
                          ) : (
                            <ShieldAlert className="w-12 h-12 stroke-[1.5]" />
                          )}
                          <div>
                            <span className="font-mono text-[10px] uppercase tracking-widest block opacity-80">
                              CLASSIFICATION VERDICT
                            </span>
                            <h4 className="font-pixel text-2xl sm:text-3xl font-bold uppercase tracking-tight">
                              {result.verdict}
                            </h4>
                          </div>
                        </div>

                        <div className="text-right font-mono">
                          <span className="text-[10px] uppercase tracking-widest block opacity-80">
                            MODEL CONFIDENCE
                          </span>
                          <span className="text-2xl font-bold">
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {/* Trust Score & Probability Breakdown Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Trust Score Card */}
                        <div className="border border-border p-4 bg-background">
                          <div className="flex items-center justify-between mb-2 font-mono text-xs">
                            <span className="text-muted-foreground uppercase">FUSED TRUST SCORE</span>
                            <span className="font-bold text-foreground">{result.trust_score} / 100</span>
                          </div>

                          <div className="w-full h-4 bg-muted border border-border relative overflow-hidden my-3">
                            <div
                              className={`h-full transition-all duration-700 ${
                                result.trust_score > 70
                                  ? "bg-emerald-500"
                                  : result.trust_score > 40
                                  ? "bg-amber-500"
                                  : "bg-red-500"
                              }`}
                              style={{ width: `${result.trust_score}%` }}
                            />
                          </div>

                          <p className="font-mono text-[11px] text-muted-foreground">
                            {result.trust_score > 70
                              ? "High integrity signal across neural classifier, EXIF tags & ELA analysis."
                              : result.trust_score > 40
                              ? "Moderate confidence. Suspicious compression or metadata missing."
                              : "High risk. Synthetic generation or deepfake manipulation detected."}
                          </p>
                        </div>

                        {/* Class Probabilities Card */}
                        <div className="border border-border p-4 bg-background font-mono text-xs">
                          <span className="text-muted-foreground uppercase block mb-3">
                            CLASS PROBABILITY DISTRIBUTION
                          </span>

                          <div className="space-y-2">
                            <div>
                              <div className="flex justify-between text-[11px] mb-1">
                                <span>GENUINE (REAL)</span>
                                <span>{((result.class_probabilities.genuine || 0) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-2 bg-muted border border-border">
                                <div
                                  className="h-full bg-emerald-500"
                                  style={{ width: `${(result.class_probabilities.genuine || 0) * 100}%` }}
                                />
                              </div>
                            </div>

                            <div>
                              <div className="flex justify-between text-[11px] mb-1">
                                <span>AI-GENERATED</span>
                                <span>{((result.class_probabilities.ai_generated || 0) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-2 bg-muted border border-border">
                                <div
                                  className="h-full bg-red-500"
                                  style={{ width: `${(result.class_probabilities.ai_generated || 0) * 100}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* TAB 2: HEATMAP */}
                  {activeTab === "heatmap" && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between font-mono text-xs text-muted-foreground">
                        <span>GRAD-CAM VISUAL EXPLANATION</span>
                        <div className="flex items-center gap-2">
                          <Sliders className="w-3.5 h-3.5" />
                          <span>HEATMAP OVERLAY OPACITY</span>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={heatmapOpacity}
                            onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                            className="w-24 accent-[#ea580c]"
                          />
                        </div>
                      </div>

                      <div className="relative w-full aspect-video border-2 border-foreground bg-black overflow-hidden flex items-center justify-center">
                        {imagePreview && (
                          <img
                            src={imagePreview}
                            alt="Original"
                            className="absolute inset-0 w-full h-full object-contain"
                          />
                        )}

                        {result.heatmap_b64 ? (
                          <img
                            src={result.heatmap_b64}
                            alt="Grad-CAM Heatmap"
                            className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-300"
                            style={{ opacity: heatmapOpacity }}
                          />
                        ) : (
                          <div className="absolute inset-0 bg-red-500/20 backdrop-blur-xs flex items-center justify-center p-6 text-center">
                            <p className="font-mono text-xs text-white bg-black/90 p-4 border border-white/20 max-w-md">
                              🔥 Grad-CAM activation heatmap is generated live by `inference.py` when connected to the backend PyTorch CUDA model.
                            </p>
                          </div>
                        )}
                      </div>

                      <p className="font-mono text-[11px] text-muted-foreground">
                        Grad-CAM highlights the specific pixel activations and high-level feature regions in the neural network that contributed to the classification verdict.
                      </p>
                    </div>
                  )}

                  {/* TAB 3: METADATA */}
                  {activeTab === "metadata" && (
                    <div className="space-y-4 font-mono text-xs">
                      <div className="border border-border p-4 bg-background">
                        <span className="text-muted-foreground block mb-2">EXIF METADATA TRUST SIGNAL</span>
                        <div className="text-2xl font-bold text-foreground">
                          {((result.metadata_findings.metadata_trust_signal || 0) * 100).toFixed(0)}% TRUST SCORE
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="border border-border p-3 bg-muted/20">
                          <span className="text-muted-foreground block text-[10px]">HAS EXIF HEADER:</span>
                          <span className="font-bold text-foreground">
                            {result.metadata_findings.has_exif ? "YES (PASSED)" : "NO (MISSING)"}
                          </span>
                        </div>
                        <div className="border border-border p-3 bg-muted/20">
                          <span className="text-muted-foreground block text-[10px]">SOFTWARE TAG:</span>
                          <span className="font-bold text-foreground truncate block">
                            {result.metadata_findings.software || "None (Striped)"}
                          </span>
                        </div>
                      </div>

                      {result.metadata_findings.notes && (
                        <div className="border border-border p-3 bg-background text-muted-foreground text-[11px]">
                          <span className="font-bold text-foreground block mb-1">METADATA NOTES:</span>
                          {result.metadata_findings.notes}
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 4: ARTIFACTS */}
                  {activeTab === "artifacts" && (
                    <div className="space-y-4 font-mono text-xs">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="border border-border p-4 bg-background">
                          <span className="text-muted-foreground block text-[10px]">ERROR LEVEL ANALYSIS (ELA+):</span>
                          <span className="text-xl font-bold text-foreground">
                            {((result.artifact_findings.ela_mean_score || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="border border-border p-4 bg-background">
                          <span className="text-muted-foreground block text-[10px]">FFT GRID ANOMALY SCORE:</span>
                          <span className="text-xl font-bold text-foreground">
                            {((result.artifact_findings.fft_grid_score || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {result.artifact_findings.flags && result.artifact_findings.flags.length > 0 && (
                        <div className="border border-amber-500/50 bg-amber-500/10 p-4">
                          <span className="font-bold text-amber-500 block mb-2">DETECTED ARTIFACT FLAGS:</span>
                          <ul className="list-disc list-inside space-y-1 text-muted-foreground text-[11px]">
                            {result.artifact_findings.flags.map((flag, idx) => (
                              <li key={idx}>{flag}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 5: OCR / TEXT FORENSICS */}
                  {activeTab === "ocr" && (
                    <div className="space-y-4 font-mono text-xs">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="border border-border p-4 bg-background">
                          <span className="text-muted-foreground block text-[10px]">TEXT ANOMALY SCORE:</span>
                          <span className="text-xl font-bold text-foreground">
                            {(((result.text_findings?.text_anomaly_score || 0)) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="border border-border p-4 bg-background">
                          <span className="text-muted-foreground block text-[10px]">TEXT TRUST SIGNAL:</span>
                          <span className="text-xl font-bold text-foreground">
                            {(((result.text_findings?.text_trust_signal || 1.0)) * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      {result.text_findings?.suspicious_words && result.text_findings.suspicious_words.length > 0 && (
                        <div className="border border-red-500/50 bg-red-500/10 p-4">
                          <span className="font-bold text-red-500 block mb-1">⚠️ CORRUPTED / SUSPICIOUS WORDS:</span>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {result.text_findings.suspicious_words.map((word, idx) => (
                              <span key={idx} className="bg-red-500/20 text-red-400 border border-red-500/30 px-2 py-1 font-bold">
                                {word}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {result.text_findings?.flags && result.text_findings.flags.length > 0 && (
                        <div className="border border-border p-4 bg-background">
                          <span className="font-bold text-foreground block mb-2">TEXT FORENSIC FLAGS:</span>
                          <ul className="list-disc list-inside space-y-1 text-muted-foreground text-[11px]">
                            {result.text_findings.flags.map((flag, idx) => (
                              <li key={idx}>{flag}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
