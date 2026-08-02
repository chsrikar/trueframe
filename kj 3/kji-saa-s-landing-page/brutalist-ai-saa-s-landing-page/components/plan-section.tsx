"use client"

import { motion } from "framer-motion"
import { ArrowRight } from "lucide-react"

const ease = [0.22, 1, 0.36, 1] as const

export function PlanSection() {
  return (
    <section className="w-full px-6 py-20 lg:px-12">
      {/* Section label */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5, ease }}
        className="flex items-center gap-4 mb-8"
      >
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          {"// SECTION: DEVELOPMENT_PLAN"}
        </span>
        <div className="flex-1 border-t border-border" />
        <span className="inline-block h-2 w-2 bg-[#ea580c] animate-blink" />
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          004
        </span>
      </motion.div>

      {/* Section Header */}
      <div className="flex flex-col gap-4 mb-12">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease }}
          className="text-2xl lg:text-4xl font-mono font-bold tracking-tight uppercase"
        >
          Frontend Architecture &
          <br />
          <span className="text-[#ea580c]">API Integration Roadmap</span>
        </motion.h2>

        <p className="text-xs lg:text-sm font-mono text-muted-foreground max-w-2xl leading-relaxed">
          How the TRUEFRAME web application was engineered, styled, and connected to the PyTorch inference backend.
        </p>
      </div>

      {/* 4-Step Roadmap Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-0 border-2 border-foreground mb-12">
        {/* Step 1: UI & Styling */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1, ease }}
          className="flex flex-col p-6 border-b-2 md:border-b-0 md:border-r-2 border-foreground bg-background"
        >
          <span className="text-xs font-mono text-[#ea580c] font-bold mb-2">// PHASE 01</span>
          <h3 className="text-base font-mono font-bold uppercase mb-3">Brutalist UI Foundation</h3>
          <p className="text-xs font-mono text-muted-foreground leading-relaxed">
            Built with Next.js 14, React Server Components, TypeScript, and TailwindCSS. Features high-contrast borders, monospace grid layouts, and Framer Motion micro-animations.
          </p>
        </motion.div>

        {/* Step 2: Interactive Playground */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.2, ease }}
          className="flex flex-col p-6 border-b-2 md:border-b-0 lg:border-r-2 border-foreground bg-background"
        >
          <span className="text-xs font-mono text-[#ea580c] font-bold mb-2">// PHASE 02</span>
          <h3 className="text-base font-mono font-bold uppercase mb-3">Forensics Playground</h3>
          <p className="text-xs font-mono text-muted-foreground leading-relaxed">
            Drag-and-drop image dropzone with instant client-side preview, file type validation, EXIF stripping check, and real-time state management.
          </p>
        </motion.div>

        {/* Step 3: Backend Connection */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3, ease }}
          className="flex flex-col p-6 border-b-2 md:border-b-0 md:border-r-2 border-foreground bg-background"
        >
          <span className="text-xs font-mono text-[#ea580c] font-bold mb-2">// PHASE 03</span>
          <h3 className="text-base font-mono font-bold uppercase mb-3">FastAPI API Bridge</h3>
          <p className="text-xs font-mono text-muted-foreground leading-relaxed">
            Connects to PyTorch via <code className="text-[#ea580c]">POST /api/analyze</code> (FastAPI + Uvicorn). Submits <code className="text-foreground font-bold">multipart/form-data</code> with automatic fallback handling.
          </p>
        </motion.div>

        {/* Step 4: Multi-Signal Visuals */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.4, ease }}
          className="flex flex-col p-6 bg-background"
        >
          <span className="text-xs font-mono text-[#ea580c] font-bold mb-2">// PHASE 04</span>
          <h3 className="text-base font-mono font-bold uppercase mb-3">Explainable Dashboard</h3>
          <p className="text-xs font-mono text-muted-foreground leading-relaxed">
            Renders Trust Score gauge (0–100), EXIF badge alerts, ELA/FFT spectral variance metrics, and interactive Grad-CAM heatmap overlays.
          </p>
        </motion.div>
      </div>

      {/* Integration Code Architecture Diagram Box */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5, delay: 0.5, ease }}
        className="flex flex-col border-2 border-foreground p-6 bg-foreground text-background font-mono text-xs"
      >
        <div className="flex items-center justify-between border-b border-background/20 pb-3 mb-4">
          <span className="text-[10px] tracking-[0.2em] uppercase text-[#ea580c]">
            ARCHITECTURE_FLOW.json
          </span>
          <span className="text-[10px] tracking-[0.2em] uppercase text-background/60">
            HTTP 200 OK
          </span>
        </div>

        <pre className="text-background/90 overflow-x-auto leading-relaxed font-mono">
{`Client Upload  ──(multipart/form-data)──►  FastAPI Backend (api_server.py:8000)
                                                    │
             ┌──────────────────────────────────────┴──────────────────────────────────────┐
             ▼                                      ▼                                      ▼
[#6] EfficientNet-B0 Classifier         Metadata EXIF Parser                   Artifact Signal Engine
 • Model Inference (PyTorch GPU)         (Camera vs AI Signatures)               (ELA+ & 2D FFT Spectrum)
 • Predict Real vs AI-Generated                     │                                      │
 • Output Class Confidence Score                    │                                      │
             │                                      │                                      │
             └──────────────────────────────────────┬──────────────────────────────────────┘
                                                    ▼
                                          Trust Fusion Engine
                                      (Weighted 50 / 25 / 25 Score)
                                                    │
Client UI Dashboard  ◄──(JSON Response + Grad-CAM Heatmap Overlay)──┘`}
        </pre>
      </motion.div>
    </section>
  )
}
