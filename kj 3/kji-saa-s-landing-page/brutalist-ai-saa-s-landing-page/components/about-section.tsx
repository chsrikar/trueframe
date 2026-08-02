"use client"

import { useEffect, useState, useRef } from "react"
import { motion, useInView } from "framer-motion"
import Image from "next/image"

const ease = [0.22, 1, 0.36, 1] as const

/* ── scramble text reveal ── */
function ScrambleText({ text, className }: { text: string; className?: string }) {
  const [display, setDisplay] = useState(text)
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: "-50px" })
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:"

  useEffect(() => {
    if (!inView) return
    let iteration = 0
    const interval = setInterval(() => {
      setDisplay(
        text
          .split("")
          .map((char, i) => {
            if (char === " ") return " "
            if (i < iteration) return text[i]
            return chars[Math.floor(Math.random() * chars.length)]
          })
          .join("")
      )
      iteration += 0.5
      if (iteration >= text.length) {
        setDisplay(text)
        clearInterval(interval)
      }
    }, 30)
    return () => clearInterval(interval)
  }, [inView, text])

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  )
}

/* ── blinking cursor ── */
function BlinkDot() {
  return <span className="inline-block h-2 w-2 bg-[#ea580c] animate-blink" />
}

/* ── live uptime counter ── */
function UptimeCounter() {
  const [seconds, setSeconds] = useState(0)

  useEffect(() => {
    const base = 31536000 + Math.floor(Math.random() * 1000000)
    setSeconds(base)
    const interval = setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => clearInterval(interval)
  }, [])

  const format = (n: number) => {
    const d = Math.floor(n / 86400)
    const h = Math.floor((n % 86400) / 3600)
    const m = Math.floor((n % 3600) / 60)
    const s = n % 60
    return `${d}d ${String(h).padStart(2, "0")}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`
  }

  return (
    <span className="font-mono text-[#ea580c]" style={{ fontVariantNumeric: "tabular-nums" }}>
      {format(seconds)}
    </span>
  )
}

/* ── stat block ── */
const STATS = [
  { label: "BACKBONE", value: "EffNet-B0" },
  { label: "FUSION_RATIO", value: "50/25/25" },
  { label: "TEST_ACCURACY", value: "85.2%" },
  { label: "TRUST_SCORE", value: "0-100" },
]

function StatBlock({ label, value, index }: { label: string; value: string; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-30px" }}
      transition={{ delay: 0.15 + index * 0.08, duration: 0.5, ease }}
      className="flex flex-col gap-1 border-2 border-foreground px-4 py-3"
    >
      <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
        {label}
      </span>
      <span className="text-xl lg:text-2xl font-mono font-bold tracking-tight">
        <ScrambleText text={value} />
      </span>
    </motion.div>
  )
}

/* ── main about section ── */
export function AboutSection() {
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
          {"// SECTION: ABOUT_TRUEFRAME"}
        </span>
        <div className="flex-1 border-t border-border" />
        <BlinkDot />
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          005
        </span>
      </motion.div>

      {/* Two-column layout: Small image, Big text */}
      <div className="flex flex-col lg:flex-row gap-0 border-2 border-foreground">
        {/* Left: Image (Small) */}
        <motion.div
          initial={{ opacity: 0, x: -30, filter: "blur(6px)" }}
          whileInView={{ opacity: 1, x: 0, filter: "blur(0px)" }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.7, ease }}
          className="relative w-full lg:w-1/4 min-h-[220px] lg:min-h-full border-b-2 lg:border-b-0 lg:border-r-2 border-foreground overflow-hidden bg-foreground"
        >
          {/* Image label overlay */}
          <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-3 py-1.5 bg-foreground/80 backdrop-blur-sm">
            <span className="text-[9px] tracking-[0.2em] uppercase text-background/60 font-mono">
              RENDER: iso_infra.obj
            </span>
            <span className="text-[9px] tracking-[0.2em] uppercase text-[#ea580c] font-mono">
              LIVE
            </span>
          </div>

          <Image
            src="/images/about-isometric.jpg"
            alt="Isometric view of AI infrastructure with server racks and data pipelines"
            fill
            className="object-cover"
            sizes="(max-width: 1024px) 100vw, 25vw"
            priority
          />

          {/* Bottom image coordinates */}
          <div className="absolute bottom-0 left-0 right-0 z-10 flex items-center justify-between px-3 py-1.5 bg-foreground/80 backdrop-blur-sm">
            <span className="text-[9px] tracking-[0.2em] uppercase text-background/40 font-mono">
              {"CAM: ISO"}
            </span>
            <span className="text-[9px] tracking-[0.2em] uppercase text-background/40 font-mono">
              {"RES: 2048x2048"}
            </span>
          </div>
        </motion.div>

        {/* Right: Content (Big Text) */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.7, delay: 0.1, ease }}
          className="flex flex-col w-full lg:w-3/4"
        >
          {/* Header bar */}
          <div className="flex items-center justify-between px-6 py-4 border-b-2 border-foreground bg-muted/20">
            <span className="text-xs tracking-[0.2em] uppercase text-muted-foreground font-mono font-bold">
              ROADMAP.md
            </span>
            <span className="text-xs tracking-[0.2em] uppercase text-[#ea580c] font-mono font-bold">
              v1.0.0
            </span>
          </div>

          {/* Content body */}
          <div className="flex-1 flex flex-col justify-between px-6 lg:px-8 py-8 lg:py-10">
            <div className="flex flex-col gap-6">
              <motion.h2
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-30px" }}
                transition={{ duration: 0.5, delay: 0.2, ease }}
                className="text-3xl lg:text-5xl font-mono font-bold tracking-tight uppercase text-balance leading-tight"
              >
                Multi-Modal Pipeline
                <br />
                <span className="text-[#ea580c]">System Architecture</span>
              </motion.h2>

              <motion.div
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-30px" }}
                transition={{ delay: 0.3, duration: 0.5, ease }}
                className="flex flex-col gap-6"
              >
                <p className="text-sm lg:text-base font-mono text-muted-foreground leading-relaxed">
                  TRUEFRAME merges deep learning, metadata forensics, and physical signal analysis into a unified 4-layer architecture:
                </p>

                {/* 4-Layer Architecture Cards Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Layer 1 */}
                  <div className="border-2 border-foreground p-5 bg-background flex flex-col gap-3">
                    <div className="flex items-center justify-between border-b-2 border-foreground pb-2">
                      <span className="text-xs font-mono font-bold text-[#ea580c]">LAYER 01</span>
                      <span className="text-[10px] font-mono text-muted-foreground uppercase font-bold">DEEP LEARNING</span>
                    </div>
                    <h4 className="text-base font-mono font-bold uppercase">CNN Classifier</h4>
                    <ul className="text-xs font-mono text-foreground/90 space-y-2 list-none p-0">
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">Depthwise Separable Convolutions:</strong> Reduces params while preserving spatial frequency representations.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">Squeeze-and-Excitation (SE):</strong> Channel-wise attention focusing on subtle forgery cues across image channels.</span>
                      </li>
                    </ul>
                    <div className="flex flex-wrap gap-1 mt-auto pt-2 border-t border-border">
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">Depthwise Conv</span>
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">SE Attention</span>
                      <span className="text-[9px] font-mono bg-foreground/5 text-foreground px-2 py-0.5 border border-foreground/20 font-bold">EffNet-B0</span>
                    </div>
                  </div>

                  {/* Layer 2 */}
                  <div className="border-2 border-foreground p-5 bg-background flex flex-col gap-3">
                    <div className="flex items-center justify-between border-b-2 border-foreground pb-2">
                      <span className="text-xs font-mono font-bold text-[#ea580c]">LAYER 02</span>
                      <span className="text-[10px] font-mono text-muted-foreground uppercase font-bold">HEADER FORENSICS</span>
                    </div>
                    <h4 className="text-base font-mono font-bold uppercase">Metadata Engine</h4>
                    <ul className="text-xs font-mono text-foreground/90 space-y-2 list-none p-0">
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">EXIF & Camera Hardware:</strong> Parses Make, Model, DateTimeOriginal, & GPS tags to verify genuine origin.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">Software Fingerprints:</strong> Detects ComfyUI, Automatic1111, Midjourney, DALL-E, and Photoshop XMP history.</span>
                      </li>
                    </ul>
                    <div className="flex flex-wrap gap-1 mt-auto pt-2 border-t border-border">
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">EXIF / XMP</span>
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">Software Tags</span>
                      <span className="text-[9px] font-mono bg-foreground/5 text-foreground px-2 py-0.5 border border-foreground/20 font-bold">Camera Fingerprint</span>
                    </div>
                  </div>

                  {/* Layer 3 */}
                  <div className="border-2 border-foreground p-5 bg-background flex flex-col gap-3">
                    <div className="flex items-center justify-between border-b-2 border-foreground pb-2">
                      <span className="text-xs font-mono font-bold text-[#ea580c]">LAYER 03</span>
                      <span className="text-[10px] font-mono text-muted-foreground uppercase font-bold">PHYSICAL SIGNALS</span>
                    </div>
                    <h4 className="text-base font-mono font-bold uppercase">Artifact Analysis</h4>
                    <ul className="text-xs font-mono text-foreground/90 space-y-2 list-none p-0">
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">Error Level Analysis (ELA+):</strong> Measures compression residuals to reveal localized editing & splicing.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">Frequency Domain (FFT):</strong> Computes 2D Fourier spectrum to detect periodic grid spikes from diffusion upscalers.</span>
                      </li>
                    </ul>
                    <div className="flex flex-wrap gap-1 mt-auto pt-2 border-t border-border">
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">ELA Engine</span>
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">2D FFT Spectrum</span>
                      <span className="text-[9px] font-mono bg-foreground/5 text-foreground px-2 py-0.5 border border-foreground/20 font-bold">Grid Detection</span>
                    </div>
                  </div>

                  {/* Layer 4 */}
                  <div className="border-2 border-foreground p-5 bg-background flex flex-col gap-3">
                    <div className="flex items-center justify-between border-b-2 border-foreground pb-2">
                      <span className="text-xs font-mono font-bold text-[#ea580c]">LAYER 04</span>
                      <span className="text-[10px] font-mono text-muted-foreground uppercase font-bold">MULTI-SIGNAL FUSION</span>
                    </div>
                    <h4 className="text-base font-mono font-bold uppercase">Trust Fusion & Grad-CAM</h4>
                    <ul className="text-xs font-mono text-foreground/90 space-y-2 list-none p-0">
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c]">▸</span>
                        <span><strong className="text-foreground">50/25/25 Weighted Fusion:</strong> Combines Classifier (50%), Metadata (25%), and Artifacts (25%) into a 0–100 score.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c]">▸</span>
                        <span><strong className="text-foreground">Grad-CAM Heatmaps:</strong> Generates activation maps highlighting exact image regions driving predictions.</span>
                      </li>
                    </ul>
                    <div className="flex flex-wrap gap-1 mt-auto pt-2 border-t border-border">
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">50+25+25 Fusion</span>
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">Grad-CAM Heatmap</span>
                      <span className="text-[9px] font-mono bg-foreground/5 text-foreground px-2 py-0.5 border border-foreground/20 font-bold">Trust Score</span>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Uptime line */}
              <motion.div
                initial={{ opacity: 0, scaleX: 0.8 }}
                whileInView={{ opacity: 1, scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4, duration: 0.5, ease }}
                style={{ transformOrigin: "left" }}
                className="flex items-center gap-3 py-4 border-t-2 border-b-2 border-foreground text-sm font-mono"
              >
                <span className="h-2 w-2 bg-[#ea580c]" />
                <span className="text-xs tracking-[0.2em] uppercase text-muted-foreground font-bold">
                  UPTIME:
                </span>
                <UptimeCounter />
              </motion.div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-0 mt-8 border-2 border-foreground">
              {STATS.map((stat, i) => (
                <StatBlock key={stat.label} {...stat} index={i} />
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
