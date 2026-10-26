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

/* ── stat block ── */
const STATS = [
  { label: "BACKBONE", value: "EffNet-B0" },
  { label: "FUSION_WEIGHTS", value: "HEURISTIC" },
  { label: "LEGACY_ACCURACY", value: "85.2%" },
  { label: "LEGACY_MACRO_F1", value: "0.850" },
  { label: "LEGACY_ROC_AUC", value: "0.912" },
  { label: "CALIBRATION", value: "N/A" },
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

const CV_MATHS = [
  {
    title: "Convolution",
    formula: "y(i,j) = Σₘ Σₙ x(i−m,j−n) k(m,n)",
    explanation: "A learned filter slides over image pixels and responds to local patterns. EfficientNet uses convolutional layers to build features for the classifier.",
  },
  {
    title: "Softmax scores",
    formula: "pₖ = eᶻᵏ / Σⱼ eᶻʲ",
    explanation: "The two model outputs become relative scores for genuine and AI-generated. In this project they are raw, uncalibrated scores—not guaranteed probabilities.",
  },
  {
    title: "Accuracy, precision, recall & F1",
    formula: "Accuracy = (TP + TN) / N\nPrecision = TP / (TP + FP)\nRecall = TP / (TP + FN)\nF1 = 2PR / (P + R)",
    explanation: "Compare predictions with known labels. Here, AI-generated is the positive class; these metrics are meaningful only on an appropriately independent labeled test set.",
  },
  {
    title: "ROC-AUC",
    formula: "AUC = ∫₀¹ TPR(FPR) dFPR",
    explanation: "Across decision thresholds, compare the true-positive rate with the false-positive rate. The area summarizes how well scores rank the two classes.",
  },
  {
    title: "2D Fourier transform (FFT)",
    formula: "F(u,v) = Σₓ Σᵧ I(x,y)e⁻²πi(ux/W + vy/H)",
    explanation: "Re-express image variation as spatial frequencies. TRUEFRAME summarizes this spectrum as an exploratory signal; it is not proof that an image is generated.",
  },
  {
    title: "Grad-CAM",
    formula: "αₖᶜ = (1/Z) Σᵢ Σⱼ ∂yᶜ/∂Aₖᵢⱼ\nLᶜ = ReLU(Σₖ αₖᶜ Aₖ)",
    explanation: "Gradients weight CNN feature maps to make a heatmap of regions influencing a selected class score. A hotspot is a visualization, not a causal explanation.",
  },
  {
    title: "Error Level Analysis (ELA)",
    formula: "D(x,y) = |I(x,y) − JPEG₇₅(I)(x,y)|",
    explanation: "Compare the image with a JPEG-recompressed copy. TRUEFRAME displays the differences; compression, saving history, and edits can all affect them.",
  },
]

function MathsCard({ title, formula, explanation, wide = false }: {
  title: string
  formula: string
  explanation: string
  wide?: boolean
}) {
  return (
    <article className={`border-2 border-foreground bg-background p-5 ${wide ? "md:col-span-2" : ""}`}>
      <h3 className="mb-3 font-mono text-sm font-bold uppercase text-[#ea580c]">{title}</h3>
      <pre className="mb-4 whitespace-pre-wrap break-words border-y border-border py-3 font-mono text-sm leading-relaxed text-foreground">
        {formula}
      </pre>
      <p className="font-mono text-xs leading-relaxed text-muted-foreground">{explanation}</p>
    </article>
  )
}

export function MathsSection() {
  return (
    <section id="maths-in-cv" className="w-full px-6 py-20 lg:px-12">
      <div className="mb-8 flex items-center gap-4">
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          {"// SECTION: MATHS_IN_COMPUTER_VISION"}
        </span>
        <div className="flex-1 border-t border-border" />
        <span className="font-mono text-[10px] tracking-[0.2em] text-[#ea580c]">006</span>
      </div>
      <div className="mb-8 max-w-3xl">
        <h2 className="mb-4 font-mono text-3xl font-bold uppercase leading-tight tracking-tight lg:text-5xl">
          Maths in <span className="text-[#ea580c]">Computer Vision</span>
        </h2>
        <p className="font-mono text-sm leading-relaxed text-muted-foreground lg:text-base">
          The equations behind TRUEFRAME’s image features, class scores, evaluation metrics, and exploratory visualizations—explained in the context of this project.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {CV_MATHS.map((item, index) => (
          <MathsCard key={item.title} {...item} wide={index === CV_MATHS.length - 1} />
        ))}
      </div>
      <p className="mt-5 border-l-2 border-[#ea580c] pl-4 font-mono text-xs leading-relaxed text-muted-foreground">
        Evaluation formulas require independent labeled examples for a valid test result. The current 17-image demo reuses its examples across splits, so its replay scores do not measure performance on unseen images.
      </p>
    </section>
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
              STATIC DEMO
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
                Research Pipeline
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
                  TRUEFRAME classifies face images as genuine or AI-generated. Metadata, ELA/FFT summaries, and Grad-CAM are exploratory diagnostics, not validated authenticity tests:
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
                        <span><strong className="text-foreground">EfficientNet-B0:</strong> Image backbone with a binary head for genuine vs AI-generated face images.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">Scope:</strong> This dataset does not train a manipulated/deepfake class or establish cross-dataset reliability.</span>
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
                        <span><strong className="text-foreground">Available metadata:</strong> Reads selected EXIF/XMP fields when present; this does not verify image origin.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">Limit:</strong> Metadata may be absent, edited, or removed, so its absence is inconclusive.</span>
                      </li>
                    </ul>
                    <div className="flex flex-wrap gap-1 mt-auto pt-2 border-t border-border">
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">EXIF / XMP</span>
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">Software Tags</span>
                      <span className="text-[9px] font-mono bg-foreground/5 text-foreground px-2 py-0.5 border border-foreground/20 font-bold">Not provenance</span>
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
                        <span><strong className="text-foreground">ELA summary:</strong> Shows recompression differences; these do not prove editing or splicing.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c] font-bold">▸</span>
                        <span><strong className="text-foreground">FFT summary:</strong> Experimental frequency features; not validated for reliable generator detection.</span>
                      </li>
                    </ul>
                    <div className="flex flex-wrap gap-1 mt-auto pt-2 border-t border-border">
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">ELA Engine</span>
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">2D FFT Spectrum</span>
                      <span className="text-[9px] font-mono bg-foreground/5 text-foreground px-2 py-0.5 border border-foreground/20 font-bold">Experimental</span>
                    </div>
                  </div>

                  {/* Layer 4 */}
                  <div className="border-2 border-foreground p-5 bg-background flex flex-col gap-3">
                    <div className="flex items-center justify-between border-b-2 border-foreground pb-2">
                      <span className="text-xs font-mono font-bold text-[#ea580c]">LAYER 04</span>
                      <span className="text-[10px] font-mono text-muted-foreground uppercase font-bold">EXPLORATORY OUTPUTS</span>
                    </div>
                    <h4 className="text-base font-mono font-bold uppercase">Heuristic Summary & Grad-CAM</h4>
                    <ul className="text-xs font-mono text-foreground/90 space-y-2 list-none p-0">
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c]">▸</span>
                        <span><strong className="text-foreground">Uncalibrated heuristic:</strong> Experimental fixed weights; not a probability, confidence, or validated trust score.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-[#ea580c]">▸</span>
                        <span><strong className="text-foreground">Grad-CAM:</strong> Activation visualization can help inspect model behavior; it is not proof of manipulation.</span>
                      </li>
                    </ul>
                    <div className="flex flex-wrap gap-1 mt-auto pt-2 border-t border-border">
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">50+25+25 Fusion</span>
                      <span className="text-[9px] font-mono bg-[#ea580c]/10 text-[#ea580c] px-2 py-0.5 border border-[#ea580c]/30 font-bold">Grad-CAM Heatmap</span>
                      <span className="text-[9px] font-mono bg-foreground/5 text-foreground px-2 py-0.5 border border-foreground/20 font-bold">Heuristic index</span>
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
                  SERVICE UPTIME:
                </span>
                <span className="font-mono text-[#ea580c]">NOT MEASURED</span>
              </motion.div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-0 mt-8 border-2 border-foreground">
              {STATS.map((stat, i) => (
                <StatBlock key={stat.label} {...stat} index={i} />
              ))}
            </div>
            <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground font-mono">
              Historical metrics from an earlier random-split run on 11,324 images; not results for the current 17-image demo or generator-held-out evaluation.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
