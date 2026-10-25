"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"

export interface PipelineNode {
  id: number
  title: string
  x: number
  y: number
  width?: number
  height?: number
  subpoints: string[]
}

const NODES: PipelineNode[] = [
  {
    id: 1,
    title: "1. Upload received",
    x: 140,
    y: 90,
    width: 220,
    height: 90,
    subpoints: [
      "Image upload to local API",
      "Request handled by analysis service",
      "No authenticity claim is made"
    ]
  },
  {
    id: 2,
    title: "2. File validation",
    x: 400,
    y: 90,
    width: 220,
    height: 90,
    subpoints: [
      "Pillow image decode check",
      "15 MB upload and pixel-count limit",
      "Unsupported or unreadable files rejected"
    ]
  },
  {
    id: 3,
    title: "3. Pre-processing",
    x: 660,
    y: 90,
    width: 220,
    height: 90,
    subpoints: [
      "Resize and normalize RGB input",
      "Create an inverted image pass",
      "Prepare model tensors"
    ]
  },
  {
    id: 4,
    title: "4. Metadata extraction",
    x: 920,
    y: 90,
    width: 230,
    height: 90,
    subpoints: [
      "Read available EXIF/XMP fields",
      "Inspect camera/software metadata",
      "Missing metadata is inconclusive"
    ]
  },
  {
    id: 5,
    title: "5. Metadata consistency check",
    x: 920,
    y: 250,
    width: 240,
    height: 90,
    subpoints: [
      "Produce a heuristic metadata signal",
      "Look for known software strings",
      "Metadata can be edited or removed"
    ]
  },
  {
    id: 6,
    title: "6. Model inference (classifier)",
    x: 630,
    y: 250,
    width: 240,
    height: 90,
    subpoints: [
      "EfficientNet-B0 binary classifier",
      "Classes: genuine / AI-generated faces",
      "Raw softmax is not calibrated"
    ]
  },
  {
    id: 7,
    title: "7. Forensic sub-modules",
    x: 340,
    y: 250,
    width: 240,
    height: 90,
    subpoints: [
      "Exploratory ELA summary",
      "Exploratory FFT summary",
      "Signals are not validated verdicts"
    ]
  },
  {
    id: 8,
    title: "8. Explainability (Grad-CAM)",
    x: 180,
    y: 410,
    width: 240,
    height: 90,
    subpoints: [
      "Generate Grad-CAM visualization",
      "Show regions influencing the classifier",
      "Heatmap is not proof of manipulation"
    ]
  },
  {
    id: 9,
    title: "9. Trust Fusion / scoring",
    x: 480,
    y: 410,
    width: 240,
    height: 90,
    subpoints: [
      "Combine fixed heuristic signals",
      "Score has no calibration study",
      "Output is not a probability"
    ]
  },
  {
    id: 10,
    title: "10. Report generation & display",
    x: 780,
    y: 410,
    width: 250,
    height: 90,
    subpoints: [
      "Return classifier output and diagnostics",
      "Render the interactive dashboard",
      "Offer a downloadable demo report"
    ]
  }
]

const PATHS = [
  "M 250 90 L 290 90",
  "M 510 90 L 550 90",
  "M 770 90 L 805 90",
  "M 1035 90 L 1070 90 L 1070 250 L 1040 250",
  "M 800 250 L 750 250",
  "M 510 250 L 460 250",
  "M 220 250 L 60 250 L 60 410 L 60 410",
  "M 300 410 L 360 410",
  "M 600 410 L 655 410",
  "M 905 410 L 1070 410 L 1070 540 L 400 540 L 400 575"
]

const MAIN_COMET_PATH =
  "M 140 90 L 400 90 L 660 90 L 920 90 L 1070 90 L 1070 250 L 920 250 L 630 250 L 340 250 L 60 250 L 60 410 L 180 410 L 480 410 L 780 410 L 1070 410 L 1070 540 L 400 540 L 400 620"

export function WorkflowDiagram() {
  const [mounted, setMounted] = useState(false)
  const [activeNodeId, setActiveNodeId] = useState<number | null>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <div className="h-[750px] w-full" />
  }

  return (
    <div className="relative w-full max-w-[1400px] mx-auto overflow-visible py-4 font-mono">
      <svg
        viewBox="0 0 1180 720"
        className="w-full h-auto overflow-visible"
        role="img"
        aria-label="Interactive Node Graph Architecture Diagram for TRUEFRAME"
      >
        {/* Base Static Flow Paths */}
        {PATHS.map((d, i) => (
          <motion.path
            key={`path-${i}`}
            d={d}
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth={2}
            initial={{ pathLength: 0, opacity: 0 }}
            whileInView={{ pathLength: 1, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.1 + i * 0.08, ease: "easeInOut" }}
          />
        ))}

        {/* Animated Glowing Orange Comet */}
        <motion.path
          d={MAIN_COMET_PATH}
          fill="none"
          stroke="#ea580c"
          strokeWidth={3.5}
          strokeLinecap="round"
          initial={{ pathLength: 0, pathOffset: 0, opacity: 0 }}
          animate={{
            pathLength: [0, 0.06, 0.06, 0],
            pathOffset: [0, 0, 0.94, 1],
            opacity: [0, 1, 1, 0]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
          style={{ filter: "drop-shadow(0 0 8px #ea580c)" }}
        />

        {/* 10 SVG Node Boxes with HTML Subpoints */}
        {NODES.map((node, i) => {
          const isActive = activeNodeId === node.id
          const w = node.width || 220
          const h = node.height || 90
          const leftX = node.x - w / 2
          const topY = node.y - h / 2

          return (
            <motion.g
              key={node.id}
              initial={{ opacity: 0, scale: 0.85 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.2 + i * 0.08, type: "spring", bounce: 0.3 }}
              onMouseEnter={() => setActiveNodeId(node.id)}
              onMouseLeave={() => setActiveNodeId(null)}
              className="cursor-pointer"
            >
              <foreignObject x={leftX} y={topY} width={w} height={h}>
                <div
                  className={`w-full h-full border-2 p-2.5 flex flex-col justify-between transition-all duration-200 shadow-sm ${
                    isActive
                      ? "border-[#ea580c] bg-[#ea580c]/10 shadow-[4px_4px_0px_0px_rgba(234,88,12,1)] scale-102"
                      : "border-foreground bg-card hover:border-[#ea580c]"
                  }`}
                >
                  {/* Node Title */}
                  <div className="flex items-center justify-between border-b border-border pb-1 mb-1">
                    <span className="text-[11px] font-bold text-foreground truncate block">
                      {node.title}
                    </span>
                    <span className="text-[9px] font-bold text-[#ea580c] bg-[#ea580c]/10 px-1 rounded">
                      #{node.id}
                    </span>
                  </div>

                  {/* Subpoints List */}
                  <ul className="space-y-0.5 text-[9.5px] text-muted-foreground leading-tight">
                    {node.subpoints.map((sp, idx) => (
                      <li key={idx} className="truncate flex items-center gap-1">
                        <span className="text-[#ea580c] text-[8px] font-bold">•</span>
                        <span>{sp}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </foreignObject>
            </motion.g>
          )
        })}

        {/* Large Final Output Box */}
        <motion.g
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 1.2 }}
        >
          <foreignObject x={230} y={570} width={340} height={125}>
            <div className="w-full h-full border-2 border-foreground bg-muted p-4 flex flex-col justify-center shadow-lg">
              <h3 className="font-pixel text-lg text-foreground mb-1 uppercase tracking-wide">OUTPUT</h3>
              <h2 className="font-mono text-2xl text-[#ea580c] font-bold mb-1 uppercase tracking-tighter">HEAT MAP & REPORT</h2>
              <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-widest border-t border-border pt-2">RESEARCH DEMO OUTPUT</p>
            </div>
          </foreignObject>
        </motion.g>

        {/* Metric Badges */}
        {[
          { label: "Test accuracy", value: "PENDING", x: 630, y: 570 },
          { label: "Macro F1", value: "PENDING", x: 790, y: 570 },
          { label: "Test ROC-AUC", value: "PENDING", x: 630, y: 635 },
          { label: "Calibration", value: "N/A", x: 790, y: 635 },
        ].map((metric, i) => (
          <motion.g
            key={metric.label}
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: 1.4 + i * 0.1 }}
          >
            <foreignObject x={metric.x} y={metric.y} width={140} height={60}>
              <div className="w-full h-full border border-border flex flex-col items-center justify-center bg-background/50 hover:bg-muted transition-colors shadow-sm">
                <span className="text-[9px] text-muted-foreground font-mono uppercase tracking-widest">{metric.label}</span>
                <span className="text-base text-foreground font-mono font-bold mt-0.5">{metric.value}</span>
              </div>
            </foreignObject>
          </motion.g>
        ))}
      </svg>
    </div>
  )
}
