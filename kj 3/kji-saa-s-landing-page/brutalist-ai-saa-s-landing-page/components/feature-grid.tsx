"use client"

import { motion } from "framer-motion"
import { Shield, Cpu, Binary, Code, UserCheck, Terminal } from "lucide-react"

const ease = [0.22, 1, 0.36, 1] as const

const TEAM_MEMBERS = [
  {
    id: "srikar",
    name: "CHINTHAPENTA SRIKAR",
    role: "Research & Model Architect",
    badge: "ARCHITECTURE",
    icon: Cpu,
    bio: "Engineered the EfficientNet-B0 2-phase fine-tuning pipeline, custom PyTorch datasets, and Trust Fusion multi-modal scoring engine.",
    skills: ["PyTorch", "EfficientNet-B0", "Trust Fusion", "Grad-CAM"],
  },
  {
    id: "kalidas",
    name: "KALIDAS KJ",
    role: "Full Stack & UI Engineer",
    badge: "FULL-STACK",
    icon: Code,
    bio: "Built the Brutalist AI web application, interactive Forensics Playground, real-time heatmap renderer, and FastAPI server API bridge.",
    skills: ["Next.js 14", "TypeScript", "TailwindCSS", "FastAPI"],
  },
  {
    id: "abu",
    name: "ABOO SHAMAR",
    role: "UI/UX & Visual Designer",
    badge: "DESIGNING",
    icon: Binary,
    bio: "Crafted the neo-brutalist high-contrast visual identity, interactive dashboard layout, typography grid, and user flow aesthetics.",
    skills: ["Brutalist UI", "Framer Motion", "Visual Design", "UX Architecture"],
  },
  {
    id: "anto",
    name: "ANTO GEROM T",
    role: "Forensics & Artifacts Engineer",
    badge: "FORENSICS",
    icon: Shield,
    bio: "Developed EXIF metadata extraction, ELA+ compression analysis, and 2D FFT spectral frequency peak detection algorithms.",
    skills: ["Metadata EXIF", "ELA+ Forensics", "FFT Spectrum", "Signal Fusion"],
  },
]

export function FeatureGrid() {
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
          {"// SECTION: TEAM_MEMBERS"}
        </span>
        <div className="flex-1 border-t border-border" />
        <span className="inline-block h-2 w-2 bg-[#ea580c] animate-blink" />
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">003</span>
      </motion.div>

      {/* Header */}
      <div className="mb-12">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease }}
          className="text-3xl lg:text-5xl font-mono font-bold tracking-tight uppercase"
        >
          TEAM <span className="text-[#ea580c]">TRUEFRAME</span>
        </motion.h2>
      </div>

      {/* Team Grid (Big 2x2 Card Grid) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-0 border-2 border-foreground">
        {TEAM_MEMBERS.map((member, i) => {
          const Icon = member.icon
          const isRight = i % 2 === 1
          const isBottom = i >= 2

          return (
            <motion.div
              key={member.id}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12, duration: 0.5, ease }}
              className={`flex flex-col p-8 lg:p-10 min-h-[300px] bg-background ${
                !isRight ? "md:border-r-2 border-foreground" : ""
              } ${!isBottom ? "border-b-2 border-foreground" : ""}`}
            >
              {/* Header */}
              <div className="flex items-center justify-between border-b-2 border-foreground pb-5 mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-[#ea580c] text-background">
                    <Icon size={20} strokeWidth={2.5} />
                  </div>
                  <span className="text-xs font-mono tracking-[0.2em] text-[#ea580c] uppercase font-bold">
                    {member.badge}
                  </span>
                </div>
                <span className="text-xs font-mono opacity-50 font-bold">0{i + 1}</span>
              </div>

              {/* Title & Role */}
              <h3 className="text-2xl lg:text-3xl font-mono font-bold uppercase tracking-tight mb-2">
                {member.name}
              </h3>
              <span className="text-xs lg:text-sm font-mono text-muted-foreground uppercase tracking-wider mb-8 font-semibold">
                {member.role}
              </span>

              {/* Skills badges */}
              <div className="mt-auto pt-5 border-t-2 border-foreground/10 flex flex-wrap gap-2">
                {member.skills.map((skill) => (
                  <span
                    key={skill}
                    className="text-xs font-mono uppercase bg-foreground/5 text-foreground px-3 py-1 border-2 border-foreground/20 font-bold"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </motion.div>
          )
        })}
      </div>
    </section>
  )
}
