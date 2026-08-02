"use client"

import { motion } from "framer-motion"
import { Shield, BrainCircuit, Activity, Fingerprint, Users, Expand, ShieldAlert } from "lucide-react"

const ease = [0.22, 1, 0.36, 1] as const

const strengths = [
  {
    icon: Expand,
    title: "Generalization",
    description: "Built to detect tomorrow's deepfakes, not just today's. Our feature set is generator-agnostic."
  },
  {
    icon: Activity,
    title: "Calibrated Confidence",
    description: "We don't just output a verdict. The Trust Fusion Score includes uncertainty calibration to prevent misleading heatmaps."
  },
  {
    icon: ShieldAlert,
    title: "Adversarial Robustness",
    description: "Hardened against evasion. The pipeline checks for adversarial perturbations and re-compression attacks."
  },
  {
    icon: Fingerprint,
    title: "C2PA Provenance",
    description: "Integrates with emerging content credentials standards as a complementary signal for holistic verification."
  },
  {
    icon: Users,
    title: "Human-in-the-Loop",
    description: "Designed for investigators. Expert reviewers can correct verdicts, feeding back into continuous model fine-tuning."
  },
  {
    icon: BrainCircuit,
    title: "Comprehensive Coverage",
    description: "Detects both AI-generated diffusion fakes and traditional copy-move/splicing forgeries using ELA."
  }
]

export function CoreStrengths() {
  return (
    <section className="w-full px-6 py-20 lg:px-12 bg-muted/50 border-y-2 border-foreground">
      {/* Section label */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5, ease }}
        className="flex items-center gap-4 mb-12"
      >
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          {"// SECTION: CORE_STRENGTHS"}
        </span>
        <div className="flex-1 border-t border-border" />
        <span className="inline-block h-2 w-2 bg-[#ea580c] animate-blink" />
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          007
        </span>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {strengths.map((item, i) => {
          const Icon = item.icon
          return (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1, ease }}
              className="p-6 border-2 border-foreground bg-background hover:bg-muted transition-colors duration-300 group"
            >
              <div className="mb-4 text-[#ea580c]">
                <Icon size={24} strokeWidth={1.5} />
              </div>
              <h3 className="font-mono text-sm font-bold uppercase tracking-wider text-foreground mb-2 group-hover:text-[#ea580c] transition-colors">
                {item.title}
              </h3>
              <p className="text-xs text-muted-foreground font-mono leading-relaxed">
                {item.description}
              </p>
            </motion.div>
          )
        })}
      </div>
    </section>
  )
}
