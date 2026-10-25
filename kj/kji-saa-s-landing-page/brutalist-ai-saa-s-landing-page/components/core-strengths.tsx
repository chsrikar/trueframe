"use client"

import { motion } from "framer-motion"
import { Shield, BrainCircuit, Activity, Fingerprint, Users, Expand, ShieldAlert } from "lucide-react"

const ease = [0.22, 1, 0.36, 1] as const

const strengths = [
  {
    icon: Expand,
    title: "Generator Holdout",
    description: "Synthetic generator families are separated across training, validation, and test. This is a limited stress test, not proof of universal generalization."
  },
  {
    icon: Activity,
    title: "Uncalibrated Scores",
    description: "Classifier confidence is raw softmax output. The optional fusion score is heuristic and should not be interpreted as probability or verified authenticity."
  },
  {
    icon: ShieldAlert,
    title: "Research Scope",
    description: "The trained labels cover genuine and AI-generated face images. Manipulated and face-swap examples are outside the current training scope."
  },
  {
    icon: Fingerprint,
    title: "Metadata Signals",
    description: "The demo reads available EXIF and XMP fields. Metadata can be removed or changed and is not proof of provenance."
  },
  {
    icon: Users,
    title: "Explainability",
    description: "Grad-CAM maps show image regions that influence a prediction. They are visual explanations, not localized proof of manipulation."
  },
  {
    icon: BrainCircuit,
    title: "Known Data Gap",
    description: "The AI class currently comes from one dataset family. Independent datasets for both labels are needed to measure cross-dataset performance."
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
