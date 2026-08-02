"use client"

import { Cpu } from "lucide-react"
import { motion } from "framer-motion"
import { ThemeToggle } from "@/components/theme-toggle"

export function Navbar() {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="w-full px-6 pt-6 lg:px-12 lg:pt-8"
    >
      <div className="flex items-center justify-between">
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          className="flex items-center gap-3.5"
        >
          <Cpu size={26} strokeWidth={1.8} className="text-[#ea580c]" />
          <span className="text-lg sm:text-xl font-mono tracking-[0.2em] uppercase font-extrabold text-foreground">
            TRUEFRAME
          </span>
        </motion.div>

        {/* Right side: ThemeToggle only */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="flex items-center gap-4"
        >
          <ThemeToggle />
        </motion.div>
      </div>
    </motion.div>
  )
}
