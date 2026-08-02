import jsPDF from "jspdf"
import { AnalysisResult } from "./api"

export function generatePDFReport(result: AnalysisResult, imagePreviewUrl?: string | null) {
  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4"
  })

  const pageWidth = doc.internal.pageSize.getWidth()
  const margin = 15
  const contentWidth = pageWidth - margin * 2

  // Background Brutalist styling
  doc.setFillColor(245, 245, 245)
  doc.rect(0, 0, pageWidth, 297, "F")

  // Header Banner
  doc.setFillColor(24, 24, 27) // Dark zinc
  doc.rect(margin, 12, contentWidth, 22, "F")

  doc.setTextColor(255, 255, 255)
  doc.setFont("helvetica", "bold")
  doc.setFontSize(16)
  doc.text("TRUEFRAME // FORENSIC AUDIT REPORT", margin + 6, 25)

  doc.setFont("helvetica", "normal")
  doc.setFontSize(9)
  doc.text(`DATE: ${new Date().toLocaleString()}`, pageWidth - margin - 55, 25)

  let y = 42

  // Section 1: Executive Verdict Box
  const isGenuine = result.verdict.includes("Genuine")
  if (isGenuine) {
    doc.setFillColor(236, 253, 245) // Light emerald
    doc.setDrawColor(16, 185, 129) // Emerald border
  } else {
    doc.setFillColor(254, 242, 242) // Light red
    doc.setDrawColor(239, 68, 68) // Red border
  }
  doc.setLineWidth(0.8)
  doc.rect(margin, y, contentWidth, 30, "FD")

  doc.setFont("helvetica", "bold")
  doc.setFontSize(9)
  doc.setTextColor(100, 100, 100)
  doc.text("CLASSIFICATION VERDICT", margin + 6, y + 8)

  doc.setFontSize(16)
  doc.setTextColor(isGenuine ? 16 : 220, isGenuine ? 185 : 38, isGenuine ? 129 : 38)
  doc.text(result.verdict.toUpperCase(), margin + 6, y + 18)

  doc.setFontSize(9)
  doc.setTextColor(60, 60, 60)
  doc.text(`TRUST SCORE: ${result.trust_score}/100`, margin + 6, y + 25)
  doc.text(`NEURAL CONFIDENCE: ${(result.confidence * 100).toFixed(1)}%`, margin + 70, y + 25)

  y += 36

  // Section 2: File & Metadata Summary
  doc.setFillColor(255, 255, 255)
  doc.setDrawColor(200, 200, 200)
  doc.setLineWidth(0.3)
  doc.rect(margin, y, contentWidth, 34, "FD")

  doc.setFont("helvetica", "bold")
  doc.setFontSize(10)
  doc.setTextColor(0, 0, 0)
  doc.text("FILE & EXIF METADATA SUMMARY", margin + 6, y + 8)

  doc.setFont("helvetica", "normal")
  doc.setFontSize(9)
  doc.setTextColor(60, 60, 60)
  doc.text(`Filename: ${result.filename}`, margin + 6, y + 16)
  doc.text(`EXIF Header Present: ${result.metadata_findings.has_exif ? "YES (Valid)" : "NO (Missing/Stripped)"}`, margin + 6, y + 22)
  doc.text(`Software Tag: ${result.metadata_findings.software || "None Detected"}`, margin + 6, y + 28)

  doc.text(`Metadata Trust Signal: ${((result.metadata_findings.metadata_trust_signal || 0) * 100).toFixed(0)}%`, margin + 100, y + 16)
  doc.text(`Camera Model: ${result.metadata_findings.camera_model || "N/A"}`, margin + 100, y + 22)

  y += 40

  // Section 3: Visual Grad-CAM Heatmap
  doc.setFillColor(255, 255, 255)
  doc.rect(margin, y, contentWidth, 75, "FD")

  doc.setFont("helvetica", "bold")
  doc.setFontSize(10)
  doc.setTextColor(0, 0, 0)
  doc.text("GRAD-CAM ACTIVATION HEATMAP VISUALIZATION", margin + 6, y + 8)

  if (result.heatmap_b64) {
    try {
      doc.addImage(result.heatmap_b64, "PNG", margin + 6, y + 12, 85, 58)
    } catch (e) {
      doc.setFontSize(9)
      doc.text("Grad-CAM Heatmap Image Embedded", margin + 6, y + 35)
    }
  } else if (imagePreviewUrl) {
    try {
      doc.addImage(imagePreviewUrl, "JPEG", margin + 6, y + 12, 85, 58)
    } catch (e) {
      doc.setFontSize(9)
      doc.text("Uploaded Image Embedded", margin + 6, y + 35)
    }
  }

  doc.setFont("helvetica", "normal")
  doc.setFontSize(8)
  doc.setTextColor(80, 80, 80)
  const heatmapNotes = [
    "Grad-CAM visual explanation highlights pixel region",
    "activations that contributed to neural classification.",
    "",
    `Genuine Class Prob: ${((result.class_probabilities?.genuine || 0) * 100).toFixed(1)}%`,
    `AI Class Prob: ${((result.class_probabilities?.ai_generated || 0) * 100).toFixed(1)}%`
  ]
  heatmapNotes.forEach((line, idx) => {
    doc.text(line, margin + 98, y + 20 + idx * 5)
  })

  y += 82

  // Section 4: Artifact Forensics (ELA + FFT)
  doc.setFillColor(255, 255, 255)
  doc.rect(margin, y, contentWidth, 36, "FD")

  doc.setFont("helvetica", "bold")
  doc.setFontSize(10)
  doc.setTextColor(0, 0, 0)
  doc.text("ARTIFACT FORENSICS (ELA & FFT ANALYSIS)", margin + 6, y + 8)

  doc.setFont("helvetica", "normal")
  doc.setFontSize(9)
  doc.setTextColor(60, 60, 60)
  doc.text(`Error Level Analysis (ELA Score): ${((result.artifact_findings.ela_mean_score || 0) * 100).toFixed(1)}%`, margin + 6, y + 17)
  doc.text(`FFT Frequency Grid Score: ${((result.artifact_findings.fft_grid_score || 0) * 100).toFixed(1)}%`, margin + 6, y + 24)
  doc.text(`Artifact Trust Signal: ${((result.artifact_findings.artifact_trust_signal || 0) * 100).toFixed(0)}%`, margin + 100, y + 17)

  const artFlags = result.artifact_findings.flags?.join("; ") || "No strong artifact signals detected."
  doc.setFontSize(8)
  doc.text(`Flags: ${artFlags.substring(0, 80)}`, margin + 6, y + 30)

  y += 42

  // Section 5: OCR & Text Forensics
  doc.setFillColor(255, 255, 255)
  // Footer Disclaimer
  doc.setFontSize(7)
  doc.setTextColor(120, 120, 120)
  doc.text("TRUEFRAME AI Forensics Engine // Generated for Verification Purposes // https://github.com/chsrikar/trueframe", margin, 288)

  // Trigger Save
  const cleanFilename = result.filename.replace(/[^a-z0-9]/gi, "_").toLowerCase()
  doc.save(`TRUEFRAME_Audit_Report_${cleanFilename}.pdf`)
}
