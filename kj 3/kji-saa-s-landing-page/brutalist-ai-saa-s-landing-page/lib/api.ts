export interface AnalysisResult {
  filename: string
  verdict: "Genuine (Real)" | "AI-Generated" | "Manipulated / Synthetic"
  confidence: number
  trust_score: number
  class_probabilities: {
    genuine: number
    ai_generated: number
    [key: string]: number
  }
  heatmap_b64?: string | null
  metadata_findings: {
    metadata_trust_signal: number
    has_exif: boolean
    software?: string | null
    camera_make?: string | null
    camera_model?: string | null
    notes?: string
    [key: string]: any
  }
  artifact_findings: {
    artifact_trust_signal: number
    ela_mean_score: number
    fft_grid_score: number
    flags?: string[]
    [key: string]: any
  }
  text_findings?: {
    text_detected?: boolean
    detected_words?: string[]
    suspicious_words?: string[]
    text_anomaly_score?: number
    text_trust_signal?: number
    flags?: string[]
    [key: string]: any
  }
  fusion_weights_used?: Record<string, number>
}

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" })
    if (!res.ok) return false
    const data = await res.json()
    return data.status === "healthy"
  } catch (err) {
    return false
  }
}

export async function analyzeImageWithAPI(file: File): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${BACKEND_URL}/analyze`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `Analysis failed with status ${response.status}`)
  }

  return await response.json()
}

export const MOCK_SAMPLES: Record<string, AnalysisResult & { previewUrl: string }> = {
  genuine_portrait: {
    filename: "sample_raw_camera_portrait.jpg",
    previewUrl: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=800&auto=format&fit=crop",
    verdict: "Genuine (Real)",
    confidence: 0.9842,
    trust_score: 96,
    class_probabilities: {
      genuine: 0.9842,
      ai_generated: 0.0158,
    },
    heatmap_b64: null,
    metadata_findings: {
      metadata_trust_signal: 0.95,
      has_exif: true,
      camera_make: "Canon",
      camera_model: "Canon EOS R5",
      software: "Adobe Lightroom 12.0",
      notes: "Authentic EXIF tags verified. Unmodified sensor noise spectrum detected.",
    },
    artifact_findings: {
      artifact_trust_signal: 0.94,
      ela_mean_score: 0.12,
      fft_grid_score: 0.04,
      flags: [],
    },
  },
  ai_synthetic_art: {
    filename: "sample_midjourney_v6_cyberpunk.png",
    previewUrl: "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=800&auto=format&fit=crop",
    verdict: "AI-Generated",
    confidence: 0.9961,
    trust_score: 12,
    class_probabilities: {
      genuine: 0.0039,
      ai_generated: 0.9961,
    },
    heatmap_b64: null,
    metadata_findings: {
      metadata_trust_signal: 0.15,
      has_exif: false,
      camera_make: null,
      camera_model: null,
      software: null,
      notes: "Missing EXIF header tags. Synthetic resolution grid pattern identified.",
    },
    artifact_findings: {
      artifact_trust_signal: 0.10,
      ela_mean_score: 0.88,
      fft_grid_score: 0.92,
      flags: [
        "High-frequency periodic FFT lattice anomaly detected",
        "Uniform ELA error distribution characteristic of AI diffusion models",
      ],
    },
  },
  manipulated_face: {
    filename: "sample_deepfake_face_swap.jpg",
    previewUrl: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=800&auto=format&fit=crop",
    verdict: "AI-Generated",
    confidence: 0.9418,
    trust_score: 28,
    class_probabilities: {
      genuine: 0.0582,
      ai_generated: 0.9418,
    },
    heatmap_b64: null,
    metadata_findings: {
      metadata_trust_signal: 0.40,
      has_exif: true,
      software: "Photoshop 2024 / FaceFusion v2",
      notes: "EXIF software header indicates post-processing editor. Boundary blend artifacts found.",
    },
    artifact_findings: {
      artifact_trust_signal: 0.25,
      ela_mean_score: 0.74,
      fft_grid_score: 0.65,
      flags: [
        "Facial region boundary ELA compression disparity",
        "Splicing artifact detected around eye & jaw contours",
      ],
    },
  },
}
