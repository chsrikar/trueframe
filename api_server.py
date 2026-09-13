"""
api_server.py — FastAPI backend server for TRUEFRAME AI Image Forensics.
Serves endpoints /health and /analyze with Dual-Domain Inversion & Grad-CAM Fusion.
"""
import io
import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image

from inference import analyze_image, _get_model_and_device

app = FastAPI(
    title="TRUEFRAME AI Forensics API",
    description="Dual-Domain AI image authenticity analysis & Grad-CAM spatial agreement engine",
    version="2.0.0"
)

# Enable CORS for Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pil_to_b64(pil_img: Optional[Image.Image]) -> Optional[str]:
    """Helper to convert PIL Image to base64 data URL string."""
    if pil_img is None:
        return None
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


@app.get("/health")
def health_check():
    try:
        model, device = _get_model_and_device()
        return {
            "status": "healthy",
            "version": "2.0.0 (Dual Grad-CAM & Fusion)",
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model_loaded": model is not None
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "cuda_available": torch.cuda.is_available()
        }


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image")

    temp_path = None
    try:
        # Save uploaded image bytes to a temp file
        extension = Path(file.filename).suffix if file.filename else ".jpg"
        if not extension:
            extension = ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        # Run full TRUEFRAME Dual-Domain analysis pipeline
        result = analyze_image(temp_path)

        # Convert heatmaps to base64 strings
        heatmap_fused_b64 = _pil_to_b64(result.get("heatmap_fused"))
        heatmap_normal_b64 = _pil_to_b64(result.get("heatmap_normal"))
        heatmap_inverted_b64 = _pil_to_b64(result.get("heatmap_inverted"))

        # Serialized response data (OCR removed, Dual Grad-CAM details added)
        response_data = {
            "filename": file.filename,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "trust_score": result["trust_score"],
            "agreement_score": result["agreement_score"],
            "agreement_details": result["agreement_details"],
            "class_probabilities": result["class_probabilities"],
            "inverted_prediction": result.get("inverted_prediction", {}),
            "heatmap_b64": heatmap_fused_b64,  # default alias to fused heatmap
            "heatmap_fused_b64": heatmap_fused_b64,
            "heatmap_normal_b64": heatmap_normal_b64,
            "heatmap_inverted_b64": heatmap_inverted_b64,
            "metadata_findings": result["metadata_findings"],
            "artifact_findings": result["artifact_findings"],
            "fusion_weights_used": result["fusion_weights_used"],
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TRUEFRAME AI FastAPI server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
