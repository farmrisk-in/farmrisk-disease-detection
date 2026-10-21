"""
Crop Disease Detection & Advisory Backend.
Provides both FastAPI REST API endpoints (for custom frontends) and an interactive Gradio UI.

Environment variables:
    MODEL_ID        HF repo or local folder (default: vegadurmin/crop-disease-deit-small)
    KB_DIR          folder with disease_kb.json, chunks.jsonl, coverage.csv (default: ./kb)
    GEMINI_API_KEY  Gemini API key (optional, for LLM advisory & grounding)
    GEMINI_MODEL    default: gemini-3.1-flash-lite
    FILL_MODE       how to fill sections missing from KB: ai | search | auto | off (default: ai)
    HF_TOKEN        Hugging Face access token (if model repo is private)
    HOST            Server host (default: 0.0.0.0)
    PORT            Server port (default: 7860)
    LOW_CONFIDENCE_THRESHOLD Minimum threshold for confident prediction (default: 0.50)
"""
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
import torch
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModelForImageClassification
import uvicorn

from rag import Advisor, KnowledgeBase, split_sections

# ---- configuration -----------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_ID = os.getenv("MODEL_ID", "vegadurmin/crop-disease-deit-small")
KB_DIR = os.getenv("KB_DIR", str(APP_DIR / "kb"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FILL_MODE = os.getenv("FILL_MODE", "ai")  # off | ai | search | auto
HF_TOKEN = os.getenv("HF_TOKEN")
LOW_CONF = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.50"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))

# ---- classifier setup --------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] Loading model {MODEL_ID} on device {device}...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device).eval()
side = model.config.image_size
tf = T.Compose([
    T.Resize((side, side)),
    T.ToTensor(),
    T.Normalize(processor.image_mean, processor.image_std),
])

# ---- knowledge base & advisor setup ------------------------------------------
kb = KnowledgeBase(KB_DIR)
advisor = Advisor(kb, GEMINI_MODEL, fill_mode=FILL_MODE)

print("\n[map] model class -> KB entry mapping:")
for i in range(model.config.num_labels):
    lab = model.config.id2label[i]
    print(f"  {lab:40s} -> {kb.resolve(lab)}")
print()


# ---- helper functions --------------------------------------------------------
def is_healthy(label: str) -> bool:
    """Return True if the label denotes a healthy plant."""
    return "healthy" in label.lower()


def extract_crop_name(label: str) -> str:
    """Extract crop name from label (e.g., 'Cotton_Aphid' -> 'Cotton')."""
    if label.startswith("Bell_Pepper"):
        return "Bell Pepper"
    if "_" in label:
        return label.split("_")[0]
    return "General"


def classify_image(image: Image.Image) -> Dict[str, float]:
    """Run model inference on a PIL image and return all class probabilities."""
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(pixel_values=x).logits.softmax(-1)[0].cpu()
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


def get_advisory_payload(label: str) -> Dict[str, Any]:
    """Fetch structured advisory for a label."""
    if not label:
        return {
            "disease": "",
            "is_healthy": False,
            "kb_key": None,
            "markdown": "",
            "sections": {},
            "context_markdown": "",
            "search_html": "",
        }

    kb_key = kb.resolve(label)
    if is_healthy(label):
        healthy_md = (
            f"## {label}\n\nNo disease or pest detected. "
            "Keep monitoring the crop regularly."
        )
        return {
            "disease": label,
            "is_healthy": True,
            "kb_key": kb_key,
            "markdown": healthy_md,
            "sections": {"Status": "Healthy plant. No treatment needed."},
            "context_markdown": "",
            "search_html": "",
        }

    md, ctx, search_html = advisor.advise(label)
    sections = split_sections(md)
    return {
        "disease": label,
        "is_healthy": False,
        "kb_key": kb_key,
        "markdown": md,
        "sections": sections,
        "context_markdown": ctx,
        "search_html": search_html,
    }


def analyze_image_full(image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    """Perform both classification and advisory generation."""
    scores = classify_image(image)
    sorted_classes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_prob = sorted_classes[0]

    predictions = [
        {
            "label": label,
            "confidence": round(prob, 4),
            "confidence_pct": f"{prob * 100:.1f}%",
            "crop": extract_crop_name(label),
            "is_healthy": is_healthy(label),
            "has_kb_entry": kb.resolve(label) is not None,
            "kb_key": kb.resolve(label),
        }
        for label, prob in sorted_classes[:top_k]
    ]

    is_low_confidence = top_prob < LOW_CONF
    note = (
        f"Low confidence ({top_prob:.0%}). The image may be unclear or belong to an untrained class."
        if is_low_confidence
        else f"Prediction: {top_label} ({top_prob:.0%})"
    )

    advisory_data = get_advisory_payload(top_label)

    return {
        "prediction": {
            "top_prediction": top_label,
            "confidence": round(top_prob, 4),
            "confidence_pct": f"{top_prob * 100:.1f}%",
            "is_healthy": is_healthy(top_label),
            "is_low_confidence": is_low_confidence,
            "threshold": LOW_CONF,
            "note": note,
            "top_k": predictions,
        },
        "advisory": advisory_data,
    }


# ---- Pydantic Schemas for API Documentation ----------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    model_id: str
    device: str
    total_classes: int
    gemini_configured: bool
    fill_mode: str


class InfoResponse(BaseModel):
    model_id: str
    num_classes: int
    low_confidence_threshold: float
    gemini_model: str
    fill_mode: str
    supported_crops: List[str]


class PredictionItem(BaseModel):
    label: str
    confidence: float
    confidence_pct: str
    crop: str
    is_healthy: bool
    has_kb_entry: bool
    kb_key: Optional[str] = None


class PredictResponse(BaseModel):
    top_prediction: str
    confidence: float
    confidence_pct: str
    is_healthy: bool
    is_low_confidence: bool
    threshold: float
    note: str
    top_k: List[PredictionItem]


class AdvisoryRequest(BaseModel):
    disease: str = Field(..., description="Disease class name (e.g. Rice_Blast or Cotton_Aphid)")


class AdvisoryResponse(BaseModel):
    disease: str
    is_healthy: bool
    kb_key: Optional[str]
    markdown: str
    sections: Dict[str, str]
    context_markdown: Optional[str] = ""
    search_html: Optional[str] = ""


class AnalyzeResponse(BaseModel):
    prediction: PredictResponse
    advisory: AdvisoryResponse


class ClassItem(BaseModel):
    id: int
    label: str
    crop: str
    is_healthy: bool
    has_kb_entry: bool
    kb_key: Optional[str]


# ---- FastAPI Application -----------------------------------------------------
app = FastAPI(
    title="Crop Disease Detection & Advisory API",
    description=(
        "REST API for crop disease detection and agronomic advisory. "
        "Allows frontend applications to upload crop leaf images, obtain disease classifications, "
        "and retrieve RAG/Gemini-backed agronomic guidance."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for external frontends (e.g. React, Next.js, Vue, mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- FastAPI Endpoints -------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint indicating model status and configuration."""
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "device": device,
        "total_classes": model.config.num_labels,
        "gemini_configured": advisor.client is not None,
        "fill_mode": FILL_MODE,
    }


@app.get("/api/v1/info", response_model=InfoResponse, tags=["System"])
def model_info():
    """Return model metadata, threshold settings, and supported crop list."""
    crops = sorted(list({extract_crop_name(model.config.id2label[i]) for i in range(model.config.num_labels)}))
    return {
        "model_id": MODEL_ID,
        "num_classes": model.config.num_labels,
        "low_confidence_threshold": LOW_CONF,
        "gemini_model": GEMINI_MODEL,
        "fill_mode": FILL_MODE,
        "supported_crops": crops,
    }


@app.get("/api/v1/classes", response_model=List[ClassItem], tags=["Metadata"])
@app.get("/classes", response_model=List[ClassItem], tags=["Metadata"])
def list_classes():
    """List all 110 supported disease and healthy classes with their metadata."""
    classes = []
    for i in range(model.config.num_labels):
        lab = model.config.id2label[i]
        classes.append({
            "id": i,
            "label": lab,
            "crop": extract_crop_name(lab),
            "is_healthy": is_healthy(lab),
            "has_kb_entry": kb.resolve(lab) is not None,
            "kb_key": kb.resolve(lab),
        })
    return classes


@app.get("/api/v1/crops", tags=["Metadata"])
@app.get("/crops", tags=["Metadata"])
def list_crops():
    """List supported crops and the disease/condition classes associated with each."""
    crop_map: Dict[str, List[str]] = {}
    for i in range(model.config.num_labels):
        lab = model.config.id2label[i]
        c = extract_crop_name(lab)
        crop_map.setdefault(c, []).append(lab)
    return {
        "total_crops": len(crop_map),
        "crops": [
            {"crop": c, "classes_count": len(items), "classes": items}
            for c, items in sorted(crop_map.items())
        ],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Crop leaf image file (JPEG, PNG, WEBP)"),
    top_k: int = Query(5, ge=1, le=20, description="Number of top predictions to return"),
):
    """
    Classify an uploaded crop leaf image.
    Returns the top predicted class, confidence, low-confidence flag, and top K candidates.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    res = analyze_image_full(image, top_k=top_k)
    return res["prediction"]


@app.post("/advisory", response_model=AdvisoryResponse, tags=["Advisory"])
@app.post("/api/v1/advisory", response_model=AdvisoryResponse, tags=["Advisory"])
def get_advisory(req: AdvisoryRequest):
    """
    Generate agronomic advisory for a given disease class label.
    Uses curated knowledge base and Gemini LLM.
    """
    payload = get_advisory_payload(req.disease)
    return payload


@app.get("/api/v1/advisory/{disease_name}", response_model=AdvisoryResponse, tags=["Advisory"])
def get_advisory_by_name(disease_name: str):
    """GET endpoint to fetch advisory for a disease label (URL safe)."""
    payload = get_advisory_payload(disease_name)
    return payload


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference & Advisory"])
@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Inference & Advisory"])
async def analyze_image_endpoint(
    file: UploadFile = File(..., description="Crop leaf image file (JPEG, PNG, WEBP)"),
    top_k: int = Query(5, ge=1, le=20, description="Number of top predictions to return"),
):
    """
    All-in-one endpoint: Upload an image to get both predictions and full agronomic advisory
    for the top predicted disease in a single API call.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    return analyze_image_full(image, top_k=top_k)


@app.get("/api/v1/kb/{disease_label}", tags=["Knowledge Base"])
def get_kb_entry(disease_label: str):
    """Retrieve raw Knowledge Base records and matching chunks for a label."""
    kb_key = kb.resolve(disease_label)
    if not kb_key:
        return {
            "found": False,
            "query": disease_label,
            "kb_key": None,
            "entry": None,
            "chunks_count": 0,
            "chunks": [],
        }

    entry_data = kb.entries.get(kb_key)
    chunks = kb.retrieve(kb_key, disease_label, k=5)
    return {
        "found": True,
        "query": disease_label,
        "kb_key": kb_key,
        "entry": entry_data,
        "chunks_count": len(chunks),
        "chunks": chunks,
    }


# ---- Gradio UI ---------------------------------------------------------------
def gradio_advisory_for(label):
    if not label:
        return "", "", ""
    if is_healthy(label):
        return (
            f"## {label}\n\nNo disease or pest detected. Keep monitoring the crop regularly.",
            "",
            "",
        )
    return advisor.advise(label)


def gradio_analyze(image):
    if image is None:
        return {}, "Please upload an image.", gr.update(choices=[], value=None), "", "", ""
    scores = classify_image(image)
    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    top, p = top5[0], scores[top5[0]]

    if p < LOW_CONF:
        note = (
            f"**Low confidence ({p:.0%}).** The image may be unclear, or it may show "
            "a crop or disease the model was not trained on. "
            "Try a clearer photo, or pick a candidate below to see its advisory."
        )
        return scores, note, gr.update(choices=top5, value=None), "", "", ""

    note = f"**Prediction:** {top} ({p:.0%})"
    md, ctx, search_html = gradio_advisory_for(top)
    return scores, note, gr.update(choices=top5, value=top), md, ctx, search_html


with gr.Blocks(title="Crop Disease Detection & Advisory") as demo:
    gr.Markdown(
        "# 🌿 Crop Disease Detection & Advisory\n"
        "Upload a photo of a cotton, rice, maize, wheat, or sugarcane plant (or other supported crops). "
        "The model identifies the disease or pest, and the advisory is generated from a curated knowledge base.\n\n"
        "**Note:** REST API endpoints are available! Explore the interactive API documentation at [`/docs`](/docs)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(type="pil", label="Leaf or plant photo")
            btn = gr.Button("Analyze", variant="primary")
            label_out = gr.Label(num_top_classes=5, label="Top 5 predictions")
            note = gr.Markdown()
            choice = gr.Dropdown(label="Show advisory for", choices=[], interactive=True)
        with gr.Column(scale=2):
            advisory = gr.Markdown()
            search_html = gr.HTML()
            with gr.Accordion("Retrieved knowledge-base context", open=False):
                ctx = gr.Markdown()

    outputs = [label_out, note, choice, advisory, ctx, search_html]
    btn.click(gradio_analyze, img, outputs)
    img.upload(gradio_analyze, img, outputs)
    choice.input(gradio_advisory_for, choice, [advisory, ctx, search_html])

# Mount Gradio onto the FastAPI app so both UI and API live on the same port
app = gr.mount_gradio_app(app, demo, path="/")


# ---- Server Entrypoint -------------------------------------------------------
if __name__ == "__main__":
    print(f"\n🚀 Server starting on http://{HOST}:{PORT}")
    print(f"📖 Swagger OpenAPI docs available at: http://{HOST}:{PORT}/docs")
    print(f"🌿 Gradio UI available at: http://{HOST}:{PORT}/\n")
    uvicorn.run(app, host=HOST, port=PORT)
