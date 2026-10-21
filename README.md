---
title: Crop Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# Crop Disease Detection & Advisory Backend

Upload a photo of a crop leaf or plant. The vision transformer model identifies the disease or pest, and the system generates an agronomic advisory from a curated knowledge base and Gemini LLM.

This repository serves both:
1. **Interactive Web UI**: Built with Gradio (rendered at `/`).
2. **REST API**: Built with FastAPI and full CORS support for external web, mobile, and custom frontends. Interactive Swagger docs are available at [`/docs`](/docs).

---

## How It Works on Hugging Face Spaces (No Docker Needed)

* **Why is there no Dockerfile?**  
  Hugging Face Spaces natively supports predefined SDKs. The YAML frontmatter in this `README.md` specifies `sdk: gradio` and `app_file: app.py`. Hugging Face automatically provisions an optimized Python environment, installs `requirements.txt`, and runs `python app.py`. No Docker container setup is required.
* **Where does the model come from?**  
  The application automatically downloads and caches the weights for [`vegadurmin/crop-disease-deit-small`](https://huggingface.co/vegadurmin/crop-disease-deit-small) directly from the Hugging Face Model Hub using the `transformers` library (`AutoImageProcessor` and `AutoModelForImageClassification`). It is a fine-tuned DeiT (Data-efficient Image Transformer) small model covering 110 crop conditions and diseases across Cotton, Rice, Maize, Wheat, Sugarcane, Apple, Tomato, Potato, and more.

---

## Quickstart / Running Locally

### 1. Setup Virtual Environment & Install Dependencies

Using `uv` (recommended) or standard `python3`:

```bash
# Using uv (fastest)
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment variables:
* `MODEL_ID`: Hugging Face repo ID or local model path (default: `vegadurmin/crop-disease-deit-small`)
* `KB_DIR`: Path to knowledge base directory (default: `./kb`)
* `GEMINI_API_KEY`: Google Gemini API key for dynamic advisory and search grounding (optional)
* `GEMINI_MODEL`: Gemini model version (default: `gemini-3.1-flash-lite`)
* `FILL_MODE`: How to fill missing knowledge base sections: `ai`, `search`, `auto`, or `off` (default: `ai`)
* `PORT`: Server port (default: `7860`)

### 3. Start the Server

```bash
python app.py
# or using uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

* Gradio Web UI: `http://localhost:7860/`
* Swagger API Docs: `http://localhost:7860/docs`
* OpenAPI JSON: `http://localhost:7860/openapi.json`

### 4. Run Integration Tests

```bash
python test_api.py
```

---

## REST API Reference for Frontend Developers

CORS is enabled for all origins (`*`), allowing direct requests from any frontend (React, Next.js, Vue, Svelte, iOS/Android apps).

### 1. Health Check
* **Endpoint:** `GET /health` or `GET /api/v1/health`
* **Response:**
  ```json
  {
    "status": "ok",
    "model_id": "vegadurmin/crop-disease-deit-small",
    "device": "cpu",
    "total_classes": 110,
    "gemini_configured": false,
    "fill_mode": "ai"
  }
  ```

### 2. Model & Crops Info
* **Endpoint:** `GET /api/v1/info`
* Returns metadata on supported crops, total classes, and confidence threshold.
* **Endpoint:** `GET /api/v1/crops`
* Returns a breakdown of crops and the disease classes available for each.
* **Endpoint:** `GET /api/v1/classes`
* Returns all 110 supported labels with metadata (`is_healthy`, `crop`, `has_kb_entry`).

### 3. Image Classification
* **Endpoint:** `POST /api/v1/predict` (also available at `/predict`)
* **Content-Type:** `multipart/form-data`
* **Form Field:** `file` (Image binary: JPEG, PNG, or WEBP)
* **Query Param:** `top_k` (optional, default `5`)
* **Response:**
  ```json
  {
    "top_prediction": "Rice_Blast",
    "confidence": 0.9421,
    "confidence_pct": "94.2%",
    "is_healthy": false,
    "is_low_confidence": false,
    "threshold": 0.50,
    "note": "Prediction: Rice_Blast (94%)",
    "top_k": [
      {
        "label": "Rice_Blast",
        "confidence": 0.9421,
        "confidence_pct": "94.2%",
        "crop": "Rice",
        "is_healthy": false,
        "has_kb_entry": true,
        "kb_key": "Blast"
      }
    ]
  }
  ```

### 4. Agronomic Advisory
* **Endpoint:** `POST /api/v1/advisory` (also available as `GET /api/v1/advisory/{disease_name}`)
* **Body:**
  ```json
  {
    "disease": "Rice_Blast"
  }
  ```
* **Response:**
  ```json
  {
    "disease": "Rice_Blast",
    "is_healthy": false,
    "kb_key": "Blast",
    "markdown": "## Rice_Blast\n\n### Symptoms...",
    "sections": {
      "Symptoms": "...",
      "Pathogen": "...",
      "Favourable Conditions": "...",
      "Survival and Mode of Spread": "...",
      "Management": "..."
    },
    "context_markdown": "...",
    "search_html": ""
  }
  ```

### 5. All-in-One Analysis (Prediction + Advisory)
* **Endpoint:** `POST /api/v1/analyze` (also available at `/analyze`)
* **Content-Type:** `multipart/form-data`
* **Form Field:** `file` (Image binary)
* Performs image classification and retrieves advisory for the top detected condition in a single network round-trip.

### 6. Direct Knowledge Base Lookup
* **Endpoint:** `GET /api/v1/kb/{disease_label}`
* Returns raw curated knowledge base records and chunk text.

---

## Frontend Integration Example (JavaScript / TypeScript)

```typescript
// Example: Sending an image from a file input to /api/v1/analyze
async function analyzeCropImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:7860/api/v1/analyze", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  console.log("Prediction:", data.prediction.top_prediction);
  console.log("Confidence:", data.prediction.confidence_pct);
  console.log("Advisory Sections:", data.advisory.sections);
  return data;
}
```
