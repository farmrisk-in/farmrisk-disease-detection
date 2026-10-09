"""
Crop disease detection + RAG advisory. Gradio app for a Hugging Face Space.

Environment variables:
    MODEL_ID        HF repo or local folder of best_model
    KB_DIR          folder with disease_kb.json, chunks.jsonl, coverage.csv (default: ./kb)
    GEMINI_API_KEY  Gemini API key (set as a Space secret)
    GEMINI_MODEL    default: gemini-3.1-flash-lite
    FILL_MODE       how to fill sections missing from the KB:
                    ai (default) Gemini general knowledge, labelled, no citations
                    search       Google Search grounding with sources (needs grounding quota)
                    auto         search first, AI knowledge if search fails
                    off          show "Not available" only
    HF_TOKEN        only if the model repo is private

Local test:
    MODEL_ID=/path/to/best_model KB_DIR=/path/to/kb GEMINI_API_KEY=... python app.py
"""
import os
from pathlib import Path

import gradio as gr
import torch
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModelForImageClassification

from rag import Advisor, KnowledgeBase

APP_DIR = Path(__file__).resolve().parent
MODEL_ID = os.getenv("MODEL_ID", "YOUR_HF_USERNAME/crop-disease-deit-small")
KB_DIR = os.getenv("KB_DIR", str(APP_DIR / "kb"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FILL_MODE = os.getenv("FILL_MODE", "ai")  # off | ai | search | auto
HF_TOKEN = os.getenv("HF_TOKEN")
LOW_CONF = 0.50

# ---- classifier ---------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device).eval()
side = model.config.image_size
tf = T.Compose([
    T.Resize((side, side)),
    T.ToTensor(),
    T.Normalize(processor.image_mean, processor.image_std),
])

# ---- knowledge base + advisor ---------------------------------------------------
kb = KnowledgeBase(KB_DIR)
advisor = Advisor(kb, GEMINI_MODEL, fill_mode=FILL_MODE)

print("\n[map] model class -> KB entry")
for i in range(model.config.num_labels):
    lab = model.config.id2label[i]
    print(f"  {lab:40s} -> {kb.resolve(lab)}")
print()


def is_healthy(label):
    return "healthy" in label.lower()


def classify(image):
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(pixel_values=x).logits.softmax(-1)[0].cpu()
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


def advisory_for(label):
    if not label:
        return "", "", ""
    if is_healthy(label):
        return (f"## {label}\n\nNo disease or pest detected. "
                "Keep monitoring the crop regularly."), "", ""
    return advisor.advise(label)


def analyze(image):
    if image is None:
        return {}, "Please upload an image.", gr.update(choices=[], value=None), "", "", ""
    scores = classify(image)
    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    top, p = top5[0], scores[top5[0]]

    if p < LOW_CONF:
        note = (f"**Low confidence ({p:.0%}).** The image may be unclear, or it may show "
                "a crop or disease the model was not trained on. "
                "Try a clearer photo, or pick a candidate below to see its advisory.")
        return scores, note, gr.update(choices=top5, value=None), "", "", ""

    note = f"**Prediction:** {top} ({p:.0%})"
    md, ctx, search_html = advisory_for(top)
    return scores, note, gr.update(choices=top5, value=top), md, ctx, search_html


# ---- UI -----------------------------------------------------------------------
with gr.Blocks(title="Crop Disease Detection & Advisory") as demo:
    gr.Markdown(
        "# Crop Disease Detection & Advisory\n"
        "Upload a photo of a cotton, rice, maize, wheat or sugarcane plant. "
        "The model identifies the disease or pest. "
        "The advisory is generated from a curated knowledge base.\n\n"
        "*Research prototype. Confirm with an agronomist before spraying.*"
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
            search_html = gr.HTML()  # Google Search suggestions (required when grounding is used)
            with gr.Accordion("Retrieved knowledge-base context", open=False):
                ctx = gr.Markdown()

    outputs = [label_out, note, choice, advisory, ctx, search_html]
    btn.click(analyze, img, outputs)
    img.upload(analyze, img, outputs)
    choice.input(advisory_for, choice, [advisory, ctx, search_html])

if __name__ == "__main__":
    demo.launch()
