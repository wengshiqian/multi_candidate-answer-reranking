"""
Medical Image Analysis Web Demo
================================
FastAPI backend + HTML frontend for the RAGMedicalReportWorkflow.

Usage:
    python web_demo.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import json
import uuid
import time
import importlib.util
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Configuration ────────────────────────────────────────────────────────────
BASE_MODEL = r"D:\my_huggingface_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5"
LORA_ADAPTER = r"d:\weng\llama-factory\saves\Qwen2.5-VL-7B-Instruct\lora\exp_dropout_10_full-bdsadam"
RAG_KNOWLEDGE = r"d:/weng/llama-factory/data/medical_guidelines_rag.txt"

HOST = "0.0.0.0"
PORT = 7860
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# ── Import the workflow from langgraph_shiyan(3).py ──────────────────────────
def load_workflow_module():
    """Dynamically import the workflow module (filename has parentheses)."""
    module_path = os.path.join(os.path.dirname(__file__), "langgraph_shiyan(3).py")
    spec = importlib.util.spec_from_file_location("langgraph_workflow", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

workflow_module = load_workflow_module()
RAGMedicalReportWorkflow = workflow_module.RAGMedicalReportWorkflow
MedicalReportState = workflow_module.MedicalReportState

# ── Initialize the model (loaded once at startup) ───────────────────────────
print("\n" + "=" * 80)
print("🌐 Starting Medical Image Analysis Web Demo")
print("=" * 80)

workflow_engine = RAGMedicalReportWorkflow(
    base_model_path=BASE_MODEL,
    lora_adapter_path=LORA_ADAPTER,
    rag_knowledge_path=RAG_KNOWLEDGE,
)

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Medical Image Analysis", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images as static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page."""
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    """
    Accept an uploaded medical image + question, run the LangGraph workflow,
    and return structured results.
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {image.content_type}. Allowed: {allowed_types}",
        )

    # Save uploaded file
    file_ext = Path(image.filename).suffix or ".png"
    file_id = f"{uuid.uuid4().hex[:12]}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_id)

    contents = await image.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Build initial state
    initial_state = MedicalReportState(
        sample_id=file_id,
        image_path=file_path,
        question=question.strip(),
        modality="Unknown",
        body_part="Unknown",
        rag_context="",
        initial_analysis="",
        plan_outline="",
        draft_report="",
        candidate_answers=[],
        validation_result={},
        quality_metrics={},
        criticism="",
        final_report="",
        refinement_count=0,
        metadata={},
    )

    # Run the workflow
    start_time = time.time()
    try:
        final_state = workflow_engine.workflow.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    elapsed = round(time.time() - start_time, 2)

    # Extract results
    metadata = final_state.get("metadata", {})

    result = {
        "success": True,
        "elapsed_seconds": elapsed,
        "image_url": f"/uploads/{file_id}",
        "question": question.strip(),
        "answer": final_state.get("draft_report", ""),
        "reviewer_applied": metadata.get("reviewer_applied", False),
        "reviewer_changes": metadata.get("reviewer_changes", False),
        "reviewer_fallback": metadata.get("reviewer_fallback", False),
    }

    return JSONResponse(content=result)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": True}


if __name__ == "__main__":
    print(f"\n🌐 Open http://localhost:{PORT} in your browser")
    print(f"📁 Uploads saved to: {UPLOAD_DIR}")
    print("=" * 80 + "\n")
    uvicorn.run(app, host=HOST, port=PORT)
