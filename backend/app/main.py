"""FastAPI entry point.

Run from the backend/ folder:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI

app = FastAPI(title="FinGuard AI - TabNet Auditor API")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": False}
