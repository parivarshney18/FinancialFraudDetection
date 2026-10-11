# Financial Fraud Detection using TabNet

An explainable-AI approach for Indian banking transactions.
B.Tech Final Year Project - ABESEC Ghaziabad, CSE, Session 2026-27.

Team: Pari Varshney, Neha Awasthi, Nandini Sharma
Guide: Ms. Shipra Gautam

## Structure
- `ai/`        offline ML pipeline (EDA, preprocessing, CV, training, evaluation, explainability)
- `backend/`   FastAPI service (auth, scoring, explanations, reports)
- `frontend/`  Streamlit audit dashboard
- `database/`  SQL schema and seed data
- `docs/`      blueprint, architecture, experiment log, viva Q&A
- `tests/`     pytest tests

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

## Status
Phase 0 - project setup. See `docs/blueprint.md`.
