"""
Model registry and active deployment metadata routes.
"""

from fastapi import APIRouter
from backend.app.services.model_registry_service import (
    get_active_model_details,
    get_ieee_model_details,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/active")
def get_active_model():
    """Retrieve metadata and parameter configuration of currently loaded TabNet model."""
    return get_active_model_details()


@router.get("/ieee")
def get_ieee_models():
    """Retrieve runtime metadata of deployed IEEE-CIS multi-model suite."""
    return get_ieee_model_details()
