"""
User administration routes.
"""

from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.schemas import UserResponse
from backend.app.models.entities import User
from backend.app.core.dependencies import require_role

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/", response_model=List[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin"])),
):
    """List all registered system users (Admin authorization required)."""
    return db.query(User).all()
