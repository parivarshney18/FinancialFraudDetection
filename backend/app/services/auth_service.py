"""
User authentication and account management service.
"""

from typing import Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.models.entities import User
from backend.app.schemas.schemas import UserCreate
from backend.app.core.security import verify_password, hash_password


def authenticate_user(email: str, password: str, db: Session) -> Optional[User]:
    """Verify user credentials and return the active User entity if valid."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_user_by_email(email: str, db: Session) -> Optional[User]:
    """Retrieve user entity by email address."""
    return db.query(User).filter(User.email == email).first()


def create_user(user_in: UserCreate, db: Session) -> User:
    """Create a new user account with hashed password."""
    existing = get_user_by_email(user_in.email, db)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    user = User(
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        full_name=user_in.full_name,
        role=user_in.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
