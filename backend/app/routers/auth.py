"""
Authentication and user session routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.schemas import Token, UserLogin, UserResponse, UserCreate
from backend.app.services.auth_service import authenticate_user, create_user
from backend.app.core.security import create_access_token
from backend.app.core.dependencies import get_current_user
from backend.app.models.entities import User

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Authenticate user with email and password, returning JWT access token."""
    user = authenticate_user(email=form_data.username, password=form_data.password, db=db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
        email=user.email,
    )


@router.post("/login-json", response_model=Token)
def login_json(
    credentials: UserLogin,
    db: Session = Depends(get_db),
):
    """JSON-body alternative login endpoint for frontend convenience."""
    user = authenticate_user(email=credentials.email, password=credentials.password, db=db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
        email=user.email,
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Retrieve profile and role of currently authenticated user."""
    return current_user


@router.post("/register", response_model=UserResponse)
def register(
    user_in: UserCreate,
    db: Session = Depends(get_db),
):
    """Register a new user account."""
    return create_user(user_in, db)
