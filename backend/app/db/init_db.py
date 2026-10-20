"""
Database initialization and default user seeding.
"""

from backend.app.db.session import engine, SessionLocal
from backend.app.models.entities import Base, User
from backend.app.core.security import hash_password


def init_db() -> None:
    """Create database tables and seed baseline users if not already present."""
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Seed default admin if missing
        admin = db.query(User).filter(User.email == "admin@finguard.ai").first()
        if not admin:
            admin = User(
                email="admin@finguard.ai",
                hashed_password=hash_password("admin123"),
                full_name="Lead Compliance Auditor",
                role="admin",
                is_active=True,
            )
            db.add(admin)

        # Seed default analyst if missing
        analyst = db.query(User).filter(User.email == "analyst@finguard.ai").first()
        if not analyst:
            analyst = User(
                email="analyst@finguard.ai",
                hashed_password=hash_password("analyst123"),
                full_name="Senior Fraud Analyst",
                role="analyst",
                is_active=True,
            )
            db.add(analyst)

        db.commit()
    finally:
        db.close()
