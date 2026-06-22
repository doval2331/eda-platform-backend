"""Crea el usuario demo si no existe (ejecutar una vez tras instalar deps)."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.db import SessionLocal, User, init_db
from app.services.auth.auth import hash_password


def main() -> None:
    settings = get_settings()
    init_db()
    email = settings.demo_user_email.strip().lower()

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"Usuario ya existe: {email}")
            return
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(settings.demo_user_password),
            nombre=settings.demo_user_nombre,
            activo=True,
            created_at=datetime.now(timezone.utc),
            ultimo_login_at=None,
        )
        db.add(user)
        db.commit()
        print("Usuario demo creado:")
        print(f"  Email:    {email}")
        print(f"  Password: {settings.demo_user_password}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
