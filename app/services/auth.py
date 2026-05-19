"""Autenticación JWT y contraseñas (patrón similar a gestión-archivo)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import User


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(*, user_id: str, email: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "type": "auth",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("Token inválido o expirado") from exc
    if payload.get("type") != "auth" or not payload.get("sub"):
        raise ValueError("Token inválido")
    return payload


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    normalized = email.strip().lower()
    user = db.query(User).filter(User.email == normalized).first()
    if user is None or not user.activo:
        return None
    if not verify_password(password, user.password_hash):
        return None
    user.ultimo_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.get(User, user_id)
