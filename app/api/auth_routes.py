from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db import User, get_db
from app.schemas import LoginRequest, LoginResponse, UserPublic
from app.services.auth import authenticate_user, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user_public(user: User) -> UserPublic:
    return UserPublic(
        id=user.id,
        email=user.email,
        nombre=user.nombre,
        activo=user.activo,
    )


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Annotated[Session, Depends(get_db)]):
    user = authenticate_user(db, body.email, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas",
        )
    token = create_access_token(user_id=user.id, email=user.email)
    return LoginResponse(
        token=token,
        token_type="bearer",
        user=_user_public(user),
    )


@router.get("/me", response_model=UserPublic)
def me(current_user: Annotated[User, Depends(get_current_user)]):
    return _user_public(current_user)
