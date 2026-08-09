from app.models.users import userSchema
from fastapi import Depends, HTTPException, APIRouter
from app.database import SessionLocal, Base
from typing import Annotated
from app.auth.jwt_handle import creat_jwt, decode_jwt
from starlette import status
from decouple import config  
from ..auth.auth import oauth2_bearer, db

router = APIRouter(
    prefix="/profil",
    tags=["profil"]
)


def get_current_user(
    token: Annotated[str, Depends(oauth2_bearer)],
    db: db
):
    payload = decode_jwt(token)
    print(payload)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    user = db.query(userSchema).filter(
        userSchema.id == int(payload["sub"])
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user


@router.get("/profil")
async def profil(
    current_user: Annotated[userSchema, Depends(get_current_user)]
):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_actived": current_user.is_actived,
    }