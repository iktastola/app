import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from config import db
from deps import get_current_user
from models import (
    LoginRequest, LoginResponse, User, UserCreate, UserSelfUpdate,
)
from services import hash_password, verify_password, create_access_token

router = APIRouter()


@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user_doc = await db.users.find_one({"email": request.email}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    if not verify_password(request.password, user_doc["password"]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    user_doc.pop("password", None)

    if isinstance(user_doc.get("created_at"), str):
        user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])

    user = User(**user_doc)
    token = create_access_token({"sub": user.id, "role": user.role})
    return LoginResponse(token=token, user=user)


@router.post("/auth/register", response_model=User)
async def register(user_data: UserCreate, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden registrar usuarios")

    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email ya registrado")

    user_dict = user_data.model_dump()
    hashed_pw = hash_password(user_dict.pop("password"))

    # Asignar la fecha de nacimiento si no se recibe
    if "birth_date" not in user_dict:
        user_dict["birth_date"] = datetime.now(timezone.utc)  # Se asigna la fecha actual si no se pasa

    if "gender" not in user_dict:
        user_dict["gender"] = "fem"  # Se asigna fem si no se pasa

    user_obj = User(**user_dict)
    doc = user_obj.model_dump()
    doc["password"] = hashed_pw
    doc["created_at"] = doc["created_at"].isoformat()

    await db.users.insert_one(doc)
    return user_obj
