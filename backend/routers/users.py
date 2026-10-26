import re
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from config import db
from deps import get_current_user
from models import User, UserBase, UserSelfUpdate
from services import hash_password

router = APIRouter()


@router.get("/users", response_model=List[User])
async def get_users(current_user: User = Depends(get_current_user)):
    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="No autorizado")

    users = await db.users.find({}, {"_id": 0, "password": 0}).to_list(1000)
    for user in users:
        if isinstance(user.get("created_at"), str):
            user["created_at"] = datetime.fromisoformat(user["created_at"])
        if isinstance(user.get("birth_date"), str):
            user["birth_date"] = datetime.fromisoformat(user["birth_date"])
        if not user.get("gender"):
            user["gender"] = "fem"

    return users


@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role == "swimmer" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="No autorizado")

    user_doc = await db.users.find_one({"id": user_id}, {"_id": 0, "password": 0})
    if not user_doc:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    if isinstance(user_doc.get("created_at"), str):
        user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])

    if isinstance(user_doc.get("birth_date"), str):
        user_doc["birth_date"] = datetime.fromisoformat(user_doc["birth_date"])

    if not user_doc.get("gender"):
        user_doc["gender"] = "fem"

    return User(**user_doc)


@router.put("/users/{user_id}", response_model=User)
async def update_user(user_id: str, user_data: UserBase, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden editar usuarios")

    existing_user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not existing_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    if user_data.email != existing_user["email"]:
        email_exists = await db.users.find_one(
            {"email": user_data.email, "id": {"$ne": user_id}}
        )
        if email_exists:
            raise HTTPException(status_code=400, detail="Email ya está en uso por otro usuario")

    update_doc = {
        "name": user_data.name,
        "email": user_data.email,
        "role": user_data.role,
    }

    # Verificar si se ha recibido un campo birth_date
    if "birth_date" in user_data.model_dump(exclude_unset=True):
        update_doc["birth_date"] = user_data.birth_date
    else:
        # Si no se pasa birth_date, asignar el valor actual
        update_doc["birth_date"] = existing_user["birth_date"] if "birth_date" in existing_user else datetime.now(timezone.utc)

    if user_data.gender is not None:
        update_doc["gender"] = user_data.gender
    else:
        update_doc["gender"] = existing_user.get("gender", "fem")

    if user_data.avatar_url is not None:
        update_doc["avatar_url"] = user_data.avatar_url
    else:
        update_doc["avatar_url"] = existing_user.get("avatar_url")

    if "monthly_fee" in user_data.model_dump(exclude_unset=True):
        if user_data.monthly_fee is not None and user_data.monthly_fee < 0:
            raise HTTPException(status_code=400, detail="La cuota mensual no puede ser negativa")
        update_doc["monthly_fee"] = (
            round(float(user_data.monthly_fee), 2) if user_data.monthly_fee is not None else None
        )

    result = await db.users.update_one({"id": user_id}, {"$set": update_doc})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    updated_user = await db.users.find_one({"id": user_id}, {"_id": 0, "password": 0})
    if isinstance(updated_user.get("created_at"), str):
        updated_user["created_at"] = datetime.fromisoformat(updated_user["created_at"])

    return User(**updated_user)


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden eliminar usuarios")

    result = await db.users.delete_one({"id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    return {"message": "Usuario eliminado"}


@router.get("/users/me", response_model=User)
async def get_self_profile(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/users/me", response_model=User)
async def update_self_profile(user_data: UserSelfUpdate, current_user: User = Depends(get_current_user)):
    update_doc = {}

    if user_data.birth_date:
        update_doc["birth_date"] = user_data.birth_date

    if user_data.password:
        if len(user_data.password) < 8 or not re.search(r"\d", user_data.password):
            raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 8 caracteres y un número")
        update_doc["password"] = hash_password(user_data.password)

    if hasattr(user_data, 'avatar_url') and user_data.avatar_url:
        update_doc["avatar_url"] = user_data.avatar_url
    elif "avatar_url" in (await db.users.find_one({"id": current_user.id})):  # Check if it exists in request or keep old
        pass  # No change if not in UserSelfUpdate

    if not update_doc:
        return current_user

    result = await db.users.update_one({"id": current_user.id}, {"$set": update_doc})

    updated_user_doc = await db.users.find_one({"id": current_user.id}, {"_id": 0, "password": 0})
    if isinstance(updated_user_doc.get("created_at"), str):
        updated_user_doc["created_at"] = datetime.fromisoformat(updated_user_doc["created_at"])
    if isinstance(updated_user_doc.get("birth_date"), str):
        updated_user_doc["birth_date"] = datetime.fromisoformat(updated_user_doc["birth_date"])

    return User(**updated_user_doc)
