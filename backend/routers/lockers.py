from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from config import db
from deps import get_current_user
from models import Locker, LockerCreate, User

router = APIRouter()


@router.get("/lockers/{swimmer_id}", response_model=Locker)
async def get_locker(swimmer_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role == "swimmer" and current_user.id != swimmer_id:
        raise HTTPException(status_code=403, detail="No autorizado")

    locker_doc = await db.lockers.find_one({"swimmer_id": swimmer_id}, {"_id": 0})
    if not locker_doc:
        raise HTTPException(status_code=404, detail="Taquilla no encontrada")

    if isinstance(locker_doc.get("updated_at"), str):
        locker_doc["updated_at"] = datetime.fromisoformat(locker_doc["updated_at"])

    return Locker(**locker_doc)


@router.post("/lockers", response_model=Locker)
async def create_locker(locker_data: LockerCreate, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden gestionar taquillas")

    locker_obj = Locker(**locker_data.model_dump())
    doc = locker_obj.model_dump()
    doc["updated_at"] = doc["updated_at"].isoformat()

    await db.lockers.update_one(
        {"swimmer_id": locker_data.swimmer_id},
        {"$set": doc},
        upsert=True,
    )

    return locker_obj


@router.put("/lockers/{swimmer_id}", response_model=Locker)
async def update_locker(swimmer_id: str, locker_data: LockerCreate, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden gestionar taquillas")

    doc = locker_data.model_dump()
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()

    result = await db.lockers.update_one({"swimmer_id": swimmer_id}, {"$set": doc})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Taquilla no encontrada")

    updated_doc = await db.lockers.find_one({"swimmer_id": swimmer_id}, {"_id": 0})
    updated_doc["updated_at"] = datetime.fromisoformat(updated_doc["updated_at"])

    return Locker(**updated_doc)
