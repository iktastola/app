from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from config import db
from deps import get_current_user
from models import PersonalBest, SwimTime, SwimTimeCreate, User
from services import (
    get_category_and_gender,
    check_minimum_time,
    check_minimum_time_bizkaia,
    recalculate_personal_best,
)

router = APIRouter()


@router.post("/times", response_model=SwimTime)
async def create_swim_time(time_data: SwimTimeCreate, current_user: User = Depends(get_current_user)):

    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="Solo entrenadores y administradores pueden registrar tiempos")

    pace_100m = time_data.time_seconds / (time_data.distance / 100)

    # 🔥 SOLO si oficial calculamos mínimas
    is_minimum = False
    if time_data.oficial:
        # Buscar al nadador para usar SU birth_date y gender (no el del coach/admin)
        swimmer = await db.users.find_one({"id": time_data.swimmer_id})
        if not swimmer:
            raise HTTPException(status_code=404, detail="Nadador no encontrado")
        swimmer_birth_date = swimmer.get("birth_date")
        if isinstance(swimmer_birth_date, str):
            swimmer_birth_date = datetime.fromisoformat(swimmer_birth_date)
        category, gender = get_category_and_gender(swimmer_birth_date, swimmer.get("gender", "fem"))
        is_minimum = await check_minimum_time(
            swimmer_id=time_data.swimmer_id,
            category=f"{gender}_{category}",
            distance=time_data.distance,
            style=time_data.style,
            time_seconds=time_data.time_seconds,
            piscina_metros=time_data.piscina_metros,
            time_date=time_data.date
        )
        is_minimum_bizkaia = await check_minimum_time_bizkaia(
            swimmer_id=time_data.swimmer_id,
            category=f"{gender}_{category}",
            distance=time_data.distance,
            style=time_data.style,
            time_seconds=time_data.time_seconds,
            piscina_metros=time_data.piscina_metros,
            time_date=time_data.date
        )
    else:
        is_minimum_bizkaia = False

    time_obj = SwimTime(
        **time_data.model_dump(),
        pace_100m=pace_100m,
        recorded_by=current_user.id,
        minima="si" if is_minimum else "no",
        minima_bizkaia="si" if is_minimum_bizkaia else "no"
    )

    doc = time_obj.model_dump()
    doc["date"] = doc["date"].isoformat()
    doc["created_at"] = doc["created_at"].isoformat()

    await db.swim_times.insert_one(doc)

    # 🔥 Recalculamos PB (recalculate_personal_best ya filtra por oficial=True)
    await recalculate_personal_best(
        time_obj.swimmer_id,
        time_obj.distance,
        time_obj.style
    )

    return time_obj


@router.get("/times", response_model=List[SwimTime])
async def get_swim_times(
    swimmer_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    query = {}

    if current_user.role == "swimmer":
        query["swimmer_id"] = current_user.id
    elif swimmer_id:
        query["swimmer_id"] = swimmer_id

    times = await db.swim_times.find(query, {"_id": 0}).to_list(1000)

    for t in times:
        if isinstance(t.get("date"), str):
            t["date"] = datetime.fromisoformat(t["date"])
        if isinstance(t.get("created_at"), str):
            t["created_at"] = datetime.fromisoformat(t["created_at"])
        if "pace_100m" not in t and t.get("distance") and t.get("time_seconds"):
            t["pace_100m"] = t["time_seconds"] / (t["distance"] / 100)
        if "minima" not in t:
            t["minima"] = "no"
        if "minima_bizkaia" not in t:
            t["minima_bizkaia"] = "no"
        if "oficial" not in t:
            t["oficial"] = True  # Default
        if "piscina_metros" not in t:
            t["piscina_metros"] = 25

    times.sort(key=lambda x: x["date"].astimezone(timezone.utc) if x["date"].tzinfo is None else x["date"], reverse=True)
    return times


@router.put("/times/{time_id}", response_model=SwimTime)
async def update_swim_time(
    time_id: str,
    time_data: SwimTimeCreate,
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="Solo entrenadores y administradores pueden editar tiempos")

    doc = time_data.model_dump()
    doc["date"] = doc["date"].isoformat()
    doc["pace_100m"] = time_data.time_seconds / (time_data.distance / 100)

    is_minimum = False
    if time_data.oficial:
        swimmer = await db.users.find_one({"id": time_data.swimmer_id})
        if not swimmer:
            raise HTTPException(status_code=404, detail="Nadador no encontrado")
        category, gender = get_category_and_gender(swimmer["birth_date"], swimmer.get("gender", "fem"))

        is_minimum = await check_minimum_time(
            swimmer_id=time_data.swimmer_id,
            category=f"{gender}_{category}",
            distance=time_data.distance,
            style=time_data.style,
            time_seconds=time_data.time_seconds,
            piscina_metros=time_data.piscina_metros,
            time_date=time_data.date
        )
        is_minimum_bizkaia = await check_minimum_time_bizkaia(
            swimmer_id=time_data.swimmer_id,
            category=f"{gender}_{category}",
            distance=time_data.distance,
            style=time_data.style,
            time_seconds=time_data.time_seconds,
            piscina_metros=time_data.piscina_metros,
            time_date=time_data.date
        )
    else:
        is_minimum_bizkaia = False

    doc["minima"] = "si" if is_minimum else "no"
    doc["minima_bizkaia"] = "si" if is_minimum_bizkaia else "no"

    result = await db.swim_times.update_one({"id": time_id}, {"$set": doc})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Tiempo no encontrado")

    # 🔥 Siempre recalculamos PB, ya que pudo haber pasado de oficial a no oficial o viceversa
    await recalculate_personal_best(time_data.swimmer_id, time_data.distance, time_data.style)

    updated_doc = await db.swim_times.find_one({"id": time_id}, {"_id": 0})
    updated_doc["date"] = datetime.fromisoformat(updated_doc["date"])
    updated_doc["created_at"] = datetime.fromisoformat(updated_doc["created_at"])

    return SwimTime(**updated_doc)


@router.delete("/times/{time_id}")
async def delete_swim_time(time_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="Solo entrenadores y administradores pueden eliminar tiempos")

    time_doc = await db.swim_times.find_one({"id": time_id})
    if not time_doc:
        raise HTTPException(status_code=404, detail="Tiempo no encontrado")

    await db.swim_times.delete_one({"id": time_id})

    await recalculate_personal_best(time_doc["swimmer_id"], time_doc["distance"], time_doc["style"])

    return {"message": "Tiempo eliminado"}


@router.get("/personal-bests", response_model=List[PersonalBest])
async def get_personal_bests(
    swimmer_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    query = {}
    if current_user.role == "swimmer":
        query["swimmer_id"] = current_user.id
    elif swimmer_id:
        query["swimmer_id"] = swimmer_id

    pbs = await db.personal_bests.find(query, {"_id": 0}).to_list(1000)
    for pb in pbs:
        if isinstance(pb.get("date"), str):
            pb["date"] = datetime.fromisoformat(pb["date"])

    return pbs
