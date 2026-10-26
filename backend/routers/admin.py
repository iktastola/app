from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from config import db
from deps import get_current_user
from models import MinimasImport, User
from services import (
    get_category_and_gender,
    check_minimum_time,
    check_minimum_time_bizkaia,
    recalculate_personal_best,
)

router = APIRouter()


@router.post("/audit")
async def audit_database(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden ejecutar la auditoría")

    swimmers = await db.users.find({"role": "swimmer"}).to_list(1000)

    stats = {
        "swimmers_processed": 0,
        "times_checked": 0,
        "pbs_updated": 0
    }

    for swimmer in swimmers:
        swimmer_id = swimmer["id"]
        birth_date = swimmer.get("birth_date")
        gender = swimmer.get("gender", "fem")

        if isinstance(birth_date, str):
            birth_date = datetime.fromisoformat(birth_date)

        if not birth_date:
            continue

        category, gen = get_category_and_gender(birth_date, gender)
        category_key = f"{gen}_{category}"

        # 1. Re-check all official times for minima
        official_times = await db.swim_times.find({
            "swimmer_id": swimmer_id,
            "oficial": True
        }).to_list(1000)

        for t in official_times:
            is_minimum = await check_minimum_time(
                swimmer_id=swimmer_id,
                category=category_key,
                distance=t["distance"],
                style=t["style"],
                time_seconds=t["time_seconds"],
                piscina_metros=t.get("piscina_metros", 25),
                time_date=t.get("date")
            )

            new_minima = "si" if is_minimum else "no"
            if t.get("minima") != new_minima:
                await db.swim_times.update_one(
                    {"id": t["id"]},
                    {"$set": {"minima": new_minima}}
                )
            stats["times_checked"] += 1

            is_minimum_bizkaia = await check_minimum_time_bizkaia(
                swimmer_id=swimmer_id,
                category=category_key,
                distance=t["distance"],
                style=t["style"],
                time_seconds=t["time_seconds"],
                piscina_metros=t.get("piscina_metros", 25),
                time_date=t.get("date")
            )

            new_minima_bizkaia = "si" if is_minimum_bizkaia else "no"
            if t.get("minima_bizkaia") != new_minima_bizkaia:
                await db.swim_times.update_one(
                    {"id": t["id"]},
                    {"$set": {"minima_bizkaia": new_minima_bizkaia}}
                )

        # 2. Recalculate PBs for this swimmer
        # We can use a set of (distance, style) from all official times
        tests = set((t["distance"], t["style"]) for t in official_times)
        for dist, style in tests:
            await recalculate_personal_best(swimmer_id, dist, style)
            stats["pbs_updated"] += 1

        # 3. Clean up PBs that might exist but have no official times anymore
        # Fetch current PBs
        current_pbs = await db.personal_bests.find({"swimmer_id": swimmer_id}).to_list(100)
        for pb in current_pbs:
            if (pb["distance"], pb["style"]) not in tests:
                await db.personal_bests.delete_one({
                    "swimmer_id": swimmer_id,
                    "distance": pb["distance"],
                    "style": pb["style"]
                })

        stats["swimmers_processed"] += 1

    return {"status": "success", "stats": stats}


ALLOWED_MINIMAS_COLLECTIONS = {"BizkaiaMinimas", "EHMinimas"}


@router.post("/admin/minimas/import")
async def import_minimas(payload: MinimasImport, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden importar mínimas")

    if payload.collection not in ALLOWED_MINIMAS_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Colección no válida. Usa una de: {sorted(ALLOWED_MINIMAS_COLLECTIONS)}"
        )

    coll = db[payload.collection]
    is_bizkaia = payload.collection == "BizkaiaMinimas"

    inserted = 0
    updated = 0
    errors = []

    for i, doc in enumerate(payload.documentos):
        doc = dict(doc)
        doc.pop("_id", None)

        prueba = doc.get("prueba")
        distancia = doc.get("distancia")
        if prueba is None or distancia is None:
            errors.append(f"Documento {i}: faltan 'prueba' o 'distancia'")
            continue

        doc["distancia"] = str(distancia)
        filtro = {"prueba": prueba, "distancia": str(distancia)}

        # Bizkaia diferencia por tipo de piscina; EH no
        if is_bizkaia:
            piscina = int(doc.get("piscina_metros", 25))
            doc["piscina_metros"] = piscina
            filtro["piscina_metros"] = piscina

        result = await coll.update_one(filtro, {"$set": doc}, upsert=True)
        if result.upserted_id is not None:
            inserted += 1
        else:
            updated += 1

    return {
        "status": "success",
        "collection": payload.collection,
        "inserted": inserted,
        "updated": updated,
        "errors": errors,
    }
