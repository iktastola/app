from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from config import db
from deps import get_current_user
from models import User
from services import get_category_and_gender

router = APIRouter()

ALLOWED_MINIMAS_COLLECTIONS = {"BizkaiaMinimas", "EHMinimas"}
STYLE_ORDER = ["Libre", "Espalda", "Braza", "Mariposa", "Estilos"]


def _category_for_swimmer(swimmer: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    if not swimmer:
        return None, None

    birth_date = swimmer.get("birth_date")
    if isinstance(birth_date, str):
        try:
            birth_date = datetime.fromisoformat(birth_date)
        except ValueError:
            return None, None

    if not birth_date:
        return None, None

    gender = swimmer.get("gender", "fem")
    category, resolved_gender = get_category_and_gender(birth_date, gender)
    return f"{resolved_gender}_{category}", resolved_gender


def _distance_sort_key(doc: dict) -> int:
    try:
        return int(doc.get("distancia", 0))
    except (TypeError, ValueError):
        return 0


def _style_sort_key(doc: dict) -> int:
    prueba = doc.get("prueba")
    return STYLE_ORDER.index(prueba) if prueba in STYLE_ORDER else len(STYLE_ORDER)


@router.get("/minimas/{collection}")
async def get_minimas(
    collection: str,
    piscina_metros: Optional[int] = Query(25),
    swimmer_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    if collection not in ALLOWED_MINIMAS_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Colección no válida. Usa una de: {sorted(ALLOWED_MINIMAS_COLLECTIONS)}",
        )

    query = {}
    if collection == "BizkaiaMinimas":
        query["piscina_metros"] = piscina_metros

    documentos = await db[collection].find(query, {"_id": 0}).to_list(1000)
    documentos.sort(key=lambda doc: (_style_sort_key(doc), _distance_sort_key(doc)))

    target_swimmer = None
    if swimmer_id and (current_user.role in ("admin", "coach") or swimmer_id == current_user.id):
        target_swimmer = await db.users.find_one({"id": swimmer_id}, {"_id": 0})
    elif current_user.role == "swimmer":
        target_swimmer = current_user.model_dump()

    categoria, genero = _category_for_swimmer(target_swimmer)

    return {
        "collection": collection,
        "categoria": categoria,
        "genero": genero,
        "documentos": documentos,
    }
