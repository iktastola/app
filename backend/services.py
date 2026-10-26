from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import jwt

from config import db, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_HOURS


def get_category_and_gender(birth_date: datetime, gender: Optional[str] = "fem") -> tuple[str, str]:
    categorias = {
        2014: 'alevin_1',
        2013: 'alevin_2',
        2012: 'infantil_1',
        2011: 'infantil_2',
        2010: 'junior_1',
        2009: 'junior_2',
        2008: 'junior_3',
        2007: 'abs_jov',
        2006: 'abs_jov',
        2005: 'abs_1',
    }

    anio_nacimiento = birth_date.year
    hoy = datetime.now()

    temporada_inicio = datetime(hoy.year, 9, 1)
    temporada_actual = hoy.year if hoy >= temporada_inicio else hoy.year - 1
    REFERENCIA_TEMPORADA = 2025  # Actualizar cada temporada: +1 cuando el cohort más joven entre en alevin_1
    anio_categoria = anio_nacimiento - (temporada_actual - REFERENCIA_TEMPORADA)

    for año, categoria in categorias.items():
        if anio_categoria >= año:
            return categoria, gender

    return 'abs_1', gender


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def check_minimum_time(swimmer_id, category, distance, style, time_seconds, piscina_metros=25, time_date=None):
    # Las mínimas EH no distinguen por tipo de piscina (se usa "piscina_metros" igual)
    return await _minima_generic("EHMinimas", category, distance, style, time_seconds, piscina_metros, time_date)


async def check_minimum_time_bizkaia(swimmer_id, category, distance, style, time_seconds, piscina_metros=25, time_date=None):
    return await _minima_generic("BizkaiaMinimas", category, distance, style, time_seconds, piscina_metros, time_date)


async def _minima_generic(collection_name, category, distance, style, time_seconds, piscina_metros, time_date):
    # Marcas con un tiempo superior a 14 meses no son consideradas mínima
    if time_date is not None:
        if isinstance(time_date, str):
            time_date = datetime.fromisoformat(time_date)
        cutoff = datetime.now(timezone.utc) - timedelta(days=14 * 30)
        if time_date.tzinfo is None:
            time_date = time_date.replace(tzinfo=timezone.utc)
        if time_date < cutoff:
            return False

    coll = db[collection_name]
    doc = await coll.find_one({
        "prueba": style,
        "distancia": str(distance),
        "piscina_metros": piscina_metros,
    })

    if not doc:
        return False

    for t in doc.get("tiempos", []):
        if category in t:
            return time_seconds < t["time_seconds"]

    return False


async def update_personal_best(
    swimmer_id: str,
    distance: int,
    style: str,
    time_seconds: float,
    date: datetime,
    competition: Optional[str],
):
    doc = {
        "swimmer_id": swimmer_id,
        "distance": distance,
        "style": style,
        "best_time": time_seconds,
        "date": date.isoformat() if isinstance(date, datetime) else date,
        "competition": competition,
    }
    await db.personal_bests.update_one(
        {"swimmer_id": swimmer_id, "distance": distance, "style": style},
        {"$set": doc},
        upsert=True,
    )


async def recalculate_personal_best(swimmer_id: str, distance: int, style: str):
    times = await db.swim_times.find(
        {"swimmer_id": swimmer_id, "distance": distance, "style": style, "oficial": True}
    ).to_list(1000)

    if not times:
        await db.personal_bests.delete_one(
            {"swimmer_id": swimmer_id, "distance": distance, "style": style}
        )
        return

    best = min(times, key=lambda x: x["time_seconds"])
    date = best["date"] if isinstance(best["date"], datetime) else datetime.fromisoformat(best["date"])

    await update_personal_best(
        swimmer_id,
        distance,
        style,
        best["time_seconds"],
        date,
        best.get("competition"),
    )
