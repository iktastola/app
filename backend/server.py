from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
import uuid
import os
import sys
import logging

ROOT_DIR = Path(__file__).parent

# ================== MongoDB ==================

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

# ================== JWT ==================

SECRET_KEY = os.environ.get(
    "JWT_SECRET",
    "d3e1f0b2a7c44e3e91c09bf7396c12d8f4eb859a0f2d1c1aa45fd95b9214cba3",
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer()

# =========== LOG =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # IMPORTANTE: Render lee de stdout
    ]
)
logger = logging.getLogger("backend")


# ================== APP ==================

app = FastAPI()
api_router = APIRouter(prefix="/api")

# ================== MODELOS ==================

class UserBase(BaseModel):
    email: EmailStr
    name: str
    role: str  # swimmer, coach, admin
    birth_date: Optional[datetime] = None  # Campo opcional para la fecha de nacimiento
    gender: Optional[str] = None  # Campo opcional para la fecha de nacimiento


class UserCreate(UserBase):
    password: str


class User(UserBase):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    birth_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # Se añade birth_date
    gender: str = "fem"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    token: str
    user: User


class SwimTimeBase(BaseModel):
    swimmer_id: str
    distance: int  # 50, 100, 200, 400, 800, 1500
    style: str     # Libre, Espalda, Braza, Mariposa, Estilos...
    time_seconds: float
    date: datetime
    competition: Optional[str] = None


class SwimTimeCreate(SwimTimeBase):
    """Entrada desde el frontend. No incluye pace_100m ni recorded_by."""
    pass


class SwimTime(SwimTimeBase):
    """Modelo de salida y almacenamiento."""
    model_config = ConfigDict(extra="ignore")

    pace_100m: float = 0.0                   # por si no existe en documentos antiguos
    recorded_by: Optional[str] = None        # idem, opcional para no romper tiempos viejos
    minima: str = "no"   # 👈 AÑADIR ESTO


    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LockerBase(BaseModel):
    swimmer_id: str
    pants_size: str
    shirt_size: str
    hoodie_size: str


class LockerCreate(LockerBase):
    pass


class Locker(LockerBase):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PersonalBest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    swimmer_id: str
    distance: int
    style: str
    best_time: float
    date: datetime
    competition: Optional[str] = None

# ================== HELPERS AUTH ==================

def get_category_and_gender(birth_date: datetime, gender: Optional[str] = "fem") -> str:
    """Calcula la categoría y género del nadador."""
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

    # Obtener el año de nacimiento
    anio_nacimiento = birth_date.year

    # Determinar la temporada actual (a partir del 1 de septiembre de cada año)
    hoy = datetime.now()
    temporada_inicio = datetime(hoy.year, 9, 1)  # Temporada siempre comienza el 1 de septiembre
    if hoy < temporada_inicio:  # Si hoy es antes del 1 de septiembre, estamos en la temporada anterior
        temporada_actual = hoy.year - 1
    else:  # Si estamos después del 1 de septiembre, es la temporada actual
        temporada_actual = hoy.year

    # Calcular el año de referencia para la categoría
    anio_categoria = anio_nacimiento

    # Restar las temporadas desde 2025 (base) hasta la temporada actual
    anio_categoria -= (temporada_actual - 2025)

    # Determinar la categoría basándonos en el año de nacimiento ajustado
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


async def check_minimum_time(swimmer_id, category, distance, style, time_seconds):
    doc = await db.EHMinimas.find_one({
        "prueba": style,
        "distancia": str(distance)
    })

    if not doc:
        return False

    # ejemplo: fem_alevin_2
    key = category

    for t in doc["tiempos"]:
        if key in t:
            min_time = t["time_seconds"]
            return time_seconds < min_time

    return False


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
        if not user_doc:
            raise HTTPException(status_code=401, detail="User not found")

        if isinstance(user_doc.get("created_at"), str):
            user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])

        return User(**user_doc)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ================== AUTH ROUTES ==================

@api_router.post("/auth/login", response_model=LoginResponse)
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


@api_router.post("/auth/register", response_model=User)
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

# ================== USER ROUTES ==================

@api_router.get("/users", response_model=List[User])
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


@api_router.get("/users/{user_id}", response_model=User)
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

    if not user.get("gender"):
        user["gender"] = "fem"


    return User(**user_doc)


@api_router.put("/users/{user_id}", response_model=User)
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
    if "birth_date" in user_data.dict():
        update_doc["birth_date"] = user_data.birth_date
    else:
        # Si no se pasa birth_date, asignar el valor actual
        update_doc["birth_date"] = existing_user["birth_date"] if "birth_date" in existing_user else datetime.now(timezone.utc)

    if user_data.gender is not None:
        update_doc["gender"] = user_data.gender
    else:
        update_doc["gender"] = existing_user.get("gender", "fem")


    result = await db.users.update_one({"id": user_id}, {"$set": update_doc})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    updated_user = await db.users.find_one({"id": user_id}, {"_id": 0, "password": 0})
    if isinstance(updated_user.get("created_at"), str):
        updated_user["created_at"] = datetime.fromisoformat(updated_user["created_at"])

    return User(**updated_user)


@api_router.delete("/users/{user_id}")
async def delete_user(user_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden eliminar usuarios")

    result = await db.users.delete_one({"id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    return {"message": "Usuario eliminado"}

# ================== SWIM TIMES ROUTES ==================

@api_router.post("/times", response_model=SwimTime)
async def create_swim_time(time_data: SwimTimeCreate, current_user: User = Depends(get_current_user)):
    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="Solo entrenadores y administradores pueden registrar tiempos")

    # Calcular pace por 100 metros
    pace_100m = time_data.time_seconds / (time_data.distance / 100) if time_data.distance else 0.0
    gender_value = current_user.gender if current_user.gender else "fem"

    # Obtener categoría y género
    category, gender = get_category_and_gender(current_user.birth_date,gender_value)

    logger.info("category es: %s", category )
    logger.info("gender es: %s", gender )

    # Verificar si es una mínima
    is_minimum = await check_minimum_time(
        swimmer_id=current_user.id,
        category=f"{gender}_{category}",
        distance=time_data.distance,
        style=time_data.style,
        time_seconds=time_data.time_seconds
    )

    # Crear el objeto SwimTime
    time_obj = SwimTime(
        **time_data.model_dump(),
        pace_100m=pace_100m,
        recorded_by=current_user.id,
        minima="si" if is_minimum else "no"
    )

    doc = time_obj.model_dump()
    doc["date"] = doc["date"].isoformat()
    doc["created_at"] = doc["created_at"].isoformat()

    # Insertar el tiempo
    await db.swim_times.insert_one(doc)

    await update_personal_best(
        time_obj.swimmer_id,
        time_obj.distance,
        time_obj.style,
        time_obj.time_seconds,
        time_obj.date,
        time_obj.competition,
    )

    return time_obj

@api_router.get("/times", response_model=List[SwimTime])
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
        # Por si hay documentos antiguos sin pace_100m
        if "pace_100m" not in t and t.get("distance") and t.get("time_seconds") is not None:
            t["pace_100m"] = t["time_seconds"] / (t["distance"] / 100)
        # Incluir la información de mínima
        t["minima"] = t.get("minima", "no")

    times.sort(key=lambda x: x["date"], reverse=True)
    return times

@api_router.put("/times/{time_id}", response_model=SwimTime)
async def update_swim_time(
    time_id: str,
    time_data: SwimTimeCreate,
    current_user: User = Depends(get_current_user),
):
    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="Solo entrenadores y administradores pueden editar tiempos")

    doc = time_data.model_dump()
    doc["date"] = doc["date"].isoformat()
    # recalcular pace_100m en la actualización también
    doc["pace_100m"] = time_data.time_seconds / (time_data.distance / 100) if time_data.distance else 0.0
    
    # Obtener categoría y género del nadador
    swimmer = await db.users.find_one({"id": time_data.swimmer_id})
    gender_value = swimmer.get("gender", "fem") or "fem"

    category, gender = get_category_and_gender(swimmer["birth_date"],gender_value)
    

    logger.info("category es: %s", category )
    logger.info("gender es: %s", gender )

    is_minimum = await check_minimum_time(
        swimmer_id=time_data.swimmer_id,
        category=f"{gender}_{category}",
        distance=time_data.distance,
        style=time_data.style,
        time_seconds=time_data.time_seconds,
    )
    
    doc["minima"] = "si" if is_minimum else "no"

    result = await db.swim_times.update_one({"id": time_id}, {"$set": doc})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Tiempo no encontrado")

    await recalculate_personal_best(time_data.swimmer_id, time_data.distance, time_data.style)

    updated_doc = await db.swim_times.find_one({"id": time_id}, {"_id": 0})
    if isinstance(updated_doc.get("date"), str):
        updated_doc["date"] = datetime.fromisoformat(updated_doc["date"])
    if isinstance(updated_doc.get("created_at"), str):
        updated_doc["created_at"] = datetime.fromisoformat(updated_doc["created_at"])

    return SwimTime(**updated_doc)


@api_router.delete("/times/{time_id}")
async def delete_swim_time(time_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="Solo entrenadores y administradores pueden eliminar tiempos")

    time_doc = await db.swim_times.find_one({"id": time_id})
    if not time_doc:
        raise HTTPException(status_code=404, detail="Tiempo no encontrado")

    await db.swim_times.delete_one({"id": time_id})

    await recalculate_personal_best(time_doc["swimmer_id"], time_doc["distance"], time_doc["style"])

    return {"message": "Tiempo eliminado"}

# ================== PERSONAL BESTS ==================

async def update_personal_best(
    swimmer_id: str,
    distance: int,
    style: str,
    time_seconds: float,
    date: datetime,
    competition: Optional[str],
):
    pb_doc = await db.personal_bests.find_one(
        {"swimmer_id": swimmer_id, "distance": distance, "style": style}
    )

    if not pb_doc or time_seconds < pb_doc["best_time"]:
        doc = {
            "swimmer_id": swimmer_id,
            "distance": distance,
            "style": style,
            "best_time": time_seconds,
            "date": date.isoformat(),
            "competition": competition,
        }
        await db.personal_bests.update_one(
            {"swimmer_id": swimmer_id, "distance": distance, "style": style},
            {"$set": doc},
            upsert=True,
        )


async def recalculate_personal_best(swimmer_id: str, distance: int, style: str):
    times = await db.swim_times.find(
        {"swimmer_id": swimmer_id, "distance": distance, "style": style}
    ).to_list(1000)

    if not times:
        await db.personal_bests.delete_one(
            {"swimmer_id": swimmer_id, "distance": distance, "style": style}
        )
        return

    best_time_doc = min(times, key=lambda x: x["time_seconds"])
    date = (
        best_time_doc["date"]
        if isinstance(best_time_doc["date"], datetime)
        else datetime.fromisoformat(best_time_doc["date"])
    )

    await update_personal_best(
        swimmer_id,
        distance,
        style,
        best_time_doc["time_seconds"],
        date,
        best_time_doc.get("competition"),
    )


@api_router.get("/personal-bests", response_model=List[PersonalBest])
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

# ================== LOCKER ROUTES ==================

@api_router.get("/lockers/{swimmer_id}", response_model=Locker)
async def get_locker(swimmer_id: str, current_user: User = Depends(get_current_user)):
    if current_user.role == "swimmer" and current_user.id != swimmer_id:
        raise HTTPException(status_code=403, detail="No autorizado")

    locker_doc = await db.lockers.find_one({"swimmer_id": swimmer_id}, {"_id": 0})
    if not locker_doc:
        raise HTTPException(status_code=404, detail="Taquilla no encontrada")

    if isinstance(locker_doc.get("updated_at"), str):
        locker_doc["updated_at"] = datetime.fromisoformat(locker_doc["updated_at"])

    return Locker(**locker_doc)


@api_router.post("/lockers", response_model=Locker)
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


@api_router.put("/lockers/{swimmer_id}", response_model=Locker)
async def update_locker(
    swimmer_id: str,
    locker_data: LockerCreate,
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Solo administradores pueden gestionar taquillas")

    doc = locker_data.model_dump()
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()

    result = await db.lockers.update_one({"swimmer_id": swimmer_id}, {"$set": doc})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Taquilla no encontrada")

    updated_doc = await db.lockers.find_one({"swimmer_id": swimmer_id}, {"_id": 0})
    if isinstance(updated_doc.get("updated_at"), str):
        updated_doc["updated_at"] = datetime.fromisoformat(updated_doc["updated_at"])

    return Locker(**updated_doc)

# ================== APP SETUP ==================

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

