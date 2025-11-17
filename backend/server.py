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

# ================== APP ==================

app = FastAPI()
api_router = APIRouter(prefix="/api")

# ================== MODELOS ==================

class UserBase(BaseModel):
    email: EmailStr
    name: str
    role: str  # swimmer, coach, admin


class UserCreate(UserBase):
    password: str


class User(UserBase):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


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

    time_obj = SwimTime(
        **time_data.model_dump(),
        pace_100m=pace_100m,
        recorded_by=current_user.id,
    )

    doc = time_obj.model_dump()
    doc["date"] = doc["date"].isoformat()
    doc["created_at"] = doc["created_at"].isoformat()

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
        # recorded_by puede faltar, y está permitido en el modelo

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

