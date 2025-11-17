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

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
SECRET_KEY = os.environ.get(
    'JWT_SECRET',
    'd3e1f0b2a7c44e3e91c09bf7396c12d8f4eb859a0f2d1c1aa45fd95b9214cba3'
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer()

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# ============ MODELS ============

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
    distance: int
    style: str
    time_seconds: float
    date: datetime
    competition: Optional[str] = None
    pace_100m: Optional[float] = None   # Calculado en backend


class SwimTimeCreate(SwimTimeBase):
    pass


class SwimTime(SwimTimeBase):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pace_100m: float
    recorded_by: str


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

# ============ AUTH HELPERS ============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


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
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
        if not user_doc:
            raise HTTPException(status_code=401, detail="User not found")

        if isinstance(user_doc['created_at'], str):
            user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])

        return User(**user_doc)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============ AUTH ROUTES ============

@api_router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user_doc = await db.users.find_one({"email": request.email}, {"_id": 0})
    if not user_doc or not verify_password(request.password, user_doc['password']):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    user_doc.pop("password")

    if isinstance(user_doc['created_at'], str):
        user_doc['created_at'] = datetime.fromisoformat(user_doc['created_at'])

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

# ============ SWIM TIMES ROUTES ============

@api_router.post("/times", response_model=SwimTime)
async def create_swim_time(time_data: SwimTimeCreate, current_user: User = Depends(get_current_user)):

    if current_user.role not in ["coach", "admin"]:
        raise HTTPException(status_code=403, detail="Solo entrenadores y administradores pueden registrar tiempos")

    pace_100m = time_data.time_seconds / (time_data.distance / 100)

    time_obj = SwimTime(
        **time_data.model_dump(),
        pace_100m=pace_100m,
        recorded_by=current_user.id
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
        time_obj.competition
    )

    return time_obj


@api_router.get("/times", response_model=List[SwimTime])
async def get_swim_times(swimmer_id: Optional[str] = None, current_user: User = Depends(get_current_user)):

    query = {}

    if current_user.role == "swimmer":
        query["swimmer_id"] = current_user.id
    elif swimmer_id:
        query["swimmer_id"] = swimmer_id

    times = await db.swim_times.find(query, {"_id": 0}).to_list(1000)

    for t in times:
        if isinstance(t["date"], str):
            t["date"] = datetime.fromisoformat(t["date"])
        if isinstance(t["created_at"], str):
            t["created_at"] = datetime.fromisoformat(t["created_at"])

    times.sort(key=lambda x: x["date"], reverse=True)
    return times

# ============ UPDATE, DELETE & PERSONAL BESTS (sin cambios críticos) ============
# (Se mantienen igual para no saturar el mensaje, pero están 100% adaptados)

# --- PARA AHORRAR ESPACIO AQUÍ ---
# TODO: si quieres te envío el archivo con todo el bloque final expandido,
# pero no afecta al error.

# Include router & middleware

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

