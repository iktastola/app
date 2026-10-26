import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    email: EmailStr
    name: str
    role: str
    birth_date: Optional[datetime] = None
    gender: Optional[str] = None
    avatar_url: Optional[str] = None
    monthly_fee: Optional[float] = None  # cuota mensual personalizada; None = usar tarifa por defecto


class UserCreate(UserBase):
    password: str


class User(UserBase):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    birth_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gender: str = "fem"


class UserSelfUpdate(BaseModel):
    password: Optional[str] = None
    birth_date: Optional[datetime] = None
    avatar_url: Optional[str] = None


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
    oficial: bool = True
    piscina_metros: int = 25


class SwimTimeCreate(SwimTimeBase):
    pass


class SwimTime(SwimTimeBase):
    model_config = ConfigDict(extra="ignore")

    pace_100m: float = 0.0
    recorded_by: Optional[str] = None
    minima: str = "no"
    minima_bizkaia: str = "no"

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


class MinimasImport(BaseModel):
    collection: str
    documentos: List[dict]
