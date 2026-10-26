import os

from fastapi import FastAPI, APIRouter
from starlette.middleware.cors import CORSMiddleware

from config import db, client, logger
from deps import get_current_user
from routers import auth, users, times, lockers, admin, minimas

app = FastAPI()
api_router = APIRouter(prefix="/api")


# ================== HEALTH ==================

@api_router.get("/health")
async def health():
    """Endpoint público, sin auth ni BBDD. Para keepalive/monitoring."""
    return {"ok": True}


# ================== MOUNT ROUTERS ==================

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(times.router)
api_router.include_router(lockers.router)
api_router.include_router(admin.router)
api_router.include_router(minimas.router)

app.include_router(api_router)

# SEPA (factory pattern: inyecta db + auth ya disponibles)
from sepa.routes import build_sepa_router
app.include_router(build_sepa_router(db, get_current_user))

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_sepa_indexes():
    from sepa.init_db import ensure_indexes
    try:
        await ensure_indexes(db)
    except Exception as e:
        logger.error(f"SEPA: fallo creando índices: {e}")


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
