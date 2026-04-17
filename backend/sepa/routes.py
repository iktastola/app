"""Endpoints HTTP del módulo SEPA.

Factory pattern: server.py llama a `build_sepa_router(db, get_current_user)`
después de tener ambos listos. Evita imports circulares.

Todos los endpoints requieren rol `admin`. El acceso al IBAN en claro (/full)
deja traza en `audit_log`.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, time, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from . import crypto
from .models import (
    BankAccount,
    BankAccountCreate,
    BankAccountPublic,
    Mandate,
    MandateBulkImport,
    MandateCreate,
    MandateStatus,
    Payment,
    PaymentCreate,
    PaymentStatus,
)


# --- utilidades fechas ------------------------------------------------------

def _date_to_dt(d: date) -> datetime:
    """pymongo no acepta datetime.date → lo subimos a datetime UTC midnight."""
    return datetime.combine(d, time.min, tzinfo=timezone.utc)


def _dt_to_date(v) -> date:
    """Inverso: lo que Mongo devuelve es datetime; lo pasamos a date."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    raise ValueError(f"No se puede convertir {v!r} a date")


# ============================================================================

def build_sepa_router(db, get_current_user) -> APIRouter:
    """Construye el APIRouter SEPA con las dependencias inyectadas."""

    router = APIRouter(prefix="/api/sepa", tags=["sepa"])

    # --- dependencias ------------------------------------------------------

    async def require_admin(current_user=Depends(get_current_user)):
        if getattr(current_user, "role", None) != "admin":
            raise HTTPException(status_code=403, detail="Solo admin")
        return current_user

    async def _audit(actor_user_id: str, action: str, target: str, meta: dict | None = None):
        await db.audit_log.insert_one({
            "id": str(uuid.uuid4()),
            "actor_user_id": actor_user_id,
            "action": action,
            "target": target,
            "meta": meta or {},
            "created_at": datetime.now(timezone.utc),
        })

    async def _user_exists(user_id: str) -> bool:
        return await db.users.find_one({"id": user_id}, {"_id": 1}) is not None

    # ========================= BANK ACCOUNTS ==============================

    @router.post("/bank-accounts", response_model=BankAccountPublic)
    async def create_bank_account(
        payload: BankAccountCreate,
        admin=Depends(require_admin),
    ):
        if not await _user_exists(payload.user_id):
            raise HTTPException(404, "Nadador no encontrado")

        try:
            iban_encrypted = crypto.encrypt_iban(payload.iban)
        except crypto.SepaCryptoError as e:
            raise HTTPException(400, f"IBAN inválido: {e}")

        iban_norm = crypto.validate_iban(payload.iban)
        last4 = crypto.iban_last4(iban_norm)

        now = datetime.now(timezone.utc)
        doc = {
            "id": str(uuid.uuid4()),
            "user_id": payload.user_id,
            "iban_encrypted": iban_encrypted,
            "iban_last4": last4,
            "holder_name": payload.holder_name,
            "bic": payload.bic,
            "created_at": now,
            "updated_at": now,
        }

        # upsert: un IBAN por nadador (uniq index lo fuerza)
        await db.bank_accounts.update_one(
            {"user_id": payload.user_id},
            {"$set": doc},
            upsert=True,
        )

        await _audit(admin.id, "bank_account.upsert", payload.user_id, {"last4": last4})

        return BankAccountPublic(
            id=doc["id"],
            user_id=doc["user_id"],
            iban_masked=crypto.mask_iban(iban_norm),
            iban_last4=last4,
            holder_name=doc["holder_name"],
            bic=doc["bic"],
            updated_at=now,
        )

    @router.get("/bank-accounts", response_model=List[BankAccountPublic])
    async def list_bank_accounts(admin=Depends(require_admin)):
        cursor = db.bank_accounts.find({}, {"_id": 0}).sort("updated_at", -1)
        out: list[BankAccountPublic] = []
        async for d in cursor:
            # masked = "ESxx **** **** **** NNNN" sin desencriptar
            masked = f"ES** **** **** **** **** {d['iban_last4']}"
            out.append(BankAccountPublic(
                id=d["id"],
                user_id=d["user_id"],
                iban_masked=masked,
                iban_last4=d["iban_last4"],
                holder_name=d["holder_name"],
                bic=d.get("bic"),
                updated_at=d["updated_at"],
            ))
        return out

    @router.get("/bank-accounts/{user_id}", response_model=BankAccountPublic)
    async def get_bank_account(user_id: str, admin=Depends(require_admin)):
        d = await db.bank_accounts.find_one({"user_id": user_id}, {"_id": 0})
        if not d:
            raise HTTPException(404, "No hay IBAN para ese nadador")

        iban_plain = crypto.decrypt_iban(d["iban_encrypted"])
        return BankAccountPublic(
            id=d["id"],
            user_id=d["user_id"],
            iban_masked=crypto.mask_iban(iban_plain),
            iban_last4=d["iban_last4"],
            holder_name=d["holder_name"],
            bic=d.get("bic"),
            updated_at=d["updated_at"],
        )

    @router.get("/bank-accounts/{user_id}/full")
    async def get_bank_account_full(user_id: str, admin=Depends(require_admin)):
        """Devuelve IBAN en claro. Deja log de auditoría."""
        d = await db.bank_accounts.find_one({"user_id": user_id}, {"_id": 0})
        if not d:
            raise HTTPException(404, "No hay IBAN para ese nadador")

        iban_plain = crypto.decrypt_iban(d["iban_encrypted"])
        await _audit(admin.id, "bank_account.read_full", user_id, {"last4": d["iban_last4"]})

        return {
            "user_id": user_id,
            "iban": iban_plain,
            "holder_name": d["holder_name"],
            "bic": d.get("bic"),
        }

    @router.delete("/bank-accounts/{user_id}")
    async def delete_bank_account(user_id: str, admin=Depends(require_admin)):
        res = await db.bank_accounts.delete_one({"user_id": user_id})
        if res.deleted_count == 0:
            raise HTTPException(404, "No hay IBAN para ese nadador")
        await _audit(admin.id, "bank_account.delete", user_id)
        return {"deleted": True}

    # ============================= MANDATES ===============================

    def _new_mandate_id() -> str:
        return f"MNDT-{uuid.uuid4()}"

    @router.post("/mandates", response_model=Mandate)
    async def create_mandate(payload: MandateCreate, admin=Depends(require_admin)):
        if not await _user_exists(payload.user_id):
            raise HTTPException(404, "Nadador no encontrado")

        mandate_id = payload.mandate_id or _new_mandate_id()
        doc = {
            "id": str(uuid.uuid4()),
            "user_id": payload.user_id,
            "mandate_id": mandate_id,
            "signature_date": _date_to_dt(payload.signature_date),
            "type": payload.type,
            "status": "active",
            "first_used_at": None,
            "created_at": datetime.now(timezone.utc),
            "cancelled_at": None,
        }
        try:
            await db.sepa_mandates.insert_one(doc)
        except Exception as e:
            # duplicate key por el índice único
            if "duplicate" in str(e).lower() or "E11000" in str(e):
                raise HTTPException(409, f"mandate_id ya existe: {mandate_id}")
            raise

        await _audit(admin.id, "mandate.create", payload.user_id, {"mandate_id": mandate_id})

        return Mandate(
            id=doc["id"],
            user_id=doc["user_id"],
            mandate_id=mandate_id,
            signature_date=_dt_to_date(doc["signature_date"]),
            type=doc["type"],
            status=doc["status"],
            created_at=doc["created_at"],
        )

    @router.post("/mandates/bulk-import")
    async def bulk_import_mandates(
        payload: MandateBulkImport,
        admin=Depends(require_admin),
    ):
        created, failed = [], []
        for m in payload.mandates:
            if not await _user_exists(m.user_id):
                failed.append({"user_id": m.user_id, "reason": "user_not_found"})
                continue
            mandate_id = m.mandate_id or _new_mandate_id()
            doc = {
                "id": str(uuid.uuid4()),
                "user_id": m.user_id,
                "mandate_id": mandate_id,
                "signature_date": _date_to_dt(m.signature_date),
                "type": m.type,
                "status": "active",
                "first_used_at": None,
                "created_at": datetime.now(timezone.utc),
                "cancelled_at": None,
            }
            try:
                await db.sepa_mandates.insert_one(doc)
                created.append({"user_id": m.user_id, "mandate_id": mandate_id})
            except Exception as e:
                failed.append({
                    "user_id": m.user_id,
                    "mandate_id": mandate_id,
                    "reason": str(e)[:120],
                })

        await _audit(admin.id, "mandate.bulk_import", "all",
                     {"created": len(created), "failed": len(failed)})
        return {"created": created, "failed": failed}

    @router.get("/mandates", response_model=List[Mandate])
    async def list_mandates(
        user_id: Optional[str] = Query(None),
        status: Optional[MandateStatus] = Query(None),
        admin=Depends(require_admin),
    ):
        q: dict = {}
        if user_id:
            q["user_id"] = user_id
        if status:
            q["status"] = status
        docs = await db.sepa_mandates.find(q, {"_id": 0}).sort("created_at", -1).to_list(1000)
        out: list[Mandate] = []
        for d in docs:
            d["signature_date"] = _dt_to_date(d["signature_date"])
            out.append(Mandate(**d))
        return out

    @router.delete("/mandates/{mandate_id}")
    async def cancel_mandate(mandate_id: str, admin=Depends(require_admin)):
        """Cancelación, NO borrado (son documentos legales)."""
        res = await db.sepa_mandates.update_one(
            {"mandate_id": mandate_id, "status": "active"},
            {"$set": {"status": "cancelled", "cancelled_at": datetime.now(timezone.utc)}},
        )
        if res.matched_count == 0:
            raise HTTPException(404, "Mandato no encontrado o ya cancelado")
        await _audit(admin.id, "mandate.cancel", mandate_id)
        return {"cancelled": True}

    # ============================= PAYMENTS ===============================

    @router.post("/payments", response_model=Payment)
    async def create_payment(payload: PaymentCreate, admin=Depends(require_admin)):
        if not await _user_exists(payload.user_id):
            raise HTTPException(404, "Nadador no encontrado")
        if payload.amount <= 0:
            raise HTTPException(400, "El importe debe ser > 0")

        doc = {
            "id": str(uuid.uuid4()),
            "user_id": payload.user_id,
            "amount": round(float(payload.amount), 2),
            "currency": "EUR",
            "concept": payload.concept,
            "due_date": _date_to_dt(payload.due_date),
            "billing_period": payload.billing_period,
            "status": "pending",
            "sequence_type": None,
            "remesa_id": None,
            "end_to_end_id": None,
            "created_at": datetime.now(timezone.utc),
            "returned_at": None,
            "return_reason": None,
        }
        try:
            await db.payments.insert_one(doc)
        except Exception as e:
            if "duplicate" in str(e).lower() or "E11000" in str(e):
                raise HTTPException(
                    409,
                    f"Ya existe un pago para {payload.user_id} en {payload.billing_period}",
                )
            raise

        return Payment(
            id=doc["id"],
            user_id=doc["user_id"],
            amount=doc["amount"],
            concept=doc["concept"],
            due_date=_dt_to_date(doc["due_date"]),
            billing_period=doc["billing_period"],
            status=doc["status"],
            created_at=doc["created_at"],
        )

    @router.get("/payments", response_model=List[Payment])
    async def list_payments(
        user_id: Optional[str] = Query(None),
        status: Optional[PaymentStatus] = Query(None),
        date_from: Optional[date] = Query(None),
        date_to: Optional[date] = Query(None),
        admin=Depends(require_admin),
    ):
        q: dict = {}
        if user_id:
            q["user_id"] = user_id
        if status:
            q["status"] = status
        if date_from or date_to:
            q["due_date"] = {}
            if date_from:
                q["due_date"]["$gte"] = _date_to_dt(date_from)
            if date_to:
                q["due_date"]["$lte"] = _date_to_dt(date_to)

        docs = await db.payments.find(q, {"_id": 0}).sort("due_date", 1).to_list(2000)
        out: list[Payment] = []
        for d in docs:
            d["due_date"] = _dt_to_date(d["due_date"])
            out.append(Payment(**d))
        return out

    @router.delete("/payments/{payment_id}")
    async def delete_payment(payment_id: str, admin=Depends(require_admin)):
        """Solo se puede borrar si está pendiente."""
        existing = await db.payments.find_one({"id": payment_id}, {"status": 1})
        if not existing:
            raise HTTPException(404, "Pago no encontrado")
        if existing.get("status") != "pending":
            raise HTTPException(
                409,
                f"No se puede borrar un pago en estado '{existing.get('status')}'",
            )
        await db.payments.delete_one({"id": payment_id})
        await _audit(admin.id, "payment.delete", payment_id)
        return {"deleted": True}

    return router
