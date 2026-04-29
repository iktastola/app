"""Facturación mensual de cuotas.

Genera los cobros pendientes (45€ por nadador) para un mes concreto de la
temporada. Idempotente gracias al índice único (user_id, billing_period) en
la colección `payments`: llamar dos veces con el mismo mes no duplica nada.

Regla de negocio:
- Solo usuarios con role="swimmer".
- Solo si tienen bank_account Y un mandato activo.
- Los meses permitidos son los de BILLING_MONTHS (sep–jun).
- due_date = día 1 del mes facturado.
"""
from __future__ import annotations

import re
import uuid
from datetime import date, datetime, time, timezone

from . import config


_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")


class BillingError(Exception):
    """Error de parseo o de política de negocio."""


def _parse_month(month: str) -> tuple[int, int, date]:
    """'2026-05' → (2026, 5, date(2026,5,1)). Valida formato y rango."""
    m = _MONTH_RE.match(month or "")
    if not m:
        raise BillingError(f"Formato de mes inválido: {month!r}. Usa 'YYYY-MM'.")
    year, month_int = int(m.group(1)), int(m.group(2))
    if not (1 <= month_int <= 12):
        raise BillingError(f"Mes fuera de rango: {month_int}")
    if month_int not in config.BILLING_MONTHS:
        raise BillingError(
            f"El mes {month_int:02d} no está en el calendario de cobros "
            f"(permitidos: {config.BILLING_MONTHS})"
        )
    return year, month_int, date(year, month_int, 1)


async def run_monthly_billing(db, month: str, actor_user_id: str) -> dict:
    """Crea los `payments` de un mes. Devuelve resumen detallado."""
    year, month_int, due_date = _parse_month(month)
    due_dt = datetime.combine(due_date, time.min, tzinfo=timezone.utc)
    concept = f"Cuota {month_int:02d}/{year}"

    created: list[dict] = []
    already_billed: list[dict] = []
    missing_iban: list[dict] = []
    missing_mandate: list[dict] = []

    swimmers = await db.users.find(
        {"role": "swimmer"},
        {"_id": 0, "id": 1, "name": 1, "email": 1, "monthly_fee": 1},
    ).to_list(5000)

    for s in swimmers:
        uid = s["id"]
        brief = {"user_id": uid, "name": s.get("name"), "email": s.get("email")}

        if not await db.bank_accounts.find_one({"user_id": uid}, {"_id": 1}):
            missing_iban.append(brief)
            continue
        if not await db.sepa_mandates.find_one(
            {"user_id": uid, "status": "active"}, {"_id": 1}
        ):
            missing_mandate.append(brief)
            continue

        # Cuota personalizada del nadador, o tarifa por defecto si no tiene
        fee_override = s.get("monthly_fee")
        amount = round(float(fee_override), 2) if fee_override is not None else config.MONTHLY_FEE_EUR

        doc = {
            "id": str(uuid.uuid4()),
            "user_id": uid,
            "amount": amount,
            "currency": config.CURRENCY,
            "concept": concept,
            "due_date": due_dt,
            "billing_period": month,
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
            created.append({**brief, "payment_id": doc["id"], "amount": amount})
        except Exception as e:
            # El índice único (user_id, billing_period) dispara E11000 si ya
            # se facturó este mes a este usuario → lo saltamos.
            if "E11000" in str(e) or "duplicate" in str(e).lower():
                already_billed.append(brief)
            else:
                raise

    await db.audit_log.insert_one({
        "id": str(uuid.uuid4()),
        "actor_user_id": actor_user_id,
        "action": "billing.run",
        "target": month,
        "meta": {
            "created": len(created),
            "already_billed": len(already_billed),
            "missing_iban": len(missing_iban),
            "missing_mandate": len(missing_mandate),
        },
        "created_at": datetime.now(timezone.utc),
    })

    return {
        "month": month,
        "due_date": due_date.isoformat(),
        "amount_each": config.MONTHLY_FEE_EUR,  # tarifa por defecto (los nadadores con cuota propia se aplican individualmente)
        "default_fee": config.MONTHLY_FEE_EUR,
        "total_swimmers": len(swimmers),
        "created": len(created),
        "already_billed": len(already_billed),
        "missing_iban": len(missing_iban),
        "missing_mandate": len(missing_mandate),
        "details": {
            "created": created,
            "already_billed": already_billed,
            "missing_iban": missing_iban,
            "missing_mandate": missing_mandate,
        },
    }
