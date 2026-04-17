"""Orquestación de generación de remesas SEPA.

Une los módulos: lee Mongo, valida cada pago (IBAN + mandato activo),
construye el input del xml_builder, llama al builder, persiste la remesa y
marca los pagos como `in_remesa`.

Este módulo hace los accesos a BBDD; el xml_builder es puro.
"""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, time, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from . import config, crypto, xml_builder

logger = logging.getLogger("backend")


class RemesaError(Exception):
    """Error de orquestación (sin pagos, ICS no configurado, ...)."""


def _date_to_dt(d: date) -> datetime:
    return datetime.combine(d, time.min, tzinfo=timezone.utc)


def _dt_to_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    raise ValueError(f"No se puede convertir {v!r} a date")


def _message_id() -> str:
    """AST-YYYYMMDDHHMMSS-HHHHHH (24 chars, bajo el límite de 35)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"AST-{ts}-{uuid.uuid4().hex[:6]}"


def _end_to_end_id(remesa_msg_id: str, n: int) -> str:
    """E2E único y estable dentro de la remesa."""
    # remesa_msg_id=24, -Nxxx ≤ 11 → total ≤ 35
    return f"{remesa_msg_id}-{n:05d}"


async def generate_remesa(
    db,
    date_from: date,
    date_to: date,
    collection_date: date | None,
    actor_user_id: str,
) -> dict[str, Any]:
    """Genera una remesa con los pagos pendientes en el rango indicado.

    Devuelve dict con `remesa_doc`, `xml_bytes`, `summary`, `excluded`.
    """
    if date_from > date_to:
        raise RemesaError("date_from debe ser ≤ date_to")

    # Busca pagos pendientes en el rango
    payments = await db.payments.find(
        {
            "status": "pending",
            "due_date": {"$gte": _date_to_dt(date_from), "$lte": _date_to_dt(date_to)},
        },
        {"_id": 0},
    ).sort("due_date", 1).to_list(5000)

    if not payments:
        raise RemesaError(
            f"No hay pagos pendientes entre {date_from} y {date_to}"
        )

    # Config del acreedor
    creditor = xml_builder.Creditor(
        name=config.CREDITOR_NAME,
        iban=config.CREDITOR_IBAN,
        bic=config.CREDITOR_BIC,
        creditor_id=config.CREDITOR_ID,
    )

    msg_id = _message_id()
    coll_date = collection_date or date_to

    # Construir transacciones + listar excluidos
    txs: list[xml_builder.SepaTransaction] = []
    included_payments: list[dict] = []
    excluded: list[dict] = []
    mandates_to_mark_first_used: list[str] = []

    for idx, pmt in enumerate(payments, start=1):
        uid = pmt["user_id"]

        user = await db.users.find_one({"id": uid}, {"_id": 0, "name": 1, "email": 1})
        if not user:
            excluded.append({"payment_id": pmt["id"], "user_id": uid, "reason": "user_not_found"})
            continue

        bank = await db.bank_accounts.find_one({"user_id": uid}, {"_id": 0})
        if not bank:
            excluded.append({"payment_id": pmt["id"], "user_id": uid, "reason": "no_iban"})
            continue

        mandate = await db.sepa_mandates.find_one(
            {"user_id": uid, "status": "active"}, {"_id": 0}
        )
        if not mandate:
            excluded.append({"payment_id": pmt["id"], "user_id": uid, "reason": "no_active_mandate"})
            continue

        try:
            iban_plain = crypto.decrypt_iban(bank["iban_encrypted"])
        except Exception as e:
            logger.error(f"No se pudo descifrar IBAN de {uid}: {e}")
            excluded.append({"payment_id": pmt["id"], "user_id": uid, "reason": "iban_decrypt_error"})
            continue

        # FRST si el mandato no se ha usado todavía; RCUR si ya
        seq_type = "FRST" if mandate.get("first_used_at") is None else "RCUR"
        if seq_type == "FRST":
            mandates_to_mark_first_used.append(mandate["mandate_id"])

        e2e_id = _end_to_end_id(msg_id, idx)

        txs.append(xml_builder.SepaTransaction(
            end_to_end_id=e2e_id,
            amount=float(pmt["amount"]),
            mandate_id=mandate["mandate_id"],
            mandate_signature_date=_dt_to_date(mandate["signature_date"]),
            sequence_type=seq_type,
            debtor_name=bank.get("holder_name") or user.get("name", ""),
            debtor_iban=iban_plain,
            debtor_bic=bank.get("bic"),
            concept=pmt.get("concept", ""),
        ))
        included_payments.append({
            **pmt,
            "_seq_type": seq_type,
            "_e2e_id": e2e_id,
        })

    if not txs:
        raise RemesaError(
            f"Ningún pago pasó las validaciones (excluidos: {len(excluded)})"
        )

    # Construir el XML (lanza SepaXmlError si algo está mal)
    remesa_input = xml_builder.SepaRemesaInput(
        message_id=msg_id,
        creation_datetime=datetime.now(timezone.utc),
        collection_date=coll_date,
        creditor=creditor,
        transactions=txs,
    )
    xml_bytes = xml_builder.build_sepa_xml(remesa_input)
    xml_hash = xml_builder.xml_sha256(xml_bytes)

    # Cuadrar total
    total = sum(Decimal(str(t.amount)) for t in txs).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    # Persistir la remesa
    remesa_id = str(uuid.uuid4())
    remesa_doc = {
        "id": remesa_id,
        "message_id": msg_id,
        "created_at": datetime.now(timezone.utc),
        "created_by": actor_user_id,
        "collection_date": _date_to_dt(coll_date),
        "date_from": _date_to_dt(date_from),
        "date_to": _date_to_dt(date_to),
        "total_amount": float(total),
        "n_txs": len(txs),
        "xml_sha256": xml_hash,
        "xml_bytes": xml_bytes,   # binario; Mongo lo guarda como Binary
        "status": "generated",
        "payment_ids": [p["id"] for p in included_payments],
        "excluded": excluded,
    }
    await db.remesas.insert_one(remesa_doc)

    # Marcar pagos como in_remesa (bulk)
    from pymongo import UpdateOne
    ops = []
    for p in included_payments:
        ops.append(UpdateOne(
            {"id": p["id"]},
            {"$set": {
                "status": "in_remesa",
                "remesa_id": remesa_id,
                "end_to_end_id": p["_e2e_id"],
                "sequence_type": p["_seq_type"],
            }},
        ))
    if ops:
        try:
            await db.payments.bulk_write(ops, ordered=False)
        except Exception as e:
            logger.error(f"Fallo al marcar pagos in_remesa (remesa {remesa_id}): {e}")

    # Marcar mandatos usados por primera vez (bulk)
    if mandates_to_mark_first_used:
        try:
            await db.sepa_mandates.update_many(
                {"mandate_id": {"$in": mandates_to_mark_first_used}},
                {"$set": {"first_used_at": datetime.now(timezone.utc)}},
            )
        except Exception as e:
            logger.error(f"Fallo al marcar mandatos first_used_at: {e}")

    # Auditoría
    await db.audit_log.insert_one({
        "id": str(uuid.uuid4()),
        "actor_user_id": actor_user_id,
        "action": "remesa.generate",
        "target": remesa_id,
        "meta": {
            "message_id": msg_id,
            "n_txs": len(txs),
            "total": float(total),
            "n_excluded": len(excluded),
        },
        "created_at": datetime.now(timezone.utc),
    })

    # Aviso si faltan días hábiles de antelación
    warnings: list[str] = []
    lead = (coll_date - date.today()).days
    has_frst = any(t.sequence_type == "FRST" for t in txs)
    min_lead = config.LEAD_TIME_DAYS_FRST if has_frst else config.LEAD_TIME_DAYS_RCUR
    if lead < min_lead:
        warnings.append(
            f"Margen de {lead} días hasta {coll_date}; SEPA recomienda "
            f"al menos {min_lead} para {'FRST' if has_frst else 'RCUR'}."
        )

    return {
        "remesa_id": remesa_id,
        "message_id": msg_id,
        "collection_date": coll_date.isoformat(),
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "total_amount": float(total),
        "n_txs": len(txs),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "warnings": warnings,
        "xml_sha256": xml_hash,
    }
