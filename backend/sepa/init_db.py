"""Creación idempotente de índices de MongoDB para SEPA.

Se llama desde el evento `startup` de FastAPI. En Mongo `create_index` no
falla si el índice ya existe con las mismas opciones, así que es seguro
ejecutarlo en cada arranque.

Índices críticos:
- Unicidad donde SEPA la exige (mandate_id, message_id) → evita errores con
  el banco, que rechaza ficheros con IDs duplicados.
- Unicidad de (user_id, billing_period) en payments → evita cobrar dos veces
  el mismo mes al mismo nadador si alguien pulsa "Generar cobros" dos veces.
- Compuestos para las queries calientes (pagos pendientes por fecha,
  mandatos activos por usuario).
"""
from __future__ import annotations

import logging

from pymongo import ASCENDING, DESCENDING

logger = logging.getLogger("backend")


async def ensure_indexes(db) -> None:
    """Crea todos los índices SEPA. Idempotente."""

    # --- bank_accounts -----------------------------------------------------
    # Un IBAN por nadador.
    await db.bank_accounts.create_index(
        [("user_id", ASCENDING)],
        unique=True,
        name="uq_bank_accounts_user_id",
    )

    # --- sepa_mandates -----------------------------------------------------
    # mandate_id único (obligatorio por norma SEPA).
    await db.sepa_mandates.create_index(
        [("mandate_id", ASCENDING)],
        unique=True,
        name="uq_sepa_mandates_mandate_id",
    )
    # Búsqueda del mandato activo de un usuario.
    await db.sepa_mandates.create_index(
        [("user_id", ASCENDING), ("status", ASCENDING)],
        name="ix_sepa_mandates_user_status",
    )

    # --- payments ----------------------------------------------------------
    # Evita duplicar el cobro del mismo mes al mismo nadador.
    # sparse: permite múltiples docs sin billing_period (cobros puntuales).
    await db.payments.create_index(
        [("user_id", ASCENDING), ("billing_period", ASCENDING)],
        unique=True,
        sparse=True,
        name="uq_payments_user_period",
    )
    # Query principal del endpoint generate: pendientes por rango de fecha.
    await db.payments.create_index(
        [("status", ASCENDING), ("due_date", ASCENDING)],
        name="ix_payments_status_due",
    )
    # Consulta de pagos de una remesa concreta.
    await db.payments.create_index(
        [("remesa_id", ASCENDING)],
        name="ix_payments_remesa",
    )
    # end_to_end_id único cuando se asigna.
    await db.payments.create_index(
        [("end_to_end_id", ASCENDING)],
        unique=True,
        sparse=True,
        name="uq_payments_e2e",
    )

    # --- remesas -----------------------------------------------------------
    # message_id único (obligatorio por norma SEPA).
    await db.remesas.create_index(
        [("message_id", ASCENDING)],
        unique=True,
        name="uq_remesas_message_id",
    )
    # Histórico ordenado por fecha (UI de admin).
    await db.remesas.create_index(
        [("created_at", DESCENDING)],
        name="ix_remesas_created_at",
    )

    # --- audit_log ---------------------------------------------------------
    # Lectura de logs recientes.
    await db.audit_log.create_index(
        [("created_at", DESCENDING)],
        name="ix_audit_created_at",
    )
    # Consulta por actor (quién hizo qué).
    await db.audit_log.create_index(
        [("actor_user_id", ASCENDING), ("created_at", DESCENDING)],
        name="ix_audit_actor_created",
    )

    logger.info("SEPA: índices verificados/creados")
