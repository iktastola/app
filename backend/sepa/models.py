"""Modelos Pydantic para el módulo SEPA.

Convenciones:
- `id` UUID propio (igual que el resto del backend) para no depender de _id.
- Fechas en UTC (datetime con tz) para registros; date pura para fechas de
  negocio (firma, vencimiento, fecha de cargo).
- IBAN del deudor: NUNCA en claro en la BBDD → solo `iban_encrypted` +
  `iban_last4`. Los modelos *Create reciben el IBAN en claro (por HTTPS) y el
  service lo cifra antes de guardar.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SequenceType = Literal["FRST", "RCUR", "OOFF", "FNAL"]
MandateStatus = Literal["active", "cancelled", "expired"]
PaymentStatus = Literal["pending", "in_remesa", "paid", "returned", "failed"]
RemesaStatus = Literal["generated", "sent", "partially_returned", "closed"]


# =========================== BankAccount ==================================

class BankAccountCreate(BaseModel):
    """Entrada del endpoint: admin guarda el IBAN de un nadador."""
    user_id: str
    iban: str                       # en claro, se cifra al guardar
    holder_name: str                # titular de la cuenta (puede != nadador)
    bic: Optional[str] = None       # opcional; pain.008.001.02 no lo exige


class BankAccount(BaseModel):
    """Lo que se guarda en la colección `bank_accounts`.

    `iban_encrypted` es el token AES-GCM. `iban_last4` es público (para UI).
    """
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    iban_encrypted: str
    iban_last4: str
    holder_name: str
    bic: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BankAccountPublic(BaseModel):
    """Lo que devuelve la API al frontend (sin IBAN en claro)."""
    id: str
    user_id: str
    iban_masked: str                # 'ES76 **** **** **** 5766'
    iban_last4: str
    holder_name: str
    bic: Optional[str] = None
    updated_at: datetime


# =========================== Mandate ======================================

class MandateCreate(BaseModel):
    user_id: str
    mandate_id: Optional[str] = None  # si no se da, se genera
    signature_date: date
    type: SequenceType = "RCUR"       # RCUR para cuotas recurrentes


class Mandate(BaseModel):
    """Autorización firmada del nadador para que el club le cobre.

    Sin mandato activo, un nadador NO puede entrar en remesa.
    """
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    mandate_id: str                 # identificador enviado al banco (único)
    signature_date: date
    type: SequenceType = "RCUR"
    status: MandateStatus = "active"
    first_used_at: Optional[datetime] = None  # se rellena tras primer FRST
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cancelled_at: Optional[datetime] = None


class MandateBulkImport(BaseModel):
    """Para cargar mandatos ya firmados (PDF/papel) de golpe."""
    mandates: List[MandateCreate]


# =========================== Payment ======================================

class PaymentCreate(BaseModel):
    user_id: str
    amount: float
    concept: str
    due_date: date
    billing_period: Optional[str] = None  # '2026-05' para cuotas mensuales


class Payment(BaseModel):
    """Un cobro pendiente. Al generar remesa pasa a `in_remesa`."""
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    amount: float
    currency: str = "EUR"
    concept: str
    due_date: date
    billing_period: Optional[str] = None
    status: PaymentStatus = "pending"
    sequence_type: Optional[SequenceType] = None  # se fija al meter en remesa
    remesa_id: Optional[str] = None
    end_to_end_id: Optional[str] = None           # único por transacción SEPA
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    returned_at: Optional[datetime] = None
    return_reason: Optional[str] = None


# =========================== Remesa =======================================

class RemesaGenerateRequest(BaseModel):
    """Body del endpoint que el frontend llama con el botón."""
    date_from: date                 # pagos con due_date >= date_from
    date_to: date                   # pagos con due_date <= date_to
    collection_date: Optional[date] = None  # si no, se calcula con plazos


class RemesaExcludedPayment(BaseModel):
    payment_id: str
    user_id: str
    reason: str                     # 'no_iban', 'no_mandate', 'mandate_cancelled', ...


class Remesa(BaseModel):
    """Fichero SEPA generado. Guarda metadatos para auditoría."""
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str                 # MsgId del XML (único)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str                 # user_id del admin que la generó
    collection_date: date
    total_amount: float
    n_txs: int
    xml_sha256: str                 # hash del XML para detectar reenvíos
    status: RemesaStatus = "generated"
    payment_ids: List[str] = Field(default_factory=list)
    excluded: List[RemesaExcludedPayment] = Field(default_factory=list)


class RemesaSummary(BaseModel):
    """Resumen que se devuelve junto con el XML al frontend."""
    remesa_id: str
    message_id: str
    collection_date: date
    total_amount: float
    n_txs: int
    n_excluded: int
    excluded: List[RemesaExcludedPayment] = Field(default_factory=list)
