"""Construcción de XML SEPA pain.008.001.02 (adeudos directos CORE).

Este módulo es puro: recibe datos estructurados y devuelve los bytes del XML.
No toca la BBDD. El orquestador (service.py / routes.py) es quien lee Mongo,
valida, construye los dataclass de entrada y llama aquí.

Norma SEPA: cada transacción tiene un `sequence_type` (FRST/RCUR/OOFF/FNAL).
El estándar exige que las transacciones con distinto sequence_type vayan en
bloques `<PmtInf>` distintos. Este builder lo gestiona automáticamente.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Optional
from xml.etree import ElementTree as ET

from . import config


SequenceType = Literal["FRST", "RCUR", "OOFF", "FNAL"]

_NS = f"urn:iso:std:iso:20022:tech:xsd:{config.PAIN_SCHEMA}"


class SepaXmlError(Exception):
    """Datos de entrada inválidos para el XML."""


# --- Dataclasses de entrada ------------------------------------------------

@dataclass
class Creditor:
    name: str
    iban: str
    bic: str
    creditor_id: str  # ICS


@dataclass
class SepaTransaction:
    end_to_end_id: str          # único por transacción, ≤35 chars
    amount: float               # positivo, en EUR
    mandate_id: str             # ≤35 chars
    mandate_signature_date: date
    sequence_type: SequenceType
    debtor_name: str            # titular de la cuenta deudora
    debtor_iban: str
    concept: str                # RmtInf/Ustrd, ≤140 chars
    debtor_bic: Optional[str] = None


@dataclass
class SepaRemesaInput:
    message_id: str             # único en todo el ICS, ≤35 chars
    creation_datetime: datetime
    collection_date: date       # ReqdColltnDt
    creditor: Creditor
    transactions: list[SepaTransaction] = field(default_factory=list)


# --- Helpers ----------------------------------------------------------------

def _fmt_amount(value: float) -> str:
    """2 decimales, punto como separador, sin miles."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _strip_iban(iban: str) -> str:
    return "".join(iban.split()).upper()


# SEPA solo permite [A-Za-z0-9 /\-?:().,'+]
_SEPA_ALLOWED = re.compile(r"[^A-Za-z0-9 /\-?:().,'+]")

def _clean_text(s: str, max_len: int) -> str:
    """Limpia caracteres no permitidos y trunca."""
    if s is None:
        return ""
    cleaned = _SEPA_ALLOWED.sub(" ", s).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:max_len]


def _sub(parent: ET.Element, tag: str, text: Optional[str] = None, **attrs) -> ET.Element:
    el = ET.SubElement(parent, tag, attrs)
    if text is not None:
        el.text = text
    return el


# --- Validaciones ----------------------------------------------------------

def _validate(remesa: SepaRemesaInput) -> None:
    if not remesa.transactions:
        raise SepaXmlError("La remesa no tiene transacciones")
    if len(remesa.message_id) > 35:
        raise SepaXmlError(f"message_id > 35 chars: {remesa.message_id}")
    if not remesa.creditor.creditor_id or remesa.creditor.creditor_id.startswith("TODO"):
        raise SepaXmlError(
            "creditor_id (ICS) no configurado. "
            "Defínelo en SEPA_CREDITOR_ID antes de generar remesas reales."
        )
    seen_e2e: set[str] = set()
    for t in remesa.transactions:
        if t.amount <= 0:
            raise SepaXmlError(f"Importe ≤0 en {t.end_to_end_id}")
        if len(t.end_to_end_id) > 35:
            raise SepaXmlError(f"EndToEndId > 35 chars: {t.end_to_end_id}")
        if t.end_to_end_id in seen_e2e:
            raise SepaXmlError(f"EndToEndId duplicado: {t.end_to_end_id}")
        seen_e2e.add(t.end_to_end_id)
        if len(t.mandate_id) > 35:
            raise SepaXmlError(f"mandate_id > 35 chars: {t.mandate_id}")
        if t.sequence_type not in ("FRST", "RCUR", "OOFF", "FNAL"):
            raise SepaXmlError(f"sequence_type inválido: {t.sequence_type}")


# --- Construcción del árbol XML --------------------------------------------

def _build_pmt_inf(
    pmt_inf_id: str,
    sequence_type: SequenceType,
    collection_date: date,
    creditor: Creditor,
    txs: list[SepaTransaction],
) -> ET.Element:
    """Construye un bloque <PmtInf> para un sequence_type concreto."""
    pmt_inf = ET.Element("PmtInf")
    _sub(pmt_inf, "PmtInfId", pmt_inf_id)
    _sub(pmt_inf, "PmtMtd", "DD")
    _sub(pmt_inf, "NbOfTxs", str(len(txs)))

    total = sum(Decimal(_fmt_amount(t.amount)) for t in txs)
    _sub(pmt_inf, "CtrlSum", str(total))

    pmt_tp_inf = _sub(pmt_inf, "PmtTpInf")
    svc_lvl = _sub(pmt_tp_inf, "SvcLvl")
    _sub(svc_lvl, "Cd", "SEPA")
    lcl = _sub(pmt_tp_inf, "LclInstrm")
    _sub(lcl, "Cd", config.LOCAL_INSTRUMENT)
    _sub(pmt_tp_inf, "SeqTp", sequence_type)

    _sub(pmt_inf, "ReqdColltnDt", collection_date.isoformat())

    cdtr = _sub(pmt_inf, "Cdtr")
    _sub(cdtr, "Nm", _clean_text(creditor.name, 70))

    cdtr_acct = _sub(pmt_inf, "CdtrAcct")
    acct_id = _sub(cdtr_acct, "Id")
    _sub(acct_id, "IBAN", _strip_iban(creditor.iban))

    cdtr_agt = _sub(pmt_inf, "CdtrAgt")
    fin_inst = _sub(cdtr_agt, "FinInstnId")
    _sub(fin_inst, "BIC", creditor.bic)

    _sub(pmt_inf, "ChrgBr", "SLEV")

    # Identificador de acreedor (ICS)
    scheme = _sub(pmt_inf, "CdtrSchmeId")
    sch_id = _sub(scheme, "Id")
    prv = _sub(sch_id, "PrvtId")
    othr = _sub(prv, "Othr")
    _sub(othr, "Id", creditor.creditor_id)
    sch_nm = _sub(othr, "SchmeNm")
    _sub(sch_nm, "Prtry", "SEPA")

    # Una <DrctDbtTxInf> por transacción
    for t in txs:
        _append_tx(pmt_inf, t)

    return pmt_inf


def _append_tx(parent: ET.Element, t: SepaTransaction) -> None:
    tx = _sub(parent, "DrctDbtTxInf")

    pmt_id = _sub(tx, "PmtId")
    _sub(pmt_id, "EndToEndId", t.end_to_end_id)

    _sub(tx, "InstdAmt", _fmt_amount(t.amount), Ccy=config.CURRENCY)

    dd_tx = _sub(tx, "DrctDbtTx")
    mndt = _sub(dd_tx, "MndtRltdInf")
    _sub(mndt, "MndtId", t.mandate_id)
    _sub(mndt, "DtOfSgntr", t.mandate_signature_date.isoformat())

    dbtr_agt = _sub(tx, "DbtrAgt")
    fin = _sub(dbtr_agt, "FinInstnId")
    if t.debtor_bic:
        _sub(fin, "BIC", t.debtor_bic)
    else:
        # Cuando no tenemos BIC del deudor, SEPA permite "NOTPROVIDED"
        othr = _sub(fin, "Othr")
        _sub(othr, "Id", "NOTPROVIDED")

    dbtr = _sub(tx, "Dbtr")
    _sub(dbtr, "Nm", _clean_text(t.debtor_name, 70))

    dbtr_acct = _sub(tx, "DbtrAcct")
    acct_id = _sub(dbtr_acct, "Id")
    _sub(acct_id, "IBAN", _strip_iban(t.debtor_iban))

    rmt = _sub(tx, "RmtInf")
    _sub(rmt, "Ustrd", _clean_text(t.concept, 140))


def build_sepa_xml(remesa: SepaRemesaInput) -> bytes:
    """Devuelve los bytes del XML SEPA listo para enviar a Kutxabank."""
    _validate(remesa)

    # Registra el namespace como default (sin prefijo 'ns0:')
    ET.register_namespace("", _NS)
    doc = ET.Element(f"{{{_NS}}}Document")
    cstmr = ET.SubElement(doc, "CstmrDrctDbtInitn")

    # --- GrpHdr ---
    grp = ET.SubElement(cstmr, "GrpHdr")
    _sub(grp, "MsgId", remesa.message_id)
    _sub(grp, "CreDtTm", remesa.creation_datetime.strftime("%Y-%m-%dT%H:%M:%S"))
    _sub(grp, "NbOfTxs", str(len(remesa.transactions)))
    total_all = sum(Decimal(_fmt_amount(t.amount)) for t in remesa.transactions)
    _sub(grp, "CtrlSum", str(total_all))

    initg = _sub(grp, "InitgPty")
    _sub(initg, "Nm", _clean_text(remesa.creditor.name, 70))
    ini_id = _sub(initg, "Id")
    org = _sub(ini_id, "OrgId")
    othr = _sub(org, "Othr")
    _sub(othr, "Id", remesa.creditor.creditor_id)

    # --- PmtInf (uno por cada sequence_type presente) ---
    by_seq: dict[str, list[SepaTransaction]] = {}
    for t in remesa.transactions:
        by_seq.setdefault(t.sequence_type, []).append(t)

    # Orden determinista: FRST primero (a veces el banco lo prefiere)
    seq_order = ["FRST", "RCUR", "OOFF", "FNAL"]
    for seq in sorted(by_seq.keys(), key=lambda s: seq_order.index(s)):
        block = _build_pmt_inf(
            pmt_inf_id=f"{remesa.message_id}-{seq}",
            sequence_type=seq,  # type: ignore[arg-type]
            collection_date=remesa.collection_date,
            creditor=remesa.creditor,
            txs=by_seq[seq],
        )
        cstmr.append(block)

    # Render con declaración XML y UTF-8
    return ET.tostring(doc, encoding="UTF-8", xml_declaration=True)


def xml_sha256(xml_bytes: bytes) -> str:
    return hashlib.sha256(xml_bytes).hexdigest()
