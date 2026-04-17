"""Cifrado de IBAN y validación.

Usa AES-256-GCM (autenticado) con una clave de 32 bytes leída de
SEPA_ENCRYPTION_KEY. Cada cifrado genera un nonce aleatorio de 12 bytes que
se concatena delante del ciphertext y se codifica todo en base64-url.

Formato guardado en BBDD:
    base64url( nonce(12B) || ciphertext || tag(16B) )

También incluye validación IBAN por checksum mod-97 (ISO 13616), que detecta
erratas típicas (dígitos mal copiados, transposiciones) antes de aceptar.
"""
from __future__ import annotations

import base64
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import config

_NONCE_BYTES = 12


class SepaCryptoError(Exception):
    """Error de cifrado/descifrado o IBAN inválido."""


# --- Clave ------------------------------------------------------------------

def _load_key() -> bytes:
    raw = config.ENCRYPTION_KEY_B64
    if not raw:
        raise SepaCryptoError(
            "SEPA_ENCRYPTION_KEY no está definida. "
            "Genera una con: python -c \"import secrets,base64; "
            "print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())\""
        )
    try:
        key = base64.urlsafe_b64decode(raw)
    except Exception as e:
        raise SepaCryptoError(f"SEPA_ENCRYPTION_KEY no es base64 válido: {e}")
    if len(key) != 32:
        raise SepaCryptoError(
            f"SEPA_ENCRYPTION_KEY debe decodificar a 32 bytes (AES-256), no {len(key)}"
        )
    return key


# Cache de la clave + cifrador (se instancia 1 vez por proceso).
_cipher: AESGCM | None = None


def _get_cipher() -> AESGCM:
    global _cipher
    if _cipher is None:
        _cipher = AESGCM(_load_key())
    return _cipher


# --- IBAN: normalización y validación --------------------------------------

def normalize_iban(iban: str) -> str:
    """Quita espacios y pasa a mayúsculas. No valida."""
    if iban is None:
        raise SepaCryptoError("IBAN vacío")
    return "".join(iban.split()).upper()


def validate_iban(iban: str) -> str:
    """Valida formato + checksum mod-97. Devuelve el IBAN normalizado.

    Lanza SepaCryptoError si es inválido.
    """
    norm = normalize_iban(iban)

    if len(norm) < 15 or len(norm) > 34:
        raise SepaCryptoError(f"IBAN longitud inválida: {len(norm)}")
    if not norm[:2].isalpha() or not norm[2:4].isdigit():
        raise SepaCryptoError("IBAN: los 4 primeros caracteres deben ser 2 letras + 2 dígitos")
    if not norm[4:].isalnum():
        raise SepaCryptoError("IBAN contiene caracteres no válidos")

    # Mover los 4 primeros al final y convertir letras a dígitos (A=10, B=11, ...)
    rearranged = norm[4:] + norm[:4]
    numeric = "".join(
        str(ord(c) - 55) if c.isalpha() else c for c in rearranged
    )
    if int(numeric) % 97 != 1:
        raise SepaCryptoError("IBAN: checksum mod-97 incorrecto")

    return norm


def mask_iban(iban: str) -> str:
    """Enmascara el cuerpo de la cuenta: 'ES76 **** **** **** 5766'.

    Mantiene visibles país + dígitos de control (4 primeros) y los 4 últimos.
    Esos 8 caracteres por sí solos no identifican la cuenta.
    """
    norm = normalize_iban(iban)
    head = norm[:4]
    last4 = norm[-4:]
    hidden = "*" * (len(norm) - 8)
    full = head + hidden + last4
    return " ".join(full[i:i + 4] for i in range(0, len(full), 4))


def iban_last4(iban: str) -> str:
    return normalize_iban(iban)[-4:]


# --- Cifrado / descifrado ---------------------------------------------------

def encrypt_iban(iban_plain: str) -> str:
    """Valida el IBAN y devuelve el token cifrado (base64-url)."""
    iban_norm = validate_iban(iban_plain)
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ct = _get_cipher().encrypt(nonce, iban_norm.encode("utf-8"), None)
    return base64.urlsafe_b64encode(nonce + ct).decode("ascii")


def decrypt_iban(token: str) -> str:
    """Descifra un token producido por encrypt_iban."""
    try:
        raw = base64.urlsafe_b64decode(token)
    except Exception as e:
        raise SepaCryptoError(f"Token IBAN con base64 inválido: {e}")
    if len(raw) <= _NONCE_BYTES:
        raise SepaCryptoError("Token IBAN demasiado corto")
    nonce, ct = raw[:_NONCE_BYTES], raw[_NONCE_BYTES:]
    try:
        plain = _get_cipher().decrypt(nonce, ct, None)
    except Exception as e:
        raise SepaCryptoError(f"Fallo al descifrar IBAN (¿clave cambiada?): {e}")
    return plain.decode("utf-8")
