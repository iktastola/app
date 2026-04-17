"""Configuración SEPA del club.

Todas las constantes sensibles (clave de cifrado, ICS, IBAN del club) se leen
de variables de entorno. Los valores TODO... son placeholders hasta que
Kutxabank asigne el ICS.
"""
import os

# --- Acreedor (el club) -----------------------------------------------------

CREDITOR_ID = os.environ.get("SEPA_CREDITOR_ID", "TODOACREEDORIDENTIFIER")
CREDITOR_NAME = os.environ.get("SEPA_CREDITOR_NAME", "Club de Natación Astola")
CREDITOR_IBAN = os.environ.get("SEPA_CREDITOR_IBAN", "TODOIBANDELCLUB")
CREDITOR_BIC = os.environ.get("SEPA_CREDITOR_BIC", "BASKES2BXXX")  # Kutxabank
CREDITOR_COUNTRY = "ES"

# --- Cuotas -----------------------------------------------------------------

MONTHLY_FEE_EUR = 45.00
BILLING_MONTHS = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]  # sep → jun
CURRENCY = "EUR"

# --- Norma SEPA -------------------------------------------------------------

PAIN_SCHEMA = "pain.008.001.02"
LOCAL_INSTRUMENT = "CORE"  # adeudos de particulares (no B2B)

# Plazos mínimos de antelación (días hábiles) entre envío y fecha de cargo.
# Kutxabank aplica norma SEPA estándar.
LEAD_TIME_DAYS_FRST = 5
LEAD_TIME_DAYS_RCUR = 2

# --- Seguridad --------------------------------------------------------------

# Clave AES-256 (base64-url, 32 bytes decodificados). Obligatoria en prod.
ENCRYPTION_KEY_B64 = os.environ.get("SEPA_ENCRYPTION_KEY")
