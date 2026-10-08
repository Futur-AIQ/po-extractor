"""Money arithmetic shared by the generator and the extraction pipeline.

Everything is Decimal; amounts are rounded half up to paise, the convention on Indian tax
documents. The generator (datagen/) and compute/validate (app/extraction/) must round the
same way, or computed taxes would differ from the printed ones by a paisa.
"""

from decimal import ROUND_HALF_UP, Decimal

PAISA = Decimal("0.01")
RUPEE = Decimal("1")
ZERO = Decimal("0.00")


def round_money(value: Decimal) -> Decimal:
    """Round to 2 decimal places, half up (the convention on Indian tax documents)."""
    return value.quantize(PAISA, rounding=ROUND_HALF_UP)
