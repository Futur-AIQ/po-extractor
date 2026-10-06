"""PO issuers (the buyer whose ERP printed the PO) and counterparties (the vendors).

Real distribution (PRD §5): ~30 frequent issuers produce ~70% of POs. Each frequent issuer
is bound to one layout and has its own PO-number format, date-format preference and item
codes, so its POs look alike from one month to the next. The pools are built from their own
fixed seeds, so they are identical across runs and independent of any dataset seed.

    frequent_issuers()        30 fixed issuers, 5 per layout
    counterparties()          40 fixed vendors, 4 in each industrial state
    make_one_off_issuer(rng)  a random issuer with a random layout (the other ~30%)
"""

import random
from dataclasses import dataclass
from datetime import date
from functools import cache

from datagen import india
from datagen.india import CATALOGUE, Address, CatalogueItem, Company, State

ISSUER_POOL_SEED = "issuer-pool-v1"
COUNTERPARTY_POOL_SEED = "counterparty-pool-v1"
FREQUENT_ISSUER_COUNT = 30
COUNTERPARTY_COUNT = 40
UNITS_PER_ISSUER = 3  # main unit (the buyer) + two other plants/offices, same PAN

# Must match datagen.templates.LAYOUTS (checked by a test; importing it here would be circular).
LAYOUT_IDS = (
    "L1_classic_erp",
    "L2_tally_style",
    "L3_modern_minimal",
    "L4_engineering",
    "L5_label_variants",
    "L6_psu_formal",
)
# Date styles an issuer can prefer (all parseable by schema.dates.parse_po_date).
DATE_STYLES = ("dd-mm-yyyy", "dd/mm/yyyy", "dd.mm.yyyy", "dd-Mon-yyyy", "dd Mon yyyy")
BUYER_INDUSTRIES = ("industrial_equipment", "chemicals", "electrical")  # manufacturers

# PO-number style -> digits in its serial number.
PO_NUMBER_STYLES: dict[str, int] = {
    "fy_slash": 5,  # PO/2026-27/00457
    "sap": 5,  # 4500012345
    "dept": 4,  # GMM-PUR-26-1182
    "unit_fy": 4,  # GMM/PO/1182/26-27
    "yearmonth": 4,  # PO-202603-0457
}


@dataclass(frozen=True)
class Unit:
    """One GST registration of a company: a plant or office in one state."""

    state: State
    address: Address
    gstin: str


@dataclass(frozen=True)
class Issuer:
    """The company that issues POs, with the formats its ERP uses."""

    id: str
    company: Company
    pan: str
    units: tuple[Unit, ...]  # units[0] is the issuing (buyer) unit
    layout: str
    po_number_style: str
    date_style: str
    item_code_style: str
    item_code_base: int  # makes this issuer's item codes differ from other issuers'
    frequent: bool

    @property
    def main(self) -> Unit:
        return self.units[0]

    @property
    def initials(self) -> str:
        """Up to three capital letters from the meaningful words of the name, e.g. 'SGE'."""
        skip = {"pvt.", "ltd.", "private", "limited", "llp", "&", "co."}
        words = [w for w in self.company.name.split() if w.lower() not in skip]
        return "".join(w[0] for w in words).upper()[:3].ljust(3, "X")

    def po_number(self, po_date: date, serial: int) -> str:
        """This issuer's PO number for a date and serial number."""
        return format_po_number(self.po_number_style, self.initials, po_date, serial)

    def item_code(self, item: CatalogueItem) -> str:
        """This issuer's (fixed) code for a catalogue item: its ERP item master."""
        index = _CATALOGUE_INDEX[item]
        number = self.item_code_base + index
        if self.item_code_style == "dash":
            return f"{item.code_prefix}-{number:04d}"
        if self.item_code_style == "slash":
            return f"{item.code_prefix}/{number:04d}/{'ABC'[self.item_code_base % 3]}"
        return str(10_000_000 + self.item_code_base * 1_000 + index)  # SAP material number


@dataclass(frozen=True)
class Counterparty:
    """A vendor that receives POs."""

    id: str
    company: Company
    pan: str
    unit: Unit


_CATALOGUE_INDEX = {item: index for index, item in enumerate(CATALOGUE)}


# =========================================================================================
# PO numbers
# =========================================================================================


def financial_year(day: date) -> str:
    """Indian financial year (April-March) as '2026-27'."""
    start = day.year if day.month >= 4 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def format_po_number(style: str, initials: str, po_date: date, serial: int) -> str:
    """Format a PO number in one of PO_NUMBER_STYLES."""
    fy = financial_year(po_date)
    if style == "fy_slash":
        return f"PO/{fy}/{serial:05d}"
    if style == "sap":
        return f"45000{serial:05d}"
    if style == "dept":
        return f"{initials}-PUR-{po_date:%y}-{serial:04d}"
    if style == "unit_fy":
        return f"{initials}/PO/{serial}/{fy[2:]}"
    if style == "yearmonth":
        return f"PO-{po_date:%Y%m}-{serial:04d}"
    raise ValueError(f"unknown PO number style {style!r}")


def serial_range(style: str, historical: bool) -> tuple[int, int]:
    """Serials for historical POs (masters) and new POs (dataset) never overlap.

    So a dataset PO number can only match an "already processed" number on purpose.
    """
    top = 10 ** PO_NUMBER_STYLES[style] - 1
    half = top // 2
    return (1, half) if historical else (half + 1, top)


# =========================================================================================
# Pools
# =========================================================================================


def _unit(rng: random.Random, state: State, pan: str) -> Unit:
    address = india.generate_address(rng, state)
    return Unit(state, address, india.generate_gstin(rng, state.code, pan))


def _other_state(rng: random.Random, state: State) -> State:
    while (other := india.pick_industrial_state(rng)) == state:
        pass
    return other


def _build_issuer(rng: random.Random, issuer_id: str, layout: str, frequent: bool) -> Issuer:
    company = india.generate_company(rng, rng.choice(BUYER_INDUSTRIES))
    pan = india.generate_pan(rng, company.pan_holder_type, company.name)
    main_state = india.pick_industrial_state(rng)
    units = [_unit(rng, main_state, pan)]
    for _ in range(UNITS_PER_ISSUER - 1):
        state = main_state if rng.random() < 0.5 else _other_state(rng, main_state)
        units.append(_unit(rng, state, pan))
    return Issuer(
        id=issuer_id,
        company=company,
        pan=pan,
        units=tuple(units),
        layout=layout,
        po_number_style=rng.choice(list(PO_NUMBER_STYLES)),
        date_style=rng.choice(DATE_STYLES),
        item_code_style=rng.choice(india.ITEM_CODE_STYLES),
        item_code_base=rng.randint(1_000, 8_000),
        frequent=frequent,
    )


@cache
def frequent_issuers() -> tuple[Issuer, ...]:
    """The 30 frequent issuers, 5 per layout. Identical on every run."""
    rng = random.Random(ISSUER_POOL_SEED)
    return tuple(
        _build_issuer(rng, f"ISS{n + 1:02d}", LAYOUT_IDS[n % len(LAYOUT_IDS)], frequent=True)
        for n in range(FREQUENT_ISSUER_COUNT)
    )


@cache
def counterparties() -> tuple[Counterparty, ...]:
    """The 40 vendors, spread evenly over the industrial states. Identical on every run."""
    rng = random.Random(COUNTERPARTY_POOL_SEED)
    states = [india.STATES[code] for code in india.INDUSTRIAL_STATE_WEIGHTS]
    pool = []
    for n in range(COUNTERPARTY_COUNT):
        company = india.generate_company(rng, rng.choice(india.INDUSTRIES))
        pan = india.generate_pan(rng, company.pan_holder_type, company.name)
        unit = _unit(rng, states[n % len(states)], pan)
        pool.append(Counterparty(f"CP{n + 1:02d}", company, pan, unit))
    return tuple(pool)


def make_one_off_issuer(rng: random.Random) -> Issuer:
    """A random issuer that appears once: random company, state, layout and formats."""
    issuer_id = f"ONE-{rng.getrandbits(32):08x}"
    return _build_issuer(rng, issuer_id, rng.choice(LAYOUT_IDS), frequent=False)


def vendor_code(issuer: Issuer, counterparty: Counterparty) -> str:
    """The issuer's (fixed) code for a vendor, as printed on its POs."""
    rng = random.Random(f"vendor-code:{issuer.id}:{counterparty.id}")
    return rng.choice([f"V{rng.randint(10000, 99999)}", f"SUP-{rng.randint(100, 9999):04d}"])
