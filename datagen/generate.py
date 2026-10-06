"""Synthetic purchase orders with exact ground truth (PRD §11.1).

generate_po(seed, knobs) builds one PurchaseOrder from a single seed; generate_many(n, seed,
knobs) builds a reproducible dataset. Every amount is computed with Decimal and ROUND_HALF_UP,
and datagen/consistency.check_po verifies the result independently.

Rules the generator follows (PRD §6, §9.4):
- buyer/vendor GSTIN state codes match their addresses; PAN = GSTIN characters 3-12
- place_of_supply = ship-to state; vendor state == place of supply -> CGST + SGST (half the
  rate each, per line), else IGST only
- taxable = qty x rate x (1 - disc%), line taxes and totals rounded to paise
- grand_total = subtotal + taxes + freight + other - discount, rounded to whole rupees, with
  the difference in round_off (|round_off| <= 0.50)

The ground truth is a PurchaseOrder. Things that only affect rendering (T&C page) are in
GeneratedPO.meta.
"""

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from faker import Faker

from datagen import india
from datagen.consistency import RUPEE, ZERO, round_money
from datagen.india import Address, CatalogueItem, Company, State
from schema.po_schema import LineItem, PurchaseOrder

# =========================================================================================
# Knobs
# =========================================================================================


@dataclass(frozen=True)
class Knobs:
    """Difficulty knobs. Probabilities are per PO unless the name says per line."""

    min_lines: int = 30
    max_lines: int = 60
    p_inter_state: float = 0.40
    p_multiline_description_per_line: float = 0.25  # append a spec/instruction clause
    p_discount_per_line: float = 0.20
    p_header_discount: float = 0.15
    p_freight: float = 0.40
    p_other_charges: float = 0.30
    p_amendment: float = 0.15
    p_quotation_ref: float = 0.60
    p_indent_no: float = 0.50
    p_line_delivery_dates: float = 0.30  # rows carry their own delivery date
    optional_drop_rate: float = 0.15  # share of droppable optional header fields left absent
    p_bill_to_differs: float = 0.25  # bill-to is another unit of the buyer (same PAN)
    p_ship_to_differs: float = 0.30  # goods go to another plant than the bill-to
    p_tc_page: float = 0.50
    po_date_from: date = date(2025, 4, 1)
    po_date_to: date = date(2026, 9, 30)

    def __post_init__(self) -> None:
        if not 1 <= self.min_lines <= self.max_lines:
            raise ValueError("need 1 <= min_lines <= max_lines")
        if self.po_date_from > self.po_date_to:
            raise ValueError("po_date_from must not be after po_date_to")
        for name, value in vars(self).items():
            if (name.startswith("p_") or name.endswith("_rate")) and not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability in [0, 1], got {value}")


@dataclass(frozen=True)
class PoMeta:
    """Generation facts that are not PO fields but matter for rendering and analysis."""

    seed: int
    vendor_industry: str
    inter_state: bool
    has_tc_page: bool


@dataclass(frozen=True)
class GeneratedPO:
    """Ground truth PO plus its generation metadata."""

    po: PurchaseOrder
    meta: PoMeta


# Optional header fields that the drop knob may remove. Everything else is either critical,
# controlled by its own knob (amendment, quotation, indent, charges, discount), or needed to
# keep the document coherent (addresses, names, place of supply, amount in words).
DROPPABLE_FIELDS = (
    "currency", "delivery_date", "po_validity_date",
    "buyer_pan", "buyer_state", "buyer_state_code", "buyer_contact_person", "buyer_phone",
    "buyer_email",
    "vendor_code", "vendor_pan", "vendor_state", "vendor_state_code", "vendor_contact_person",
    "vendor_phone", "vendor_email",
    "bill_to_gstin", "bill_to_state_code", "ship_to_gstin", "ship_to_state_code",
    "payment_terms", "delivery_terms", "freight_terms", "mode_of_transport",
    "delivery_location", "warranty_terms", "packing_instructions",
    "total_tax", "prepared_by", "approved_by",
)  # fmt: skip

# Vendors also sell from related categories, so a 60-line PO need not repeat items.
RELATED_INDUSTRIES = {
    "industrial_equipment": ("fasteners_hardware", "electrical"),
    "chemicals": ("industrial_equipment",),
    "electrical": ("industrial_equipment", "fasteners_hardware"),
    "fasteners_hardware": ("industrial_equipment", "electrical"),
    "it_office": ("electrical", "services"),
    "services": ("it_office", "industrial_equipment"),
}
BUYER_INDUSTRIES = ("industrial_equipment", "chemicals", "electrical")  # manufacturers

DESCRIPTION_ADDENDA = (
    "Make: as per approved vendor list or equivalent",
    "Test certificate to be submitted along with supply",
    "Drawing No. {drg} Rev. {rev}",
    "Delivery in two equal lots as per schedule",
    "Each lot with batch-wise certificate of analysis",
    "Packed and labelled with PO number and item code",
    "Inspection at vendor's works before dispatch",
)
LINE_DISCOUNTS = ("2", "2.5", "5", "7.5", "10", "12", "15")
WHOLE_UNIT_UOMS = {"NOS", "SET", "BOX", "HRS"}

_faker: Faker | None = None


def _seeded_faker(rng: random.Random) -> Faker:
    """One shared Faker instance (slow to create), reseeded from `rng` for every PO."""
    global _faker
    if _faker is None:
        _faker = Faker("en_IN")
    _faker.seed_instance(rng.getrandbits(64))
    return _faker


# =========================================================================================
# Public API
# =========================================================================================


def generate_po(seed: int, knobs: Knobs | None = None) -> GeneratedPO:
    """Generate one consistent PO; the same seed and knobs always give the same PO."""
    knobs = knobs or Knobs()
    rng = random.Random(seed)
    return _Builder(rng, _seeded_faker(rng), knobs, seed).build()


def generate_many(n: int, seed: int, knobs: Knobs | None = None) -> list[GeneratedPO]:
    """Generate `n` POs. PO i uses its own seed (in meta.seed), so it can be rebuilt alone."""
    seeds = random.Random(seed)
    return [generate_po(seeds.getrandbits(63), knobs) for _ in range(n)]


# =========================================================================================
# Builder
# =========================================================================================


@dataclass
class _Party:
    """A business unit: company, location and GST registration."""

    company: Company
    state: State
    address: Address
    pan: str
    gstin: str


@dataclass
class _Builder:
    rng: random.Random
    fake: Faker
    knobs: Knobs
    seed: int
    fields: dict = field(default_factory=dict)

    def build(self) -> GeneratedPO:
        rng, knobs = self.rng, self.knobs

        buyer = self._new_party(rng.choice(BUYER_INDUSTRIES), india.pick_industrial_state(rng))
        bill_to = buyer
        if rng.random() < knobs.p_bill_to_differs:
            bill_to = self._other_unit(buyer)
        ship_to = bill_to
        if rng.random() < knobs.p_ship_to_differs:
            ship_to = self._other_unit(buyer)

        inter_state = rng.random() < knobs.p_inter_state
        vendor_state = ship_to.state if not inter_state else self._other_state(ship_to.state)
        vendor_industry = rng.choice(india.INDUSTRIES)
        vendor = self._new_party(vendor_industry, vendor_state)

        po_date = self._po_date()
        self._document_fields(buyer, po_date)
        self._party_fields(buyer, vendor, bill_to, ship_to)
        self._terms_fields(ship_to)
        lines = self._line_items(vendor_industry, inter_state, po_date)
        self._totals_fields(lines)
        self._drop_optional_fields()

        po = PurchaseOrder.model_validate({**self.fields, "line_items": lines})
        has_tc_page = rng.random() < knobs.p_tc_page
        return GeneratedPO(po, PoMeta(self.seed, vendor_industry, inter_state, has_tc_page))

    # --- parties ---------------------------------------------------------------------------

    def _new_party(self, industry: str, state: State) -> _Party:
        company = india.generate_company(self.rng, industry)
        pan = india.generate_pan(self.rng, company.pan_holder_type, company.name)
        return self._unit(company, state, pan)

    def _unit(self, company: Company, state: State, pan: str) -> _Party:
        address = india.generate_address(self.rng, state)
        return _Party(company, state, address, pan, india.generate_gstin(self.rng, state.code, pan))

    def _other_unit(self, party: _Party) -> _Party:
        """Another plant or office of the same company: same PAN, possibly another state."""
        state = party.state if self.rng.random() < 0.5 else self._other_state(party.state)
        return self._unit(party.company, state, party.pan)

    def _other_state(self, state: State) -> State:
        while (other := india.pick_industrial_state(self.rng)) == state:
            pass
        return other

    # --- header sections -------------------------------------------------------------------

    def _po_date(self) -> date:
        span = (self.knobs.po_date_to - self.knobs.po_date_from).days
        return self.knobs.po_date_from + timedelta(days=self.rng.randint(0, span))

    def _document_fields(self, buyer: _Party, po_date: date) -> None:
        rng, knobs = self.rng, self.knobs
        f = self.fields
        f["po_number"] = _po_number(rng, buyer.company, po_date)
        f["po_date"] = po_date
        if rng.random() < knobs.p_amendment:
            f["amendment_no"] = rng.choice(["1", "2", "Rev. 01", "A1", "Amendment 1"])
            f["amendment_date"] = po_date + timedelta(days=rng.randint(1, 30))
        if rng.random() < knobs.p_quotation_ref:
            f["quotation_ref"] = rng.choice(
                [
                    f"QTN/{po_date:%y}/{rng.randint(100, 9999)}",
                    f"Q-{rng.randint(10000, 99999)}",
                    f"Your offer no. {rng.randint(100, 999)}/{po_date:%Y}",
                ]
            )
            f["quotation_date"] = po_date - timedelta(days=rng.randint(3, 45))
        if rng.random() < knobs.p_indent_no:
            f["indent_no"] = rng.choice(
                [f"IND-{rng.randint(1000, 99999)}", str(rng.randint(10_000_000, 10_999_999))]
            )
        f["currency"] = "INR"
        f["delivery_date"] = po_date + timedelta(days=rng.randint(7, 90))
        f["po_validity_date"] = po_date + timedelta(days=rng.randint(90, 365))

    def _party_fields(
        self, buyer: _Party, vendor: _Party, bill_to: _Party, ship_to: _Party
    ) -> None:
        f, fake = self.fields, self.fake
        for prefix, party in (("buyer", buyer), ("vendor", vendor)):
            contact = india.generate_contact(fake, party.company.domain)
            f[f"{prefix}_name"] = party.company.name
            f[f"{prefix}_address"] = party.address.one_line()
            f[f"{prefix}_gstin"] = party.gstin
            f[f"{prefix}_pan"] = party.pan
            f[f"{prefix}_state"] = party.state.name
            f[f"{prefix}_state_code"] = party.state.code
            f[f"{prefix}_contact_person"] = contact.name
            f[f"{prefix}_phone"] = contact.phone
            f[f"{prefix}_email"] = contact.email
        f["vendor_code"] = self.rng.choice(
            [f"V{self.rng.randint(10000, 99999)}", f"SUP-{self.rng.randint(100, 9999):04d}"]
        )
        for prefix, party in (("bill_to", bill_to), ("ship_to", ship_to)):
            f[f"{prefix}_name"] = party.company.name
            f[f"{prefix}_address"] = party.address.one_line()
            f[f"{prefix}_gstin"] = party.gstin
            f[f"{prefix}_state_code"] = party.state.code

    def _terms_fields(self, ship_to: _Party) -> None:
        rng, f = self.rng, self.fields
        state = ship_to.state
        f["place_of_supply"] = rng.choice(
            [f"{state.name} ({state.code})", f"{state.code}-{state.name}", state.name]
        )
        f["payment_terms"] = rng.choice(india.PAYMENT_TERMS)
        f["delivery_terms"] = rng.choice(india.DELIVERY_TERMS)
        f["freight_terms"] = rng.choice(india.FREIGHT_TERMS)
        f["mode_of_transport"] = rng.choice(india.TRANSPORT_MODES)
        f["delivery_location"] = rng.choice(
            [ship_to.address.city, f"Our {ship_to.address.city} plant", "Stores, main plant"]
        )
        f["warranty_terms"] = rng.choice(india.WARRANTY_TERMS)
        f["packing_instructions"] = rng.choice(india.PACKING_INSTRUCTIONS)
        f["prepared_by"] = self.fake.name()
        f["approved_by"] = self.fake.name()

    # --- line items ------------------------------------------------------------------------

    def _line_items(self, industry: str, inter_state: bool, po_date: date) -> list[LineItem]:
        rng, knobs = self.rng, self.knobs
        count = rng.randint(knobs.min_lines, knobs.max_lines)
        pool = india.items_for(industry)
        for related in RELATED_INDUSTRIES[industry]:
            pool += india.items_for(related)
        chosen = rng.sample(pool, min(count, len(pool)))
        chosen += rng.choices(pool, k=count - len(chosen))  # repeats only if the pool is small

        code_style = india.pick_item_code_style(rng)
        with_codes = rng.random() >= knobs.optional_drop_rate
        with_dates = rng.random() < knobs.p_line_delivery_dates
        used_codes: set[str] = set()

        lines = []
        for line_no, item in enumerate(chosen, start=1):
            code = None
            if with_codes:
                while (code := india.generate_item_code(rng, item, code_style)) in used_codes:
                    pass
                used_codes.add(code)
            delivery = po_date + timedelta(days=rng.randint(7, 120)) if with_dates else None
            lines.append(self._line(line_no, item, code, delivery, inter_state))
        return lines

    def _line(
        self,
        line_no: int,
        item: CatalogueItem,
        code: str | None,
        delivery: date | None,
        inter_state: bool,
    ) -> LineItem:
        rng = self.rng
        rate = india.generate_unit_price(rng, item)
        quantity = _quantity(rng, item, rate)
        discount = None
        if rng.random() < self.knobs.p_discount_per_line:
            discount = Decimal(rng.choice(LINE_DISCOUNTS))
        description = item.description
        if rng.random() < self.knobs.p_multiline_description_per_line:
            addendum = rng.choice(DESCRIPTION_ADDENDA).format(
                drg=f"DRG-{rng.randint(1000, 9999)}", rev=rng.randint(0, 5)
            )
            description = f"{description}. {addendum}"

        taxable = round_money(quantity * rate * (1 - (discount or ZERO) / 100))
        cgst = sgst = igst = None
        if inter_state:
            igst = round_money(taxable * item.gst_rate / 100)
            taxes = igst
        else:
            cgst = sgst = round_money(taxable * item.gst_rate / 2 / 100)
            taxes = cgst + sgst
        return LineItem(
            line_no=line_no,
            item_code=code,
            description=description,
            hsn_sac=item.hsn_sac,
            quantity=quantity,
            uom=item.uom,
            unit_rate=rate,
            discount_pct=discount,
            taxable_value=taxable,
            gst_rate=item.gst_rate,
            cgst_amount=cgst,
            sgst_amount=sgst,
            igst_amount=igst,
            line_total=taxable + taxes,
            line_delivery_date=delivery,
        )

    # --- totals ----------------------------------------------------------------------------

    def _totals_fields(self, lines: list[LineItem]) -> None:
        rng, knobs, f = self.rng, self.knobs, self.fields
        subtotal = sum((line.taxable_value for line in lines), ZERO)
        cgst = sum((line.cgst_amount or ZERO for line in lines), ZERO)
        sgst = sum((line.sgst_amount or ZERO for line in lines), ZERO)
        igst = sum((line.igst_amount or ZERO for line in lines), ZERO)
        total_tax = cgst + sgst + igst

        freight = other = discount = None
        if rng.random() < knobs.p_freight:
            freight = Decimal(rng.randint(5, 120) * 100).quantize(Decimal("0.01"))
        if rng.random() < knobs.p_other_charges:  # e.g. P&F at 1-2% of subtotal
            other = round_money(subtotal * Decimal(rng.choice(["1", "1.5", "2"])) / 100)
        if rng.random() < knobs.p_header_discount:
            # A round "special discount" of at most 5% of the subtotal.
            options = [v for v in (500, 1000, 1500, 2000, 2500, 5000) if v <= subtotal / 20]
            discount = Decimal(rng.choice(options or [500])).quantize(Decimal("0.01"))

        charges = (freight or ZERO) + (other or ZERO) - (discount or ZERO)
        before_rounding = subtotal + total_tax + charges
        grand_total = before_rounding.quantize(RUPEE, rounding=ROUND_HALF_UP)
        grand_total = grand_total.quantize(Decimal("0.01"))
        round_off = grand_total - before_rounding

        f.update(
            subtotal=subtotal,
            discount_total=discount,
            freight_charges=freight,
            other_charges=other,
            cgst_total=cgst,
            sgst_total=sgst,
            igst_total=igst,
            total_tax=total_tax,
            round_off=round_off,
            grand_total=grand_total,
            amount_in_words=india.amount_in_words_inr(grand_total),
        )

    def _drop_optional_fields(self) -> None:
        for name in DROPPABLE_FIELDS:
            if self.rng.random() < self.knobs.optional_drop_rate:
                self.fields.pop(name, None)


# =========================================================================================
# Helpers
# =========================================================================================


def _quantity(rng: random.Random, item: CatalogueItem, rate: Decimal) -> Decimal:
    """A quantity that gives a plausible line value (Rs 2,000 - 2.5 lakh) for this rate."""
    target_value = Decimal(rng.randint(2_000, 250_000))
    raw = target_value / rate
    if item.uom in WHOLE_UNIT_UOMS:
        return max(Decimal(1), raw.quantize(Decimal(1), rounding=ROUND_HALF_UP))
    step = Decimal(1) if rng.random() < 0.6 else Decimal("0.5")
    return max(step, (raw / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step)


def _financial_year(day: date) -> str:
    """Indian financial year (April-March) as '2026-27'."""
    start = day.year if day.month >= 4 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _initials(company: Company) -> str:
    """Up to three capital letters from the meaningful words of a company name."""
    skip = {"pvt.", "ltd.", "private", "limited", "llp", "&", "co."}
    words = [w for w in company.name.split() if w.lower() not in skip]
    return "".join(w[0] for w in words).upper()[:3].ljust(3, "X")


def _po_number(rng: random.Random, buyer: Company, po_date: date) -> str:
    """PO number in one of the formats real buyers use (the style varies by buyer)."""
    fy = _financial_year(po_date)
    n = rng.randint(1, 99_999)
    style = rng.choice(["fy_slash", "sap", "dept", "unit_fy", "yearmonth"])
    if style == "fy_slash":
        return f"PO/{fy}/{n:05d}"  # PO/2026-27/00457
    if style == "sap":
        return f"45000{n:05d}"  # 4500012345
    if style == "dept":
        return f"{_initials(buyer)}-PUR-{po_date:%y}-{n % 10_000:04d}"  # GMM-PUR-26-1182
    if style == "unit_fy":
        return f"{_initials(buyer)}/PO/{n % 10_000}/{fy[2:]}"  # SGE/PO/1182/26-27
    return f"PO-{po_date:%Y%m}-{n % 10_000:04d}"  # PO-202603-0457
