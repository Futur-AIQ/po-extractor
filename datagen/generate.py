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

Who issues the PO (PRD §5): ~70% come from 30 frequent issuers (datagen/issuers.py), each
with its own layout, PO-number format and item codes; the rest from one-off issuers. Vendors
come from a fixed pool of 40 counterparties, so the mock ERP masters can know every party.

The ground truth is a PurchaseOrder. Things that only affect rendering (layout, T&C page)
and dataset facts (issuer, expected duplicate) are in GeneratedPO.meta.
"""

import random
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from faker import Faker

from datagen import india
from datagen.consistency import RUPEE, ZERO, round_money
from datagen.india import CatalogueItem
from datagen.issuers import (
    Counterparty,
    Issuer,
    Unit,
    counterparties,
    frequent_issuers,
    make_one_off_issuer,
    serial_range,
    vendor_code,
)
from schema.po_schema import LineItem, PurchaseOrder

# =========================================================================================
# Knobs
# =========================================================================================


@dataclass(frozen=True)
class Knobs:
    """Difficulty knobs. Probabilities are per PO unless the name says per line."""

    min_lines: int = 20
    max_lines: int = 80
    typical_min_lines: int = 30  # most POs have 30-50 lines (PRD §11.1)
    typical_max_lines: int = 50
    p_typical_line_count: float = 0.70  # else drawn from the full min..max range
    frequent_issuer_share: float = 0.70  # POs from the 30 frequent issuers (PRD §5)
    duplicate_po_count: int = 2  # POs whose number is pre-registered as already processed
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
    p_rows_split_across_pages: float = 0.30  # table rows may break across a page boundary
    po_date_from: date = date(2025, 4, 1)
    po_date_to: date = date(2026, 9, 30)

    def __post_init__(self) -> None:
        if not 1 <= self.min_lines <= self.max_lines:
            raise ValueError("need 1 <= min_lines <= max_lines")
        if self.typical_min_lines > self.typical_max_lines:
            raise ValueError("typical_min_lines must not exceed typical_max_lines")
        if not 0 <= self.frequent_issuer_share <= 1:
            raise ValueError("frequent_issuer_share must be in [0, 1]")
        if self.duplicate_po_count < 0:
            raise ValueError("duplicate_po_count must not be negative")
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
    rows_may_break: bool = False
    issuer_id: str = ""
    issuer_frequent: bool = False
    layout: str = ""  # the issuer's layout
    issuer_date_style: str = ""  # metadata only: templates fix the date style per layout
    expected_duplicate: bool = False  # PO number pre-registered as already processed


@dataclass(frozen=True)
class GeneratedPO:
    """Ground truth PO plus its generation metadata."""

    po: PurchaseOrder
    meta: PoMeta
    issuer: Issuer  # full profile, needed to build the ERP masters (not written to truth)


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

# Vendors also sell from related categories, so a long PO need not repeat items.
RELATED_INDUSTRIES = {
    "industrial_equipment": ("fasteners_hardware", "electrical"),
    "chemicals": ("industrial_equipment",),
    "electrical": ("industrial_equipment", "fasteners_hardware"),
    "fasteners_hardware": ("industrial_equipment", "electrical"),
    "it_office": ("electrical", "services"),
    "services": ("it_office", "industrial_equipment"),
}

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
    """Generate `n` POs. PO i uses its own seed (in meta.seed), so it can be rebuilt alone.

    PO numbers are unique per issuer (a seed that repeats one is skipped), and
    knobs.duplicate_po_count POs are marked as expected duplicates: their numbers go into the
    masters' processed list, so validation must flag them.
    """
    knobs = knobs or Knobs()
    seeds = random.Random(seed)
    pos: list[GeneratedPO] = []
    seen: set[tuple[str, str]] = set()
    while len(pos) < n:
        generated = generate_po(seeds.getrandbits(63), knobs)
        key = (generated.meta.issuer_id, generated.po.po_number)
        if key not in seen:
            seen.add(key)
            pos.append(generated)
    return _mark_expected_duplicates(pos, knobs.duplicate_po_count, seed)


def _mark_expected_duplicates(pos: list[GeneratedPO], count: int, seed: int) -> list[GeneratedPO]:
    """Mark `count` POs (preferably from frequent issuers) as expected duplicates."""
    frequent = [i for i, g in enumerate(pos) if g.meta.issuer_frequent]
    candidates = frequent if len(frequent) >= count else list(range(len(pos)))
    chosen = set(random.Random(f"duplicates:{seed}").sample(candidates, min(count, len(pos))))
    return [
        replace(g, meta=replace(g.meta, expected_duplicate=True)) if i in chosen else g
        for i, g in enumerate(pos)
    ]


# =========================================================================================
# Builder
# =========================================================================================


@dataclass
class _Builder:
    rng: random.Random
    fake: Faker
    knobs: Knobs
    seed: int
    fields: dict = field(default_factory=dict)

    def build(self) -> GeneratedPO:
        rng, knobs = self.rng, self.knobs

        issuer = self._issuer()
        bill_to = issuer.main
        if rng.random() < knobs.p_bill_to_differs:
            bill_to = rng.choice(issuer.units[1:])  # e.g. head office pays for a plant
        ship_to = bill_to
        if rng.random() < knobs.p_ship_to_differs:
            ship_to = rng.choice([unit for unit in issuer.units if unit != bill_to])

        inter_state = rng.random() < knobs.p_inter_state
        vendor = self._vendor(ship_to, inter_state)

        po_date = self._po_date()
        self._document_fields(issuer, po_date)
        self._party_fields(issuer, vendor, bill_to, ship_to)
        self._terms_fields(ship_to)
        lines = self._line_items(issuer, vendor.company.industry, inter_state, po_date)
        self._totals_fields(lines)
        self._drop_optional_fields()

        po = PurchaseOrder.model_validate({**self.fields, "line_items": lines})
        meta = PoMeta(
            seed=self.seed,
            vendor_industry=vendor.company.industry,
            inter_state=inter_state,
            has_tc_page=rng.random() < knobs.p_tc_page,
            rows_may_break=rng.random() < knobs.p_rows_split_across_pages,
            issuer_id=issuer.id,
            issuer_frequent=issuer.frequent,
            layout=issuer.layout,
            issuer_date_style=issuer.date_style,
        )
        return GeneratedPO(po, meta, issuer)

    # --- parties ---------------------------------------------------------------------------

    def _issuer(self) -> Issuer:
        """A frequent issuer (knobs.frequent_issuer_share of POs) or a one-off issuer."""
        if self.rng.random() < self.knobs.frequent_issuer_share:
            return self.rng.choice(frequent_issuers())
        return make_one_off_issuer(self.rng)

    def _vendor(self, ship_to: Unit, inter_state: bool) -> Counterparty:
        """A pool vendor in the ship-to state (intra-state) or in another state (inter-state)."""
        candidates = [
            vendor
            for vendor in counterparties()
            if (vendor.unit.state == ship_to.state) != inter_state
        ]
        return self.rng.choice(candidates)

    # --- header sections -------------------------------------------------------------------

    def _po_date(self) -> date:
        span = (self.knobs.po_date_to - self.knobs.po_date_from).days
        return self.knobs.po_date_from + timedelta(days=self.rng.randint(0, span))

    def _document_fields(self, issuer: Issuer, po_date: date) -> None:
        rng, knobs = self.rng, self.knobs
        f = self.fields
        f["po_number"] = issuer.po_number(
            po_date, rng.randint(*serial_range(issuer.po_number_style, historical=False))
        )
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
        self, issuer: Issuer, vendor: Counterparty, bill_to: Unit, ship_to: Unit
    ) -> None:
        f, fake = self.fields, self.fake
        parties = (
            ("buyer", issuer.company, issuer.pan, issuer.main),
            ("vendor", vendor.company, vendor.pan, vendor.unit),
        )
        for prefix, company, pan, unit in parties:
            contact = india.generate_contact(fake, company.domain)
            f[f"{prefix}_name"] = company.name
            f[f"{prefix}_address"] = unit.address.one_line()
            f[f"{prefix}_gstin"] = unit.gstin
            f[f"{prefix}_pan"] = pan
            f[f"{prefix}_state"] = unit.state.name
            f[f"{prefix}_state_code"] = unit.state.code
            f[f"{prefix}_contact_person"] = contact.name
            f[f"{prefix}_phone"] = contact.phone
            f[f"{prefix}_email"] = contact.email
        f["vendor_code"] = vendor_code(issuer, vendor)
        for prefix, unit in (("bill_to", bill_to), ("ship_to", ship_to)):
            f[f"{prefix}_name"] = issuer.company.name
            f[f"{prefix}_address"] = unit.address.one_line()
            f[f"{prefix}_gstin"] = unit.gstin
            f[f"{prefix}_state_code"] = unit.state.code

    def _terms_fields(self, ship_to: Unit) -> None:
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

    def _line_count(self) -> int:
        """20-80 lines, most POs 30-50 (knobs; the typical range is clipped to min..max)."""
        rng, knobs = self.rng, self.knobs
        low = max(knobs.min_lines, knobs.typical_min_lines)
        high = min(knobs.max_lines, knobs.typical_max_lines)
        if low <= high and rng.random() < knobs.p_typical_line_count:
            return rng.randint(low, high)
        return rng.randint(knobs.min_lines, knobs.max_lines)

    def _line_items(
        self, issuer: Issuer, industry: str, inter_state: bool, po_date: date
    ) -> list[LineItem]:
        rng, knobs = self.rng, self.knobs
        count = self._line_count()
        pool = india.items_for(industry)
        for related in RELATED_INDUSTRIES[industry]:
            pool += india.items_for(related)
        if count > len(pool):  # a long PO: the vendor acts as a broad-line distributor
            pool += [item for item in india.CATALOGUE if item not in pool]
        chosen = rng.sample(pool, count)  # distinct items, so item codes are unique per PO

        with_codes = rng.random() >= knobs.optional_drop_rate
        with_dates = rng.random() < knobs.p_line_delivery_dates
        lines = []
        for line_no, item in enumerate(chosen, start=1):
            code = issuer.item_code(item) if with_codes else None
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
