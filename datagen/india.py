"""Realistic Indian business data for synthetic purchase orders (PRD §11.1).

Everything random takes a `random.Random` instance, so a dataset is fully reproducible from
one seed. The global `random` module is never used. Faker (en_IN) is used only for person
names, phone numbers and email local parts, and is seeded from the same Random instance.

Contents:
    STATES / CITIES        GST state codes; real industrial cities, estates and PIN prefixes
    GSTIN / PAN            generators with the official GSTIN mod-36 checksum, validators
    companies / addresses  names consistent with industry, addresses consistent with state
    CATALOGUE              ~120 items across 6 industries with HSN/SAC, UoM, price, GST rate
    GST_SLABS              configurable GST rates and weights
    amount_in_words_inr    Indian numbering (lakh / crore) with paise
    *_TERMS pools          payment, delivery, freight, transport, warranty, packing text
"""

import random
import re
from dataclasses import dataclass
from decimal import Decimal

from faker import Faker
from num2words import num2words

# =========================================================================================
# States and cities
# =========================================================================================


@dataclass(frozen=True)
class State:
    """An Indian state or union territory with its official 2-digit GST state code."""

    code: str
    name: str


# Official GST state codes (GSTN). Legacy codes 25 (Daman & Diu, merged into 26 in 2020) and
# 28 (undivided Andhra Pradesh, now 37) are accepted by the validator but not generated.
STATES: dict[str, State] = {
    s.code: s
    for s in [
        State("01", "Jammu and Kashmir"),
        State("02", "Himachal Pradesh"),
        State("03", "Punjab"),
        State("04", "Chandigarh"),
        State("05", "Uttarakhand"),
        State("06", "Haryana"),
        State("07", "Delhi"),
        State("08", "Rajasthan"),
        State("09", "Uttar Pradesh"),
        State("10", "Bihar"),
        State("11", "Sikkim"),
        State("12", "Arunachal Pradesh"),
        State("13", "Nagaland"),
        State("14", "Manipur"),
        State("15", "Mizoram"),
        State("16", "Tripura"),
        State("17", "Meghalaya"),
        State("18", "Assam"),
        State("19", "West Bengal"),
        State("20", "Jharkhand"),
        State("21", "Odisha"),
        State("22", "Chhattisgarh"),
        State("23", "Madhya Pradesh"),
        State("24", "Gujarat"),
        State("26", "Dadra and Nagar Haveli and Daman and Diu"),
        State("27", "Maharashtra"),
        State("29", "Karnataka"),
        State("30", "Goa"),
        State("31", "Lakshadweep"),
        State("32", "Kerala"),
        State("33", "Tamil Nadu"),
        State("34", "Puducherry"),
        State("35", "Andaman and Nicobar Islands"),
        State("36", "Telangana"),
        State("37", "Andhra Pradesh"),
        State("38", "Ladakh"),
    ]
}
LEGACY_STATE_CODES = {"25", "28"}
SPECIAL_STATE_CODES = {"97", "99"}  # 97 Other Territory, 99 Centre Jurisdiction
VALID_GSTIN_STATE_CODES = set(STATES) | LEGACY_STATE_CODES | SPECIAL_STATE_CODES


@dataclass(frozen=True)
class City:
    """A real industrial city: the first 3 digits of its PIN codes and its industrial estates."""

    name: str
    pin_prefix: str
    estates: tuple[str, ...]


# ~10 major industrial states. PINs are generated as prefix + 3 random digits: plausible for
# the city, not guaranteed to be a real post office.
CITIES: dict[str, tuple[City, ...]] = {
    "24": (  # Gujarat: GIDC estates
        City("Ahmedabad", "382", ("GIDC Vatva", "GIDC Naroda", "GIDC Odhav")),
        City("Vadodara", "390", ("GIDC Makarpura", "Waghodia GIDC")),
        City("Surat", "394", ("GIDC Sachin", "GIDC Pandesara")),
        City("Rajkot", "360", ("GIDC Metoda", "Aji GIDC")),
        City("Ankleshwar", "393", ("GIDC Ankleshwar", "GIDC Panoli")),
        City("Vapi", "396", ("GIDC Vapi",)),
    ),
    "27": (  # Maharashtra: MIDC estates
        City("Pune", "411", ("MIDC Bhosari", "MIDC Pimpri")),
        City("Chakan", "410", ("MIDC Chakan Phase II",)),
        City("Navi Mumbai", "400", ("TTC Industrial Area, MIDC Mahape", "MIDC Rabale")),
        City("Nashik", "422", ("MIDC Ambad", "MIDC Satpur")),
        City("Chhatrapati Sambhajinagar", "431", ("MIDC Waluj", "MIDC Chikalthana")),
        City("Nagpur", "440", ("MIDC Hingna", "Butibori MIDC")),
    ),
    "29": (  # Karnataka: KIADB areas
        City("Bengaluru", "560", ("Peenya Industrial Area", "Bommasandra Industrial Area")),
        City("Mysuru", "570", ("Hebbal Industrial Area",)),
        City("Hubballi", "580", ("Gokul Road Industrial Estate",)),
        City("Belagavi", "590", ("Udyambag Industrial Estate",)),
        City("Mangaluru", "575", ("Baikampady Industrial Area",)),
    ),
    "33": (  # Tamil Nadu: SIPCOT / SIDCO estates
        City("Chennai", "600", ("Guindy Industrial Estate", "Ambattur Industrial Estate")),
        City("Coimbatore", "641", ("SIDCO Industrial Estate, Kurichi",)),
        City("Hosur", "635", ("SIPCOT Industrial Complex Phase I",)),
        City("Sriperumbudur", "602", ("SIPCOT Industrial Park",)),
        City("Madurai", "625", ("Kappalur Industrial Estate",)),
    ),
    "07": (  # Delhi: DSIIDC industrial areas, all PINs 110xxx
        City("New Delhi", "110", ("Okhla Industrial Area Phase II", "Naraina Industrial Area")),
        City("Delhi", "110", ("Mayapuri Industrial Area", "Wazirpur Industrial Area")),
        City("Narela", "110", ("Narela Industrial Area",)),
        City("Bawana", "110", ("DSIIDC Bawana Industrial Area",)),
    ),
    "06": (  # Haryana: HSIIDC / IMT
        City("Gurugram", "122", ("Udyog Vihar Phase IV",)),
        City("Manesar", "122", ("IMT Manesar",)),
        City("Faridabad", "121", ("Sector 24 Industrial Area",)),
        City("Bahadurgarh", "124", ("HSIIDC Industrial Estate",)),
        City("Panipat", "132", ("Sector 29 Industrial Area",)),
    ),
    "09": (  # Uttar Pradesh: UPSIDA
        City("Noida", "201", ("Sector 63", "Phase II")),
        City("Greater Noida", "201", ("Ecotech III", "Udyog Vihar")),
        City("Ghaziabad", "201", ("Sahibabad Industrial Area",)),
        City("Kanpur", "208", ("Panki Industrial Area",)),
        City("Lucknow", "226", ("Chinhat Industrial Area",)),
    ),
    "36": (  # Telangana: IDA / TSIIC
        City("Hyderabad", "500", ("IDA Balanagar", "IDA Cherlapally")),
        City("Patancheru", "502", ("IDA Patancheru",)),
        City("Medchal", "501", ("IDA Medchal",)),
        City("Warangal", "506", ("Kakatiya Industrial Estate",)),
    ),
    "19": (  # West Bengal
        City("Kolkata", "700", ("Taratala Industrial Estate",)),
        City("Howrah", "711", ("Jangalpur Industrial Estate",)),
        City("Durgapur", "713", ("Durgapur Industrial Estate",)),
        City("Haldia", "721", ("Haldia Industrial Area",)),
    ),
    "08": (  # Rajasthan: RIICO
        City("Jaipur", "302", ("Vishwakarma Industrial Area", "Sitapura Industrial Area")),
        City("Jodhpur", "342", ("Basni Industrial Area",)),
        City("Udaipur", "313", ("Madri Industrial Area",)),
        City("Bhiwadi", "301", ("RIICO Industrial Area",)),
        City("Kota", "324", ("Kota Industrial Area",)),
    ),
}

# Relative weight of each industrial state when picking a buyer or vendor location.
INDUSTRIAL_STATE_WEIGHTS: dict[str, int] = {
    "27": 20,
    "24": 18,
    "33": 14,
    "29": 12,
    "09": 9,
    "06": 8,
    "36": 7,
    "07": 5,
    "08": 4,
    "19": 3,
}


def pick_industrial_state(rng: random.Random) -> State:
    """Pick one of the states that has city data, weighted by industrial activity."""
    codes = list(INDUSTRIAL_STATE_WEIGHTS)
    weights = list(INDUSTRIAL_STATE_WEIGHTS.values())
    return STATES[rng.choices(codes, weights=weights)[0]]


# =========================================================================================
# PAN and GSTIN
# =========================================================================================

_ALNUM36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_FORMAT = re.compile(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]")
_PAN_FORMAT = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")

# PAN 4th character = holder type. C company, F firm/LLP, P individual (proprietor).
PAN_COMPANY, PAN_FIRM, PAN_PERSON = "C", "F", "P"


def generate_pan(rng: random.Random, holder_type: str = PAN_COMPANY, name: str = "") -> str:
    """Generate a PAN: AAA + holder type + initial of `name` + 4 digits + check letter.

    The 5th character is the first letter of the holder's name (surname for individuals),
    as on real PANs. The last character's algorithm is not public, so it is random.
    """
    if holder_type not in _LETTERS or len(holder_type) != 1:
        raise ValueError(f"PAN holder type must be one letter, got {holder_type!r}")
    initial = next((c for c in name.upper() if c in _LETTERS), rng.choice(_LETTERS))
    series = "".join(rng.choice(_LETTERS) for _ in range(3))
    digits = f"{rng.randint(1, 9999):04d}"
    return f"{series}{holder_type}{initial}{digits}{rng.choice(_LETTERS)}"


def gstin_check_char(first14: str) -> str:
    """Official GSTIN check character (mod 36) for the first 14 characters."""
    if len(first14) != 14 or any(c not in _ALNUM36 for c in first14):
        raise ValueError(f"expected 14 characters from 0-9A-Z, got {first14!r}")
    total = 0
    for index, char in enumerate(first14):
        product = _ALNUM36.index(char) * (1 if index % 2 == 0 else 2)
        total += product // 36 + product % 36
    return _ALNUM36[(36 - total % 36) % 36]


def generate_gstin(rng: random.Random, state_code: str, pan: str) -> str:
    """GSTIN = state code (2) + PAN (10) + entity number (1) + 'Z' + check character (1).

    The entity number counts registrations of the same PAN in one state; it is usually 1.
    """
    if state_code not in STATES:
        raise ValueError(f"unknown GST state code {state_code!r}")
    if not _PAN_FORMAT.fullmatch(pan):
        raise ValueError(f"invalid PAN {pan!r}")
    entity = rng.choices("123", weights=[85, 10, 5])[0]
    first14 = f"{state_code}{pan}{entity}Z"
    return first14 + gstin_check_char(first14)


def is_valid_gstin(gstin: str) -> bool:
    """True if `gstin` has the GSTIN format, a known state code and a correct check character."""
    if not isinstance(gstin, str) or not _GSTIN_FORMAT.fullmatch(gstin):
        return False
    if gstin[:2] not in VALID_GSTIN_STATE_CODES:
        return False
    return gstin_check_char(gstin[:14]) == gstin[14]


def pan_from_gstin(gstin: str) -> str:
    """The PAN embedded in a GSTIN (characters 3-12). Raises ValueError if the GSTIN is invalid."""
    if not is_valid_gstin(gstin):
        raise ValueError(f"invalid GSTIN {gstin!r}")
    return gstin[2:12]


# =========================================================================================
# People (Faker en_IN: names, phones, emails only)
# =========================================================================================


def make_faker(rng: random.Random) -> Faker:
    """A Faker en_IN instance seeded from `rng`. Create once per dataset and reuse it."""
    fake = Faker("en_IN")
    fake.seed_instance(rng.getrandbits(64))
    return fake


@dataclass(frozen=True)
class Contact:
    """A person at a company."""

    name: str
    phone: str
    email: str


def generate_contact(fake: Faker, domain: str) -> Contact:
    """A contact person with an Indian name, phone number and an email on the company domain."""
    first, last = fake.first_name(), fake.last_name()
    local = fake.random_element(
        [f"{first}.{last}", f"{first[0]}{last}", first, "purchase", "accounts", "sales"]
    )
    return Contact(
        name=f"{first} {last}",
        phone=fake.phone_number(),
        email=f"{re.sub(r'[^a-z0-9.]', '', local.lower())}@{domain}",
    )


# =========================================================================================
# Companies and addresses
# =========================================================================================

INDUSTRIES = (
    "industrial_equipment",
    "chemicals",
    "electrical",
    "fasteners_hardware",
    "it_office",
    "services",
)

_NAME_PREFIXES = (
    "Shree Ganesh", "Shri Balaji", "Sai", "Om", "Jay Ambe", "Maruti", "Laxmi", "Siddhi Vinayak",
    "Mahalaxmi", "Krishna", "Bharat", "Hindustan", "National", "Universal", "Supreme",
    "Precision", "Pioneer", "Apex", "Sterling", "Kalpataru", "Navkar", "Parshwa", "Sahyadri",
    "Kaveri", "Narmada", "Ganga", "Deccan", "Trident", "Vardhman", "Rajlaxmi",
)  # fmt: skip
_FAMILY_NAMES = (
    "Patel", "Shah", "Mehta", "Desai", "Joshi", "Kulkarni", "Deshpande", "Iyer", "Reddy",
    "Naidu", "Agarwal", "Gupta", "Jain", "Bansal", "Malhotra", "Kapoor", "Banerjee", "Ghosh",
    "Rao", "Pillai", "Chopra", "Sethi", "Thakkar", "Bhatt",
)  # fmt: skip
_COINED = (
    "Technocraft", "Polymech", "Hydrotech", "Unitech", "Metaltech", "Flowtek", "Electrocon",
    "Chemtrade", "Fastnet", "Infobyte", "Powertronix", "Indotech", "Duraflex", "Accuspec",
)  # fmt: skip
_SECTOR_WORDS: dict[str, tuple[str, ...]] = {
    "industrial_equipment": (
        "Engineering Works", "Engineers", "Industries", "Machine Tools", "Hydraulics",
        "Pumps & Valves", "Bearings", "Precision Components",
    ),
    "chemicals": ("Chemicals", "Chem Industries", "Polymers", "Petrochem", "Speciality Chemicals"),
    "electrical": ("Electricals", "Electric Co.", "Switchgears", "Power Systems", "Cables"),
    "fasteners_hardware": (
        "Fasteners", "Hardware Stores", "Steel Traders", "Bolts & Nuts", "Metals",
    ),
    "it_office": ("Infotech", "Computers", "Systems", "Technologies", "Office Solutions"),
    "services": ("Services", "Facility Management", "Consultants", "Logistics", "Solutions"),
}  # fmt: skip

# Legal suffix -> PAN holder type and relative weight.
_LEGAL_FORMS: tuple[tuple[str, str, int], ...] = (
    ("Pvt. Ltd.", PAN_COMPANY, 40),
    ("Private Limited", PAN_COMPANY, 12),
    ("Ltd.", PAN_COMPANY, 8),
    ("Limited", PAN_COMPANY, 5),
    ("LLP", PAN_FIRM, 12),
    ("& Co.", PAN_FIRM, 8),
    ("", PAN_PERSON, 15),  # proprietorship: PAN of the proprietor
)


_DOMAIN_NOISE = {"pvt", "ltd", "private", "limited", "llp", "co", "and", "shree", "shri", "the"}


@dataclass(frozen=True)
class Company:
    """A company name with its industry and the PAN holder type implied by its legal form."""

    name: str
    industry: str
    pan_holder_type: str

    @property
    def domain(self) -> str:
        """A plausible email/web domain from the name, e.g. 'patelengineering.in'.

        Deterministic (no random state): the TLD is chosen from a checksum of the name.
        """
        words = re.sub(r"[^a-z ]", "", self.name.lower()).split()
        stem = "".join([w for w in words if w not in _DOMAIN_NOISE][:2]) or "company"
        return stem + (".in", ".co.in", ".com")[sum(map(ord, self.name)) % 3]


def generate_company(rng: random.Random, industry: str) -> Company:
    """A company name that fits `industry`, e.g. 'Patel Engineering Works Pvt. Ltd.'."""
    if industry not in _SECTOR_WORDS:
        raise ValueError(f"unknown industry {industry!r}; expected one of {INDUSTRIES}")
    style = rng.choices(("prefix", "family", "coined"), weights=[45, 35, 20])[0]
    lead = rng.choice({"prefix": _NAME_PREFIXES, "family": _FAMILY_NAMES, "coined": _COINED}[style])
    suffix, holder_type, _ = rng.choices(_LEGAL_FORMS, weights=[f[2] for f in _LEGAL_FORMS])[0]
    if suffix == "& Co.":  # partnership firms are usually "<Family> & Co."
        lead = rng.choice(_FAMILY_NAMES)
        return Company(f"{lead} & Co.", industry, holder_type)
    name = f"{lead} {rng.choice(_SECTOR_WORDS[industry])} {suffix}".strip()
    return Company(name, industry, holder_type)


@dataclass(frozen=True)
class Address:
    """A postal address. `lines` is what gets printed, joined with ', ' or newlines."""

    lines: tuple[str, ...]
    city: str
    state: State
    pin: str

    def one_line(self) -> str:
        """The full address on one line, as stored in ground truth."""
        return ", ".join(self.lines)


_GALA_BUILDINGS = ("Shiv Shakti Industrial Estate", "Laxmi Industrial Complex", "Sai Udyog Bhavan")


def _unit_number(rng: random.Random) -> str:
    """First address line: plot, survey, shed or gala (Mumbai-style unit) number."""
    style = rng.choices(("plot", "plot_block", "survey", "shed", "gala"), [35, 20, 15, 15, 15])[0]
    if style == "plot":
        return f"Plot No. {rng.randint(1, 450)}"
    if style == "plot_block":
        return f"Plot No. {rng.choice('ABCDE')}-{rng.randint(1, 120)}"
    if style == "survey":
        return f"Survey No. {rng.randint(10, 990)}/{rng.randint(1, 9)}"
    if style == "shed":
        return f"Shed No. {rng.randint(1, 80)}"
    return f"Gala No. {rng.randint(1, 60)}, {rng.choice(_GALA_BUILDINGS)}"


def generate_address(rng: random.Random, state: State) -> Address:
    """An industrial address in `state`: plot/survey number, estate, city, state and PIN."""
    if state.code not in CITIES:
        raise ValueError(f"no city data for {state.name} ({state.code}); see CITIES")
    city = rng.choice(CITIES[state.code])
    estate = rng.choice(city.estates)
    if rng.random() < 0.3:
        estate += f", Phase {rng.choice(['I', 'II', 'III'])}"
    pin = f"{city.pin_prefix}{rng.randint(1, 999):03d}"
    lines = (_unit_number(rng), estate, f"{city.name} - {pin}", state.name)
    return Address(lines, city.name, state, pin)


# =========================================================================================
# GST slabs
# =========================================================================================

# GST rate (percent) -> relative weight. VERIFY against the current official CBIC rate
# notifications before generating a dataset (PRD §11.1). Default: the structure in force from
# 22 Sep 2025 (0 / 5 / 18 / 40). Catalogue items carry their own HSN-appropriate rate, and a
# test checks every catalogue rate is listed here; the weights are used by pick_gst_rate()
# when a rate is not tied to an item.
GST_SLABS: dict[Decimal, int] = {
    Decimal("0"): 3,
    Decimal("5"): 12,
    Decimal("18"): 83,
    Decimal("40"): 2,
}


def pick_gst_rate(rng: random.Random) -> Decimal:
    """Pick a GST rate using the GST_SLABS weights."""
    return rng.choices(list(GST_SLABS), weights=list(GST_SLABS.values()))[0]


# =========================================================================================
# Item catalogue
# =========================================================================================

UOMS = ("NOS", "KG", "MTR", "LTR", "SET", "BOX", "HRS")


@dataclass(frozen=True)
class CatalogueItem:
    """A purchasable item. `hsn_sac` is an HSN code for goods or a SAC (99xxxx) for services.

    HSN/SAC codes and rates are representative for synthetic data; verify against the CBIC
    tariff before relying on them for anything else.
    """

    industry: str
    code_prefix: str
    description: str
    hsn_sac: str
    uom: str
    price_min: Decimal
    price_max: Decimal
    gst_rate: Decimal

    @property
    def is_service(self) -> bool:
        """Services use SAC codes, which all start with 99."""
        return self.hsn_sac.startswith("99")


def _items(
    industry: str, rows: list[tuple[str, str, str, str, int, int, int]]
) -> list[CatalogueItem]:
    """Build catalogue items from compact tuples."""
    return [
        CatalogueItem(industry, code, desc, hsn, uom, Decimal(lo), Decimal(hi), Decimal(rate))
        for code, desc, hsn, uom, lo, hi, rate in rows
    ]


# (code prefix, description, HSN/SAC, UoM, min price, max price, GST %)
CATALOGUE: list[CatalogueItem] = [
    *_items("industrial_equipment", [
        ("BRG", "Deep groove ball bearing 6205-2RS, 25 x 52 x 15 mm, double rubber sealed, "
                "C3 clearance, make SKF / FAG / NBC or equivalent", "848210", "NOS", 180, 420, 18),
        ("BRG", "Taper roller bearing 32210, 50 x 90 x 24.75 mm", "848220", "NOS", 950, 1900, 18),
        ("BRG", "Spherical roller bearing 22216 EK with adapter sleeve H316", "848230", "NOS",
                4200, 7800, 18),
        ("BRG", "Needle roller bearing NK 25/20", "848240", "NOS", 260, 540, 18),
        ("PBK", "Plummer block housing SN 516 with seals and locating rings", "848330", "NOS",
                2400, 4600, 18),
        ("SFT", "Transmission shaft EN8, 40 mm dia x 1200 mm, keyway both ends", "848310", "NOS",
                2800, 5200, 18),
        ("GBX", "Helical worm reduction gearbox, ratio 30:1, input 1440 rpm, size 4 inch, "
                "foot mounted, with output shaft both sides", "848340", "NOS", 18500, 42000, 18),
        ("CPL", "Flexible jaw coupling L-110 with spider", "848360", "NOS", 650, 1450, 18),
        ("VBT", "V-belt B-64 classical section", "401039", "NOS", 210, 380, 18),
        ("CHN", "Roller chain 10B-1 simplex, 5 m length", "731511", "NOS", 1100, 2200, 18),
        ("SPR", "Sprocket 10B-1, 19 teeth, pilot bore", "848390", "NOS", 380, 760, 18),
        ("PMP", "Centrifugal monoblock pump 5 HP, 3 phase, 65 x 50 mm, head 32 m", "841370",
                "NOS", 21000, 38500, 18),
        ("IMP", "Impeller for centrifugal pump, CI FG 260, 210 mm dia", "841391", "NOS",
                3200, 6800, 18),
        ("VLV", "Gate valve cast steel 50 NB, class 150, flanged ends, IBR approved", "848180",
                "NOS", 6200, 11800, 18),
        ("VLV", "Ball valve SS 316, 25 NB, screwed ends, full bore", "848180", "NOS",
                1450, 3200, 18),
        ("VLV", "Non-return valve swing type, CI, 80 NB", "848130", "NOS", 3800, 7400, 18),
        ("VLV", "Safety relief valve 25 x 40 NB, set pressure 10.5 kg/cm2", "848140", "NOS",
                5400, 12600, 18),
        ("PNC", "Pneumatic cylinder 63 bore x 200 stroke, double acting, with magnetic piston",
                "841231", "NOS", 4800, 9200, 18),
        ("HYC", "Hydraulic cylinder 80 bore x 50 rod x 300 stroke, 160 bar", "841221", "NOS",
                14500, 28000, 18),
        ("CMP", "Reciprocating air compressor 10 HP, 12 kg/cm2, 500 L receiver", "841480",
                "NOS", 98000, 165000, 18),
        ("FLT", "Air filter element for screw compressor, OEM equivalent", "842139", "NOS",
                1200, 2900, 18),
        ("SEL", "Mechanical seal 35 mm single spring, carbon vs ceramic", "848420", "NOS",
                1800, 4200, 18),
        ("GSK", "Spiral wound gasket 50 NB, 150#, SS 316 with graphite filler", "848410",
                "NOS", 240, 520, 18),
        ("TLS", "HSS twist drill set 1-13 mm, 25 pieces, in metal box", "820750", "SET",
                1450, 3200, 18),
        ("TLS", "Grinding wheel 300 x 40 x 76.2 mm, A46 grade", "680422", "NOS", 950, 2100, 18),
        ("GAU", "Pressure gauge 100 mm dial, 0-16 kg/cm2, glycerine filled, 1/2 inch BSP",
                "902620", "NOS", 650, 1450, 18),
    ]),
    *_items("chemicals", [
        ("CHM", "Sulphuric acid 98%, technical grade, in 35 L HDPE carboys, "
                "with test certificate and MSDS for each batch", "280700", "KG",
                9, 18, 5),
        ("CHM", "Hydrochloric acid 30-33%, commercial grade", "280610", "KG", 6, 14, 18),
        ("CHM", "Nitric acid 60%, technical grade", "280800", "KG", 32, 55, 5),
        ("CHM", "Caustic soda flakes 98%, 50 kg HDPE bags", "281511", "KG", 42, 68, 18),
        ("CHM", "Soda ash light, 99.2% min", "283620", "KG", 28, 46, 18),
        ("CHM", "Hydrogen peroxide 50% w/w, 50 kg carboy", "284700", "KG", 38, 72, 18),
        ("CHM", "Sodium hypochlorite solution 10-12% available chlorine", "282890", "KG",
                12, 26, 18),
        ("SOL", "Methanol 99.85% min, in 160 kg MS drums", "290511", "KG", 28, 44, 18),
        ("SOL", "Isopropyl alcohol (IPA) 99.8%, 160 kg drum", "290512", "KG", 82, 128, 18),
        ("SOL", "Acetone 99.5%, technical grade", "291411", "KG", 68, 104, 18),
        ("SOL", "Toluene, commercial grade", "290230", "KG", 64, 98, 18),
        ("SOL", "Ethyl acetate 99.5%", "291531", "KG", 78, 118, 18),
        ("CHM", "Glycerine IP grade 99.5%", "290545", "KG", 92, 160, 18),
        ("CHM", "Citric acid monohydrate, food grade", "291814", "KG", 85, 140, 18),
        ("PIG", "Titanium dioxide rutile grade, 25 kg bags", "282300", "KG", 240, 380, 18),
        ("RES", "Epoxy resin liquid, EEW 182-192, with polyamide hardener (mix ratio 100:50 by "
                "weight), shelf life min 12 months from date of manufacture", "390730",
                "KG", 290, 520, 18),
        ("LUB", "Hydraulic oil ISO VG 68, anti-wear, 210 L barrel", "271019", "LTR", 145, 240, 18),
        ("LUB", "Gear oil ISO VG 220, EP grade", "271019", "LTR", 165, 280, 18),
        ("LUB", "Lithium complex grease NLGI 2, 18 kg bucket", "271019", "KG", 210, 360, 18),
        ("PNT", "Synthetic enamel paint, RAL 7035 light grey, 20 L", "320890", "LTR",
                240, 420, 18),
        ("PNT", "Epoxy zinc phosphate primer, two pack", "320890", "LTR", 290, 520, 18),
        ("ADH", "Industrial adhesive, synthetic rubber based, 5 kg tin", "350691", "KG",
                220, 410, 18),
        ("CHM", "Activated carbon granular 8 x 30 mesh, iodine value 1000", "380210", "KG",
                95, 180, 18),
        ("POL", "LDPE granules, film grade, MFI 2", "390110", "KG", 98, 132, 18),
        ("POL", "PP homopolymer granules, injection moulding grade", "390210", "KG", 92, 124, 18),
    ]),
    *_items("electrical", [
        ("CBL", "PVC insulated armoured power cable, 3.5 core x 95 sq mm aluminium, 1.1 kV, "
                "as per IS 1554 Part 1, in drums of 500 m", "854449", "MTR", 380, 620, 18),
        ("CBL", "Flexible multicore copper cable 4 core x 2.5 sq mm, PVC sheathed", "854449",
                "MTR", 85, 160, 18),
        ("CBL", "FRLS copper wire 1.5 sq mm single core, 90 m coil", "854449", "BOX",
                1450, 2600, 18),
        ("SWG", "MCB 32A single pole, C curve, 10 kA", "853620", "NOS", 145, 320, 18),
        ("SWG", "MCCB 250A 3 pole, 36 kA, thermal magnetic release", "853620", "NOS",
                9800, 18500, 18),
        ("SWG", "RCCB 63A 4 pole, 30 mA sensitivity", "853630", "NOS", 2600, 5400, 18),
        ("SWG", "Power contactor 3 pole 32A, coil 230 V AC", "853649", "NOS", 1350, 2900, 18),
        ("SWG", "Control relay 24 V DC, 2 changeover contacts, with base", "853641", "NOS",
                280, 620, 18),
        ("SWG", "Push button 22 mm, green, 1 NO", "853650", "NOS", 95, 240, 18),
        ("SWG", "HRC fuse link 63A, DIN type", "853610", "NOS", 120, 260, 18),
        ("MTR", "3 phase induction motor 7.5 kW (10 HP), 4 pole, 1440 rpm, foot mounted, "
                "IE3 efficiency, TEFC, IP55", "850152", "NOS", 28500, 46000, 18),
        ("MTR", "Single phase motor 0.75 kW, 2800 rpm, capacitor start", "850140", "NOS",
                5600, 9400, 18),
        ("DRV", "AC variable frequency drive 11 kW, 415 V, with keypad", "850440", "NOS",
                38000, 72000, 18),
        ("TRF", "Control transformer 500 VA, 415/230 V", "850431", "NOS", 2400, 4600, 18),
        ("PNL", "LT distribution panel with 1 incomer and 6 outgoings, powder coated, IP54",
                "853710", "NOS", 65000, 145000, 18),
        ("LGT", "LED high bay luminaire 150 W, 5700 K, IP65", "940542", "NOS", 3800, 7600, 18),
        ("LGT", "LED lamp 12 W, B22 base, cool daylight", "853952", "NOS", 75, 160, 18),
        ("BAT", "SMF VRLA battery 12 V 100 Ah", "850720", "NOS", 9200, 15800, 18),
        ("MTR", "Digital energy meter 3 phase 4 wire, RS485 Modbus", "902830", "NOS",
                3400, 7200, 18),
        ("ACC", "Double compression cable gland 32 mm, brass", "853690", "NOS", 85, 210, 18),
        ("ACC", "PVC insulation tape 18 mm x 8 m", "391910", "BOX", 180, 420, 18),
    ]),
    *_items("fasteners_hardware", [
        ("FST", "Hex bolt M12 x 50, grade 8.8, zinc plated, IS 1364, full thread, with mill test "
                "certificate, packed in 25 kg HDPE bags", "731815", "KG",
                120, 210, 18),
        ("FST", "Hex nut M12, grade 8, zinc plated", "731816", "KG", 125, 215, 18),
        ("FST", "Allen cap screw M8 x 25, grade 12.9, black finish", "731815", "BOX",
                380, 720, 18),
        ("FST", "Spring washer M12, zinc plated", "731821", "KG", 140, 260, 18),
        ("FST", "Plain washer M16, MS, zinc plated", "731822", "KG", 110, 190, 18),
        ("FST", "Self tapping screw No. 10 x 25 mm, pan head, SS 304", "731814", "BOX",
                260, 540, 18),
        ("FST", "Anchor fastener M12 x 100, sleeve type", "731819", "NOS", 28, 64, 18),
        ("FST", "Stud bolt M16 x 120 with 2 nuts, B7 / 2H", "731819", "NOS", 85, 160, 18),
        ("FST", "Blind rivet 4.8 x 12 mm, aluminium", "731823", "BOX", 210, 420, 18),
        ("FST", "Threaded rod M10 x 1 m, GI", "731819", "NOS", 65, 130, 18),
        ("HDW", "Heavy duty hinge 6 inch, MS, zinc plated", "830210", "NOS", 95, 210, 18),
        ("HDW", "Padlock 65 mm, brass body, 3 keys", "830110", "NOS", 260, 640, 18),
        ("HDW", "Hydraulic door closer for doors up to 80 kg", "830260", "NOS", 1450, 3200, 18),
        ("HDW", "Steel wire rope 12 mm, 6 x 36 IWRC, galvanised", "731210", "MTR", 120, 240, 18),
        ("STL", "MS angle 50 x 50 x 6 mm, IS 2062 E250", "721621", "KG", 58, 76, 18),
        ("STL", "GI pipe 50 NB medium class, IS 1239, 6 m length", "730630", "MTR",
                420, 640, 18),
        ("STL", "Welded wire mesh 50 x 50 mm, 3 mm wire, GI", "731420", "MTR", 160, 320, 18),
        ("STL", "SS 304 sheet 2 mm thick, 2B finish, 1250 x 2500 mm, PVC coated one side, as per "
                "ASTM A240, with mill test certificate", "721933", "KG", 230, 320, 18),
        ("HDW", "Wire nails 2 inch, bright finish", "731700", "KG", 72, 110, 18),
        ("HDW", "Bench vice 6 inch, cast iron, swivel base", "820570", "NOS", 2400, 4800, 18),
        ("HDW", "Double ended spanner set 6-32 mm, 12 pieces, chrome vanadium", "820411",
                "SET", 1100, 2600, 18),
    ]),
    *_items("it_office", [
        ("ITL", "Laptop 14 inch, Intel Core i5 13th gen, 16 GB RAM, 512 GB SSD, "
                "Windows 11 Pro, 3 years onsite warranty", "847130", "NOS", 58000, 82000, 18),
        ("ITD", "Desktop computer, Intel Core i7, 16 GB RAM, 1 TB SSD, with keyboard and mouse",
                "847150", "NOS", 62000, 92000, 18),
        ("ITM", "LED monitor 24 inch, full HD, IPS, HDMI and DP", "852852", "NOS",
                9800, 16500, 18),
        ("ITP", "Laser multifunction printer, A4, duplex, network, 40 ppm", "844331", "NOS",
                24000, 52000, 18),
        ("ITP", "Toner cartridge, black, compatible, 3000 page yield", "844399", "NOS",
                1800, 5400, 18),
        ("ITA", "Wireless keyboard and mouse combo", "847160", "SET", 1200, 2800, 18),
        ("ITS", "Portable SSD 1 TB, USB 3.2", "852351", "NOS", 6800, 11500, 18),
        ("ITN", "Managed network switch 24 port gigabit with 4 SFP uplinks", "851762", "NOS",
                18500, 42000, 18),
        ("ITN", "Wi-Fi 6 access point, ceiling mount, PoE", "851762", "NOS", 9200, 18500, 18),
        ("ITN", "Cat 6 UTP cable, 305 m box", "854449", "BOX", 6800, 12500, 18),
        ("ITU", "Online UPS 3 kVA, single phase in / single phase out, with SMF batteries for "
                "30 min backup at full load, including installation", "850440", "NOS",
                38000, 68000, 18),
        ("OFF", "Copier paper A4, 75 GSM, 500 sheets per ream, 5 reams per box", "480256",
                "BOX", 1250, 1650, 18),
        ("OFF", "Ergonomic office chair, high back, mesh, adjustable arms", "940130", "NOS",
                7800, 16500, 18),
        ("OFF", "Modular workstation 1200 x 600 mm, pre-laminated board, with pedestal",
                "940330", "NOS", 14500, 28500, 18),
        ("OFF", "Steel filing cabinet, 4 drawer, powder coated", "940310", "NOS",
                11500, 19800, 18),
        ("OFF", "Magnetic whiteboard 4 x 3 ft, aluminium frame", "961010", "NOS",
                2400, 4800, 18),
        ("OFF", "Ball pen blue, box of 50", "960810", "BOX", 250, 480, 18),
        ("OFF", "Technical reference books (printed), set of 5 volumes", "490199", "SET",
                4200, 9800, 0),
        ("CNT", "Carbonated soft drink cans 300 ml, pack of 24 (pantry supply)", "220210",
                "BOX", 720, 1100, 40),
    ]),
    *_items("services", [
        ("AMC", "Annual maintenance contract for CNC machines, 4 preventive visits per year "
                "and breakdown support within 24 hours, spares extra at actuals", "998717",
                "NOS", 85000, 240000, 18),
        ("SRV", "Repair and rewinding of 30 kW induction motor", "998717", "NOS",
                18000, 42000, 18),
        ("AMC", "Annual maintenance of computers and peripherals, 60 nodes", "998713", "NOS",
                72000, 180000, 18),
        ("INS", "Installation and commissioning of air compressor", "998732", "NOS",
                12000, 38000, 18),
        ("INS", "Electrical installation and cabling work for new production line", "995461",
                "NOS", 45000, 180000, 18),
        ("CAL", "Calibration of pressure and temperature gauges, NABL accredited, "
                "with certificates", "998349", "NOS", 450, 1200, 18),
        ("MNP", "Supply of skilled manpower (fitter), as per attendance", "998519", "HRS",
                180, 320, 18),
        ("SEC", "Security guard services, 12 hour shift", "998525", "HRS", 75, 140, 18),
        ("HKP", "Housekeeping and cleaning services, plant and office", "998533", "HRS",
                65, 120, 18),
        ("PST", "Pest control treatment, quarterly", "998531", "NOS", 6500, 18000, 18),
        ("TRN", "Road transportation of goods by GTA, Pune to Chennai, 9 MT closed body truck, "
                "including loading supervision and transit insurance", "996791",
                "NOS", 38000, 62000, 5),
        ("CUR", "Courier charges for documents and samples, monthly", "996812", "NOS",
                1800, 6500, 18),
        ("ITC", "IT consulting and network audit", "998313", "HRS", 1800, 4500, 18),
        ("DEV", "Customisation of ERP reports and integration, as per scope document "
                "rev. 2", "998314", "HRS", 1400, 3200, 18),
        ("LIC", "Antivirus licence renewal, 1 year, 100 users", "997331", "NOS",
                42000, 98000, 18),
        ("CLD", "Cloud server hosting, 8 vCPU / 32 GB, monthly", "998315", "NOS",
                18000, 42000, 18),
        ("TRG", "Training programme on industrial safety, 2 days, up to 25 participants",
                "999293", "NOS", 45000, 120000, 18),
        ("TST", "Third party inspection of pressure vessel as per IBR", "998349", "NOS",
                18000, 52000, 18),
    ]),
]  # fmt: skip


def items_for(industry: str) -> list[CatalogueItem]:
    """All catalogue items of one industry."""
    if industry not in INDUSTRIES:
        raise ValueError(f"unknown industry {industry!r}; expected one of {INDUSTRIES}")
    return [item for item in CATALOGUE if item.industry == industry]


ITEM_CODE_STYLES = ("dash", "slash", "sap")


def pick_item_code_style(rng: random.Random) -> str:
    """One buyer uses one coding style for all its items; pick it once per buyer."""
    return rng.choices(ITEM_CODE_STYLES, weights=[50, 25, 25])[0]


def generate_item_code(rng: random.Random, item: CatalogueItem, style: str | None = None) -> str:
    """A buyer item code in one of the common styles: BRG-6205, BRG/0421/A, 10004521 (SAP).

    `style` is one of ITEM_CODE_STYLES; if None, a style is picked at random.
    """
    if style is None:
        style = pick_item_code_style(rng)
    if style not in ITEM_CODE_STYLES:
        raise ValueError(f"unknown item code style {style!r}; expected one of {ITEM_CODE_STYLES}")
    if style == "dash":
        return f"{item.code_prefix}-{rng.randint(100, 9999)}"
    if style == "slash":
        return f"{item.code_prefix}/{rng.randint(1, 999):04d}/{rng.choice('ABC')}"
    return str(rng.randint(10_000_000, 19_999_999))


def generate_unit_price(rng: random.Random, item: CatalogueItem) -> Decimal:
    """A unit price within the item's range; whole rupees for most items, paise for some.

    Drawn as an integer number of paise, so no float is ever involved.
    """
    paise = rng.randint(int(item.price_min * 100), int(item.price_max * 100))
    if rng.random() < 0.6:
        paise = max(100, paise - paise % 100)  # whole rupees
    return (Decimal(paise) / 100).quantize(Decimal("0.01"))


# =========================================================================================
# Amount in words (Indian numbering)
# =========================================================================================


_CRORE = 10_000_000


def _indian_words(number: int) -> str:
    """'Two Lakh Forty Five Thousand Three Hundred Twelve' for 245312 (no 'and', no hyphens).

    Crores are handled here so any size works ("One Lakh Crore"); num2words' en_IN converter
    is only used below one crore.
    """
    if number >= _CRORE:
        crores, rest = divmod(number, _CRORE)
        head = f"{_indian_words(crores)} Crore"
        return f"{head} {_indian_words(rest)}" if rest else head
    words = num2words(number, lang="en_IN")
    words = words.replace(",", " ").replace("-", " ")
    return " ".join(w.capitalize() for w in words.split() if w != "and")


def amount_in_words_inr(amount: Decimal) -> str:
    """Amount in words in the Indian system, as printed on Indian POs and invoices.

    245312.50 -> 'Rupees Two Lakh Forty Five Thousand Three Hundred Twelve and Fifty Paise Only'
    """
    if not isinstance(amount, Decimal):
        raise TypeError(f"amount must be Decimal, got {type(amount).__name__}")
    if amount < 0:
        raise ValueError(f"amount must not be negative, got {amount}")
    if amount != amount.quantize(Decimal("0.01")):
        raise ValueError(f"amount has more than 2 decimal places: {amount}")
    rupees = int(amount)
    paise = int((amount - rupees) * 100)
    text = f"Rupees {_indian_words(rupees)}"
    if paise:
        text += f" and {_indian_words(paise)} Paise"
    return f"{text} Only"


# =========================================================================================
# Terms text pools
# =========================================================================================

PAYMENT_TERMS = (
    "30 days from date of invoice",
    "45 days from receipt of material",
    "60 days credit from date of GRN",
    "100% against delivery",
    "Advance 30%, balance 70% against delivery",
    "Payment within 30 days after receipt of material and invoice, subject to acceptance",
    "90 days by RTGS / NEFT",
    "Immediate, against proforma invoice",
    "Through LC at 60 days sight",
    "15 days from date of installation and commissioning",
)
DELIVERY_TERMS = (
    "FOR Destination",
    "FOR our works",
    "Ex-Works",
    "Door delivery at our plant",
    "FOB",
    "CIF",
    "Ex-Godown",
    "Delivered at site, unloading by us",
)
FREIGHT_TERMS = (
    "Freight paid",
    "To pay",
    "Freight extra at actuals",
    "Included in price",
    "Freight to be borne by supplier",
    "Freight extra, 2% of basic value",
)
TRANSPORT_MODES = (
    "By Road",
    "By Road (vendor's vehicle)",
    "Through transporter",
    "Courier",
    "By Air",
    "By Rail",
    "Hand delivery",
)
WARRANTY_TERMS = (
    "12 months from date of supply",
    "18 months from supply or 12 months from commissioning, whichever is earlier",
    "24 months against manufacturing defects",
    "As per manufacturer's standard warranty",
    "Not applicable",
    "36 months comprehensive onsite warranty",
)
PACKING_INSTRUCTIONS = (
    "Standard seaworthy packing",
    "Suitable for road transport, wooden box packing",
    "Packing in HDPE bags with inner liner",
    "Palletised and stretch wrapped",
    "Packing and forwarding included",
    "Each item to be tagged with PO number and item code",
    "In original manufacturer's packing, sealed",
)
