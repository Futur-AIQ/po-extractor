"""Field metadata: critical fields, extraction groups and the compact line-item column order.

The section lists mirror PRD §6.1. Tests check that they cover the PurchaseOrder header
fields exactly once, so this module and po_schema.py cannot drift apart.
"""

from schema.po_schema import LineItem

DOCUMENT_FIELDS = [
    "po_number",
    "po_date",
    "amendment_no",
    "amendment_date",
    "quotation_ref",
    "quotation_date",
    "indent_no",
    "currency",
    "delivery_date",
    "po_validity_date",
]
BUYER_FIELDS = [
    "buyer_name",
    "buyer_address",
    "buyer_gstin",
    "buyer_pan",
    "buyer_state",
    "buyer_state_code",
    "buyer_contact_person",
    "buyer_phone",
    "buyer_email",
]
VENDOR_FIELDS = [
    "vendor_name",
    "vendor_code",
    "vendor_address",
    "vendor_gstin",
    "vendor_pan",
    "vendor_state",
    "vendor_state_code",
    "vendor_contact_person",
    "vendor_phone",
    "vendor_email",
]
BILL_TO_FIELDS = ["bill_to_name", "bill_to_address", "bill_to_gstin", "bill_to_state_code"]
SHIP_TO_FIELDS = ["ship_to_name", "ship_to_address", "ship_to_gstin", "ship_to_state_code"]
TERMS_FIELDS = [
    "payment_terms",
    "delivery_terms",
    "freight_terms",
    "mode_of_transport",
    "place_of_supply",
    "delivery_location",
    "warranty_terms",
    "packing_instructions",
]
TOTALS_FIELDS = [
    "subtotal",
    "discount_total",
    "freight_charges",
    "other_charges",
    "cgst_total",
    "sgst_total",
    "igst_total",
    "total_tax",
    "round_off",
    "grand_total",
    "amount_in_words",
]
APPROVAL_FIELDS = ["prepared_by", "approved_by"]

# One LLM sub-request per group; groups run in parallel over the same cached PO text.
FIELD_GROUPS: dict[str, list[str]] = {
    "G1_HEADER_TERMS": DOCUMENT_FIELDS + TERMS_FIELDS + APPROVAL_FIELDS,
    "G2_PARTIES": BUYER_FIELDS + VENDOR_FIELDS + BILL_TO_FIELDS + SHIP_TO_FIELDS,
    "G3_TOTALS": TOTALS_FIELDS,
}

CRITICAL_HEADER_FIELDS = {
    "po_number",
    "po_date",
    "buyer_name",
    "buyer_gstin",
    "vendor_name",
    "vendor_gstin",
    "subtotal",
    "cgst_total",
    "sgst_total",
    "igst_total",
    "grand_total",
}
CRITICAL_LINE_ITEM_FIELDS = {
    "line_no",
    "description",
    "hsn_sac",
    "quantity",
    "unit_rate",
    "taxable_value",
    "gst_rate",
    "line_total",
}
CRITICAL_FIELDS = CRITICAL_HEADER_FIELDS | CRITICAL_LINE_ITEM_FIELDS

# Column order of a compact line-item row: the LineItem field order (PRD §6.2).
LINE_ITEM_COLUMNS: list[str] = list(LineItem.model_fields)
