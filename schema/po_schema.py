"""Pydantic v2 schema for an Indian GST purchase order.

Single source of truth for the extraction output (PRD §6). Field descriptions double as
instructions inside the LLM JSON schema, so they name the labels a field commonly appears under.

Critical fields (PRD §6, marked *) are required; every other field is optional.
Money, quantities and rates are Decimal so no value ever passes through a float.
Dates are extracted exactly as printed and parsed day-first in code (schema/dates.py).
"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from schema.dates import PODate


class LineItem(BaseModel):
    """One row of the PO item table (PRD §6.2). Field order = compact row column order."""

    model_config = ConfigDict(extra="forbid")

    line_no: int = Field(
        description="Serial number of the row as printed. Labels: 'S.No.', 'Sr.', 'Item No.', '#'."
    )
    item_code: str | None = Field(
        default=None,
        description="Item or part code. Labels: 'Item Code', 'Part No.', 'Material Code', 'SKU'.",
    )
    description: str = Field(
        description=(
            "Item description as printed, wrapped lines joined with a space. "
            "Labels: 'Description', 'Particulars', 'Item Description', 'Material'."
        )
    )
    hsn_sac: str = Field(
        description=(
            "HSN (goods) or SAC (services) code as printed. Labels: 'HSN', 'HSN/SAC', 'SAC'."
        )
    )
    quantity: Decimal = Field(
        description="Ordered quantity. Labels: 'Qty', 'Quantity', 'Order Qty'."
    )
    uom: str | None = Field(
        default=None,
        description=(
            "Unit of measure as printed, e.g. 'Nos', 'Kg', 'Mtr', 'Set'. Labels: 'UOM', 'Unit'."
        ),
    )
    unit_rate: Decimal = Field(
        description=(
            "Price per unit before discount and GST. Labels: 'Rate', 'Unit Price', 'Price/Unit'."
        )
    )
    discount_pct: Decimal | None = Field(
        default=None,
        description="Line discount in percent (10 means 10%). Labels: 'Disc %', 'Discount %'.",
    )
    taxable_value: Decimal = Field(
        description=(
            "Line amount after discount, before GST. "
            "Labels: 'Taxable Value', 'Taxable Amt', 'Basic Value', "
            "'Amount' (when shown before tax)."
        )
    )
    gst_rate: Decimal = Field(
        description=(
            "Total GST rate in percent (CGST+SGST, or IGST), e.g. 18. Labels: 'GST %', 'Tax Rate'."
        )
    )
    cgst_amount: Decimal | None = Field(
        default=None, description="Central GST amount for the row. Labels: 'CGST', 'CGST Amt'."
    )
    sgst_amount: Decimal | None = Field(
        default=None,
        description="State/UT GST amount for the row. Labels: 'SGST', 'UTGST', 'SGST Amt'.",
    )
    igst_amount: Decimal | None = Field(
        default=None, description="Integrated GST amount for the row. Labels: 'IGST', 'IGST Amt'."
    )
    line_total: Decimal = Field(
        description=(
            "Row amount including GST. Labels: 'Total', 'Line Total', 'Net Amount', "
            "'Amount (incl. tax)'."
        )
    )
    line_delivery_date: PODate | None = Field(
        default=None,
        description=(
            "Delivery date for this row if given per row, exactly as printed. "
            "Labels: 'Delivery Date', 'Due Date', 'Schedule'."
        ),
    )


class PurchaseOrder(BaseModel):
    """A complete purchase order: 58 header fields (PRD §6.1) plus line items."""

    model_config = ConfigDict(extra="forbid")

    # --- Document (10) ---
    po_number: str = Field(
        description=(
            "Purchase order number. "
            "Labels: 'PO No.', 'P.O. Number', 'Order No.', 'P.O. Ref', 'PO#'."
        )
    )
    po_date: PODate = Field(
        description=(
            "Date the PO was issued, exactly as printed, e.g. '14/03/2026', '14-Mar-2026'. "
            "Labels: 'PO Date', 'Order Date', 'Date', 'Dated'."
        )
    )
    amendment_no: str | None = Field(
        default=None,
        description=(
            "Amendment/revision number of the PO. Labels: 'Amendment No.', 'Rev. No.', 'Revision'."
        ),
    )
    amendment_date: PODate | None = Field(
        default=None,
        description=(
            "Date of the amendment, exactly as printed. Labels: 'Amendment Date', 'Rev. Date'."
        ),
    )
    quotation_ref: str | None = Field(
        default=None,
        description=(
            "Vendor quotation the PO is based on. "
            "Labels: 'Quotation Ref', 'Your Ref', 'Offer No.', 'Quote No.'."
        ),
    )
    quotation_date: PODate | None = Field(
        default=None,
        description=(
            "Date of the vendor quotation, exactly as printed. "
            "Labels: 'Quotation Date', 'Offer Date'."
        ),
    )
    indent_no: str | None = Field(
        default=None,
        description=(
            "Internal requisition number. Labels: 'Indent No.', 'PR No.', 'Requisition No.'."
        ),
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code, e.g. 'INR', 'USD'. Map '₹', 'Rs.', 'Rupees' to 'INR'.",
    )
    delivery_date: PODate | None = Field(
        default=None,
        description=(
            "Required delivery date for the whole order, exactly as printed. "
            "Labels: 'Delivery Date', 'Delivery By', 'Required By', 'Due Date'."
        ),
    )
    po_validity_date: PODate | None = Field(
        default=None,
        description=(
            "Date until which the PO is valid, exactly as printed. "
            "Labels: 'Valid Till', 'Validity', 'PO Expiry'."
        ),
    )

    # --- Buyer (9): the company issuing the PO, usually on the letterhead ---
    buyer_name: str = Field(
        description=(
            "Legal name of the company issuing the PO (usually the letterhead). "
            "Labels: 'Buyer', 'Purchaser', 'Customer', 'From'."
        )
    )
    buyer_address: str | None = Field(
        default=None, description="Buyer's full postal address as printed, lines joined with ', '."
    )
    buyer_gstin: str = Field(
        description=(
            "Buyer's 15-character GSTIN (2-digit state code + 10-character PAN + 3 characters). "
            "Labels: 'GSTIN', 'GST No.', 'GSTIN/UIN'."
        )
    )
    buyer_pan: str | None = Field(
        default=None, description="Buyer's 10-character PAN. Labels: 'PAN', 'PAN No.'."
    )
    buyer_state: str | None = Field(
        default=None, description="Buyer's state name, e.g. 'Maharashtra'."
    )
    buyer_state_code: str | None = Field(
        default=None,
        description="Buyer's 2-digit GST state code as text, e.g. '27'. Labels: 'State Code'.",
    )
    buyer_contact_person: str | None = Field(
        default=None,
        description=(
            "Buyer's contact person name. "
            "Labels: 'Contact Person', 'Kind Attn', 'Purchase Officer'."
        ),
    )
    buyer_phone: str | None = Field(
        default=None, description="Buyer's phone number as printed. Labels: 'Phone', 'Tel', 'Mob'."
    )
    buyer_email: str | None = Field(default=None, description="Buyer's email address.")

    # --- Vendor (10): the supplier receiving the PO ---
    vendor_name: str = Field(
        description=(
            "Legal name of the supplier receiving the PO. "
            "Labels: 'Vendor', 'Supplier', 'Seller', 'To', 'M/s'."
        )
    )
    vendor_code: str | None = Field(
        default=None,
        description=(
            "Buyer's code for the vendor. Labels: 'Vendor Code', 'Supplier Code', 'Vendor ID'."
        ),
    )
    vendor_address: str | None = Field(
        default=None, description="Vendor's full postal address as printed, lines joined with ', '."
    )
    vendor_gstin: str = Field(
        description=(
            "Vendor's 15-character GSTIN (2-digit state code + 10-character PAN + 3 characters). "
            "Labels: 'GSTIN', 'GST No.', 'Supplier GSTIN'."
        )
    )
    vendor_pan: str | None = Field(
        default=None, description="Vendor's 10-character PAN. Labels: 'PAN', 'PAN No.'."
    )
    vendor_state: str | None = Field(
        default=None, description="Vendor's state name, e.g. 'Gujarat'."
    )
    vendor_state_code: str | None = Field(
        default=None,
        description="Vendor's 2-digit GST state code as text, e.g. '24'. Labels: 'State Code'.",
    )
    vendor_contact_person: str | None = Field(
        default=None, description="Vendor's contact person name. Labels: 'Contact Person', 'Attn'."
    )
    vendor_phone: str | None = Field(
        default=None, description="Vendor's phone number as printed. Labels: 'Phone', 'Tel', 'Mob'."
    )
    vendor_email: str | None = Field(default=None, description="Vendor's email address.")

    # --- Bill-to (4) ---
    bill_to_name: str | None = Field(
        default=None,
        description=(
            "Name of the party invoices go to. Labels: 'Bill To', 'Invoice To', 'Billing Address'."
        ),
    )
    bill_to_address: str | None = Field(
        default=None, description="Bill-to postal address as printed, lines joined with ', '."
    )
    bill_to_gstin: str | None = Field(
        default=None, description="Bill-to party's 15-character GSTIN."
    )
    bill_to_state_code: str | None = Field(
        default=None, description="Bill-to 2-digit GST state code as text, e.g. '27'."
    )

    # --- Ship-to (4) ---
    ship_to_name: str | None = Field(
        default=None,
        description=(
            "Name of the party receiving the goods. "
            "Labels: 'Ship To', 'Consignee', 'Deliver To', 'Delivery Address'."
        ),
    )
    ship_to_address: str | None = Field(
        default=None, description="Ship-to postal address as printed, lines joined with ', '."
    )
    ship_to_gstin: str | None = Field(
        default=None, description="Ship-to (consignee) 15-character GSTIN."
    )
    ship_to_state_code: str | None = Field(
        default=None, description="Ship-to 2-digit GST state code as text, e.g. '27'."
    )

    # --- Terms (8) ---
    payment_terms: str | None = Field(
        default=None,
        description=(
            "Payment terms as printed, e.g. '30 days from invoice'. "
            "Labels: 'Payment Terms', 'Terms of Payment'."
        ),
    )
    delivery_terms: str | None = Field(
        default=None,
        description=(
            "Delivery/price basis, e.g. 'FOR Destination', 'Ex-Works'. "
            "Labels: 'Delivery Terms', 'Price Basis'."
        ),
    )
    freight_terms: str | None = Field(
        default=None,
        description=(
            "Who bears freight, e.g. 'Freight Paid', 'To Pay'. Labels: 'Freight Terms', 'Freight'."
        ),
    )
    mode_of_transport: str | None = Field(
        default=None,
        description=(
            "Transport mode, e.g. 'Road', 'Courier'. "
            "Labels: 'Mode of Dispatch', 'Despatch Through', 'Transport'."
        ),
    )
    place_of_supply: str | None = Field(
        default=None,
        description=(
            "GST place of supply (state, with code if shown) as printed. "
            "Labels: 'Place of Supply', 'POS'."
        ),
    )
    delivery_location: str | None = Field(
        default=None,
        description=(
            "Plant, site or city where goods are delivered. "
            "Labels: 'Delivery At', 'Place of Delivery', 'Site'."
        ),
    )
    warranty_terms: str | None = Field(
        default=None,
        description="Warranty/guarantee terms as printed. Labels: 'Warranty', 'Guarantee'.",
    )
    packing_instructions: str | None = Field(
        default=None,
        description="Packing instructions as printed. Labels: 'Packing', 'Packing Instructions'.",
    )

    # --- Totals (11) ---
    subtotal: Decimal = Field(
        description=(
            "Sum of line taxable values, before GST and charges. "
            "Labels: 'Sub Total', 'Taxable Amount', 'Basic Total', 'Total Before Tax'."
        )
    )
    discount_total: Decimal | None = Field(
        default=None,
        description=(
            "Discount amount in the totals block, as a positive number. "
            "Labels: 'Discount', 'Less: Discount'."
        ),
    )
    freight_charges: Decimal | None = Field(
        default=None,
        description="Freight amount. Labels: 'Freight', 'Freight Charges', 'Transportation'.",
    )
    other_charges: Decimal | None = Field(
        default=None,
        description=(
            "Packing, forwarding, insurance or other charges. "
            "Labels: 'P&F', 'Other Charges', 'Insurance'."
        ),
    )
    cgst_total: Decimal = Field(
        description="Total Central GST amount. Labels: 'CGST', 'Total CGST'."
    )
    sgst_total: Decimal = Field(
        description="Total State/UT GST amount. Labels: 'SGST', 'UTGST', 'Total SGST'."
    )
    igst_total: Decimal = Field(
        description="Total Integrated GST amount. Labels: 'IGST', 'Total IGST'."
    )
    total_tax: Decimal | None = Field(
        default=None,
        description="Total GST amount. Labels: 'Total Tax', 'Total GST', 'Tax Amount'.",
    )
    round_off: Decimal | None = Field(
        default=None,
        description=(
            "Rounding adjustment, negative if deducted. Labels: 'Round Off', 'Rounding', 'R/O'."
        ),
    )
    grand_total: Decimal = Field(
        description=(
            "Final PO value including GST and charges. "
            "Labels: 'Grand Total', 'Total PO Value', 'Total Amount', 'Net Payable'."
        )
    )
    amount_in_words: str | None = Field(
        default=None,
        description=(
            "Grand total in words exactly as printed. Labels: 'Amount in Words', 'Rupees ... Only'."
        ),
    )

    # --- Approval (2) ---
    prepared_by: str | None = Field(
        default=None,
        description="Name of the person who prepared the PO. Labels: 'Prepared By', 'Created By'.",
    )
    approved_by: str | None = Field(
        default=None,
        description=(
            "Name of the approver as printed. Labels: 'Approved By', 'Authorised Signatory'."
        ),
    )

    line_items: list[LineItem] = Field(description="Rows of the item table, in document order.")
