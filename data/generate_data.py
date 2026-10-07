"""Generate synthetic extractor output for the fraud pipeline.

Emits structured transaction fields with per-field confidence scores,
occasionally malformed so the ingestion layer has something to reject.
"""

import json
import os
import random
from datetime import date, timedelta

import numpy as np

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# GEN_TARGET lets a smaller dataset be generated without retraining.
N_CLEAN_TARGET = int(os.environ.get("GEN_TARGET", "284807"))
CORRUPTION_RATE = 0.012
N_RAW = int(N_CLEAN_TARGET / (1 - CORRUPTION_RATE))

VENDORS = [
    "Northgate Supply Co", "Blue Anchor Retail", "Summit Electronics",
    "Prairie Grocer", "Vantage Travel", "Cascade Software Inc",
    "Golden Aspen Jewelers", "Redline Auto Parts", "Harbor Utilities",
    "Meridian Apparel", "Ember Restaurant Group", "Quantum Crypto Exchange",
    "Fairview Pharmacy", "Ironwood Hardware", "Lakeside Bistro",
    "Sterling Gambling Co", "Nimbus Cloud Services", "Copperfield Books",
    "Alpine Outfitters", "Delta Freight Logistics",
]

CATEGORIES = [
    "electronics", "grocery", "travel", "software", "gambling",
    "jewelry", "utilities", "restaurant", "apparel", "crypto_exchange",
]
# (mean, std) transaction amount per category.
CATEGORY_AMOUNT_PARAMS = {
    "electronics": (220, 180), "grocery": (60, 40), "travel": (450, 300),
    "software": (90, 70), "gambling": (300, 400), "jewelry": (650, 500),
    "utilities": (120, 60), "restaurant": (45, 30), "apparel": (85, 60),
    "crypto_exchange": (800, 900),
}
HIGH_RISK_CATEGORIES = {"gambling", "crypto_exchange", "jewelry"}

CURRENCIES = ["USD", "CAD", "EUR", "GBP", "AUD"]
PAYMENT_METHODS = ["credit_card", "debit_card", "ach", "wire", "digital_wallet"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "NG", "RU", "CN", "BR"]
RISKY_COUNTRIES = {"NG", "RU"}

TEMPLATES = ["template_a", "template_b"]

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)
DATE_SPAN = (END_DATE - START_DATE).days

P_FRAUD = 0.04


def random_account_number() -> str:
    length = random.randint(10, 16)
    return "".join(random.choice("0123456789") for _ in range(length))


def sample_confidence(low_quality: bool) -> float:
    if low_quality:
        return float(np.clip(np.random.normal(0.55, 0.18), 0.05, 0.99))
    return float(np.clip(np.random.normal(0.94, 0.06), 0.3, 0.999))


def generate_row(row_id: int) -> dict:
    # Draw the label first, then sample features from label-conditional
    # distributions so fraud tells are correlated rather than independent.
    is_fraud = int(np.random.random() < P_FRAUD)

    if is_fraud:
        category = random.choices(
            CATEGORIES, weights=[6, 4, 6, 5, 16, 16, 4, 5, 6, 16],
        )[0]
    else:
        category = random.choices(
            CATEGORIES, weights=[11, 14, 11, 11, 3, 3, 10, 13, 11, 3],
        )[0]
    mean, std = CATEGORY_AMOUNT_PARAMS[category]

    if is_fraud:
        account_age_days = int(np.clip(np.random.exponential(22), 0, 4000))
        documents_last_24h = int(np.clip(np.random.poisson(5.5), 0, 40))
        hour_of_day = (
            int(np.clip(np.random.normal(2, 1.6), 0, 23)) if random.random() < 0.65
            else int(np.clip(np.random.normal(14, 5), 0, 23))
        )
        country_mismatch = random.random() < 0.55
        amount_mult = float(np.clip(np.random.lognormal(mean=0.35, sigma=0.6), 0.3, 8))
        payment_method = random.choices(PAYMENT_METHODS, weights=[18, 12, 8, 30, 32])[0]
        low_quality_fields = random.random() < 0.72
        billing_country = random.choices(COUNTRIES, weights=[14, 12, 8, 7, 7, 6, 16, 16, 7, 7])[0]
    else:
        account_age_days = int(np.clip(np.random.exponential(430), 0, 4000))
        documents_last_24h = int(np.clip(np.random.poisson(1.0), 0, 40))
        hour_of_day = int(np.clip(np.random.normal(14, 4.5), 0, 23))
        country_mismatch = random.random() < 0.025
        amount_mult = float(np.clip(np.random.lognormal(mean=0.0, sigma=0.35), 0.3, 3))
        payment_method = random.choices(PAYMENT_METHODS, weights=[46, 27, 13, 4, 10])[0]
        low_quality_fields = random.random() < 0.05
        billing_country = random.choices(COUNTRIES, weights=[31, 26, 11, 9, 9, 8, 2, 2, 1, 1])[0]

    shipping_country = billing_country if not country_mismatch else random.choice(COUNTRIES)
    amount = max(1.0, np.random.normal(mean, std) * amount_mult)
    tx_date = START_DATE + timedelta(days=random.randint(0, DATE_SPAN))

    return {
        "document_id": f"DOC{row_id:09d}",
        "template": random.choice(TEMPLATES),
        "vendor_name": {"value": random.choice(VENDORS), "confidence": sample_confidence(False)},
        "account_number": {"value": random_account_number(), "confidence": sample_confidence(low_quality_fields)},
        "transaction_date": {"value": tx_date.isoformat(), "confidence": sample_confidence(False)},
        "amount": {"value": round(amount, 2), "confidence": sample_confidence(low_quality_fields)},
        "currency": {"value": random.choice(CURRENCIES), "confidence": sample_confidence(False)},
        "merchant_category": {"value": category, "confidence": sample_confidence(False)},
        "payment_method": {"value": payment_method, "confidence": sample_confidence(False)},
        "billing_country": {"value": billing_country, "confidence": sample_confidence(False)},
        "shipping_country": {"value": shipping_country, "confidence": sample_confidence(low_quality_fields)},
        "account_age_days": account_age_days,
        "documents_last_24h": documents_last_24h,
        "hour_of_day": hour_of_day,
        "is_fraud": is_fraud,
    }


def corrupt_row(row: dict) -> dict:
    choice = random.randint(0, 4)
    if choice == 0:
        row["amount"]["value"] = -abs(row["amount"]["value"])
    elif choice == 1:
        row["currency"]["value"] = "ZZZ"
    elif choice == 2:
        row["account_number"]["value"] = "12"
    elif choice == 3:
        row["merchant_category"]["value"] = "unknown_category_x"
    else:
        row["amount"]["confidence"] = 1.7
    return row


def bbox_for_template(template: str) -> dict:
    if template == "template_a":
        layout = {
            "vendor_name": (0.05, 0.05, 0.40, 0.06),
            "merchant_category": (0.05, 0.13, 0.40, 0.05),
            "billing_country": (0.05, 0.42, 0.40, 0.05),
            "shipping_country": (0.55, 0.42, 0.40, 0.05),
            "currency": (0.05, 0.75, 0.15, 0.05),
            "amount": (0.55, 0.72, 0.40, 0.08),
            "payment_method": (0.05, 0.85, 0.40, 0.05),
            "account_number": (0.55, 0.05, 0.40, 0.06),
            "transaction_date": (0.55, 0.13, 0.40, 0.05),
        }
    else:
        layout = {
            "vendor_name": (0.08, 0.04, 0.38, 0.07),
            "merchant_category": (0.08, 0.15, 0.35, 0.05),
            "billing_country": (0.08, 0.46, 0.38, 0.05),
            "shipping_country": (0.50, 0.46, 0.38, 0.05),
            "currency": (0.08, 0.70, 0.15, 0.05),
            "amount": (0.50, 0.68, 0.40, 0.09),
            "payment_method": (0.08, 0.82, 0.38, 0.05),
            "account_number": (0.50, 0.04, 0.42, 0.07),
            "transaction_date": (0.50, 0.15, 0.38, 0.05),
        }
    return {k: {"x": v[0], "y": v[1], "w": v[2], "h": v[3]} for k, v in layout.items()}


def main():
    out_path = os.environ.get("RAW_PATH", "data/raw_extracted.jsonl")
    n_corrupt = int(N_RAW * CORRUPTION_RATE)
    corrupt_indices = set(random.sample(range(N_RAW), n_corrupt))

    with open(out_path, "w") as f:
        for i in range(N_RAW):
            row = generate_row(i)
            row["bounding_boxes"] = bbox_for_template(row["template"])
            if i in corrupt_indices:
                row = corrupt_row(row)
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {N_RAW} raw rows ({n_corrupt} malformed) to {out_path}")


if __name__ == "__main__":
    main()
