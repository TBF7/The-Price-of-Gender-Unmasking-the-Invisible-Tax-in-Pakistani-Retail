from bs4 import BeautifulSoup
import pandas as pd, re, glob

# ---------- helpers ----------

def extract_price(soup_block):
    """
    Extracts Rs/PKR prices from any nested text like:
    - "Rs. 595", "PKR 1,099", "Rs 0.595", "Rs 595.00", "Rs  1,249"
    - Removes ranges like "From Rs." or "To Rs."
    """
    if not soup_block:
        return None

    # combine all visible text in the price element
    text = " ".join(soup_block.stripped_strings)
    text = text.replace("\xa0", " ")  # remove non-breaking spaces
    text = re.sub(r"(From|To)\s+", "", text, flags=re.I)

    # find all numeric patterns that follow Rs/PKR or standalone numbers
    match = re.search(r"(?:Rs\.?|PKR)?\s*([\d,]+(?:\.\d+)?)", text, re.I)
    if not match:
        # fallback: search within nested children
        inner = " ".join([s for s in soup_block.stripped_strings])
        match = re.search(r"(?:Rs\.?|PKR)?\s*([\d,]+(?:\.\d+)?)", inner, re.I)

    if match:
        val_str = match.group(1).replace(",", "")
        try:
            val = float(val_str)
            # if weird decimals like 0.395 → convert to Rs 395
            if val < 10:
                val *= 1000
            return round(val, 2)
        except:
            return None
    return None


def detect_gender(title):
    """
    Enhanced gender detection:
    - catches explicit male/female terms
    - includes Urdu equivalents
    - identifies marketing words ("Men+Care", "Glow", "Cool", etc.)
    - defaults unmarked products to 'female' (market bias assumption)
    """
    if not title:
        return "female"  # assume female default if unlabeled

    t = title.lower()

    # explicit female indicators
    female_kw = [
        "women", "ladies", "lady", "her", "girl", "female",
        "for her", "she", "خواتین", "لیڈیز", "feminine"
    ]
    # explicit male indicators
    male_kw = [
        "men", "man", "gents", "gentlemen", "his", "boy", "male",
        "for him", "him", "مرد", "جینٹس", "men+care"
    ]

    # marketing cues
    female_cues = [
        "rose", "floral", "pink", "beauty", "jasmine", "glow",
        "silky", "smooth", "radiance", "moisture", "soft", "brightening"
    ]
    male_cues = [
        "sport", "cool", "fresh", "active", "energy", "ocean",
        "intense", "power", "xtreme", "deep clean", "musky"
    ]

    if any(k in t for k in male_kw + male_cues):
        return "male"
    if any(k in t for k in female_kw + female_cues):
        return "female"
    if "unisex" in t or "for all" in t or "everyone" in t or "men/women" in t or "Men/Women" in t or "women/men" in t or "Women/Men" in t:
        return "unisex"

    # default bias toward female market
    return "female"


def guess_brand(title):
    if not title:
        return None
    parts = re.split(r"[\s\-–]+", title)
    if len(parts) > 1 and parts[1][0].isupper():
        return parts[0] + " " + parts[1]
    return parts[0]


def parse_size(title):
    if not title:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|g|kg|l|oz)\b", title, re.I)
    if m:
        return m.group(0).lower()
    return None


# ---------- main parser ----------

rows = []
for file in glob.glob("*.html"):
    cat = file.split(".")[0]
    print(f"🧩 Parsing {file} ({cat}) ...")
    with open(file, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # try broad set of selectors
    cards = soup.select("div.ProductCard, div.product-card, li.product-item, div.MuiGrid-item, div[class*='product']")
    print(f"  found possible cards: {len(cards)}")

    for c in cards:
        title_el = c.select_one("h3, h2, .product-title, .title, p, span")
        title = title_el.get_text(strip=True) if title_el else None

        price_el = c.select_one(".price, .product-price, .currency, .MuiTypography-root, p, span, div")
        price = extract_price(price_el) if price_el else None

        img_el = c.select_one("img")
        img = img_el.get("src") if img_el else None

        rows.append({
            "retailer": "metro",
            "category": cat,
            "title": title,
            "brand": guess_brand(title),
            "size": parse_size(title),
            "price_pkr": price,
            "gender_target": detect_gender(title),
            "image_url": img
        })

# ---------- output ----------

df = pd.DataFrame(rows).drop_duplicates(subset=["title"]).reset_index(drop=True)
out = f"metro_manual_combined_{len(df)}.csv"
df.to_csv(out, index=False)
print(f"\n✅ Saved {out} ({len(df)} rows)")

