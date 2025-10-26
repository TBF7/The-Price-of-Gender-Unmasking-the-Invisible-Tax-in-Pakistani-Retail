# imtiaz_manual_parser.py
from bs4 import BeautifulSoup
import pandas as pd, re, glob

def extract_price(soup_block):
    """
    Extract numeric price from any nested tags using a more rigorous approach.
    
    It handles:
    1. Removes all non-digit/non-separator characters.
    2. Removes commas (assuming they are thousands separators).
    3. Converts the resulting string to a float.
    4. Applies the original correction logic (multiplying values < 10 by 1000).
    """
    if not soup_block:
        return None
    
    # 1. Join all text from the soup block
    text = " ".join(soup_block.stripped_strings)
    
    # 2. Clean the text: keep only digits, period, and comma (the potential number components)
    # Example: "Rs 1,250.50" becomes "1,250.50"
    text = re.sub(r"[^\d.,]", "", text)
    
    # 3. **CRUCIAL IMPROVEMENT:** Remove commas. This handles prices like "1,250.50" 
    # and leaves "1250.50" which is valid for float conversion.
    cleaned_text = text.replace(',', '')
    
    # Defensive check for European format (period as thousands separator, comma as decimal)
    # If the number has no period and exactly one comma, treat the comma as a decimal point.
    if cleaned_text.count('.') == 0 and cleaned_text.count(',') == 1:
        cleaned_text = cleaned_text.replace(',', '.')

    try:
        # 4. Convert to float
        val = float(cleaned_text)
        
        # 5. Apply the per-thousand correction logic (if a price is < 10, assume it's stored 
        # in thousands/a fraction of PKR and correct it).
        if val < 10 and val != 0:
            val *= 1000
        return val
    except:
        # Fails if the string is empty, or still contains multiple decimals/invalid format
        return None

def detect_gender(t):
    if not t: return "unisex"
    t = t.lower()
    if any(k in t for k in ["women","ladies","her","girl","female","خواتین","لیڈیز"]): return "female"
    if any(k in t for k in ["men","gents","his","boy","male","مرد","جینٹس"]): return "male"
    if any(k in t for k in ["unisex","for all","everyone","family"]): return "unisex"
    if any(k in t for k in ["rose","floral","pink","beauty","jasmine","glow","silky","smooth"]): return "female"
    if any(k in t for k in ["sport","cool","fresh","active","ocean","intense","energy"]): return "male"
    return "unisex"

def guess_brand(title):
    if not title: return None
    parts = re.split(r"[\s\-–]+", title)
    if len(parts) > 1 and parts[1][0].isupper():
        return parts[0] + " " + parts[1]
    return parts[0]

def parse_size(title):
    if not title: return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|g|kg|l)\b", title, re.I)
    if m: return m.group(0).lower()
    return None

def normalize_size(size):
    if not size: return None, None
    s = size.lower().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg)", s)
    if not m:
        return None, None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "kg": val, unit = val * 1000, "g"
    if unit == "l": val, unit = val * 1000, "ml"
    return val, unit

rows = []
for file in glob.glob("*.html"):
    cat = file.split(".")[0]
    print(f"🧩 Parsing {file} ({cat}) ...")
    with open(file, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # detect product cards (Imtiaz pages use multiple divs)
    cards = soup.select("div.product-card, div.MuiGrid-item, div.productCard, div.item")
    for c in cards:
        title_el = c.select_one("h3, h2, p, .title, .product-name, .MuiTypography-root")
        title = title_el.get_text(strip=True) if title_el else None
        
        # Select for common price classes
        price_el = c.select_one(".price, .product-price, .MuiTypography-root, [class*='price'], [class*='Price']") 
        price = extract_price(price_el) if price_el else None
        
        img_el = c.select_one("img")
        img = img_el.get("src") if img_el else None

        brand = guess_brand(title)
        size = parse_size(title)
        size_val, size_unit = normalize_size(size)

        rows.append({
            "retailer": "imtiaz",
            "category": cat,
            "title": title,
            "brand": brand,
            "size_raw": size,
            "size_value": size_val,
            "size_unit": size_unit,
            "price_pkr": price,
            "gender_target": detect_gender(title),
            "image_url": img
        })

df = pd.DataFrame(rows).drop_duplicates(subset=["title"]).reset_index(drop=True)
out = f"imtiaz_manual_combined_{len(df)}.csv"
df.to_csv(out, index=False)
print(f"\n✅ Saved {out} ({len(df)} rows)")