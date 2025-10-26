# daraz_scraper_full_final.py
import asyncio, re, random
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright
import regex as re2

# ---------- helpers ----------
def extract_price(text):
    if not text:
        return None
    t = re.sub(r"[^\d.,]", "", text).replace(",", "")
    try:
        val = float(t)
        if 0.05 < val < 10:  # handle 0.395 → 395
            return val * 1000
        return val
    except:
        return None


def guess_brand(title):
    if not title:
        return None
    tokens = re.split(r'[\s\-–]', title)
    if not tokens:
        return None
    brand_guess = tokens[0]
    if len(tokens) > 1 and tokens[1][0].isupper():
        brand_guess += " " + tokens[1]
    return brand_guess.strip()


def parse_size(title):
    """Extract size/pack info like 200ml, 2x150g, Pack of 3, 3 pcs."""
    if not title:
        return None
    t = title.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(ml|g|kg|l|oz)\b",          # 200ml, 150 g
        r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)(ml|g|kg|l|oz)", # 2x400ml
        r"pack\s*of\s*\d+",                              # pack of 3
        r"(\d+)\s*pcs?"                                  # 3 pcs
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(0)
    return None


def normalize_size(size):
    """Convert extracted size to numeric value + unit for comparison."""
    if not size:
        return None, None
    s = size.lower().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg|oz)", s)
    if not m:
        return None, None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "kg":
        val *= 1000
        unit = "g"
    if unit == "l":
        val *= 1000
        unit = "ml"
    return val, unit


# ---------- gender regex ----------
RX_FEMALE = [
    r"\bfor\s*women\b", r"\bfor\s*her\b", r"\b(women|woman|lad(y|ies))\b",
    r"\bgirls?\b", r"\bwomen['’]s\b", r"\bher\b", r"(خواتین|لیڈیز)"
]
RX_MALE = [
    r"\bfor\s*men\b", r"\bfor\s*him\b", r"\bmen\b", r"\bman\b", r"\bgents?\b",
    r"\bboys?\b", r"\bmen['’]s\b", r"\bhis\b", r"(مرد|جینٹس)"
]
RX_FEMALE_LINE = [
    r"\b(her|women['’]s)\s*(care|series|edition)\b",
    r"\b(rose|floral|blossom|pink|lavender|vanilla|jasmine)\b",
    r"\b(silky|soft|glow|radiance|brighten(?:ing)?|whitening)\b"
]
RX_MALE_LINE = [
    r"\bmen\+?care\b", r"\bfor\s*men\b",
    r"\b(nivea\s*men|garnier\s*men|l’oreal\s*men)\b",
    r"\b(sport|cool|ice|xtreme|power|active|fresh|ocean|energy|steel)\b",
    r"\b(shave|after[-\s]?shave|beard|moustache)\b"
]
RX_UNISEX = [r"\bunisex\b", r"\bfor\s*all\b", r"\bfamily\s*pack\b"]
RX_NEGATE = [
    r"\b(women-?owned|women[-\s]?led)\b",
    r"\b(mens(?:trual)?|period|sanitary|pads?)\b"
]

RX_FEMALE = [re2.compile(p, re2.I) for p in RX_FEMALE]
RX_MALE = [re2.compile(p, re2.I) for p in RX_MALE]
RX_FEMALE_LINE = [re2.compile(p, re2.I) for p in RX_FEMALE_LINE]
RX_MALE_LINE = [re2.compile(p, re2.I) for p in RX_MALE_LINE]
RX_UNISEX = [re2.compile(p, re2.I) for p in RX_UNISEX]
RX_NEGATE = [re2.compile(p, re2.I) for p in RX_NEGATE]


def detect_gender(title: str, brand: str = None) -> str:
    """Regex + brand aware gender detection (no category hints)."""
    if not title:
        return "unisex"
    t = title.lower()

    for rx in RX_NEGATE:
        if rx.search(t):
            return "unisex"

    fem_hits = any(rx.search(t) for rx in RX_FEMALE)
    male_hits = any(rx.search(t) for rx in RX_MALE)
    if fem_hits and not male_hits:
        return "female"
    if male_hits and not fem_hits:
        return "male"
    if fem_hits and male_hits:
        return "unisex"
    if any(rx.search(t) for rx in RX_UNISEX):
        return "unisex"

    score_f = sum(1 for rx in RX_FEMALE_LINE if rx.search(t))
    score_m = sum(1 for rx in RX_MALE_LINE if rx.search(t))

    if brand:
        b = brand.lower()
        if "men" in b and "women" not in b:
            score_m += 1
        if "women" in b or "lady" in b or "ladies" in b:
            score_f += 1
        if "men+care" in b or "men care" in b:
            score_m += 2

    if score_f >= score_m + 2 and score_f >= 2:
        return "female"
    if score_m >= score_f + 2 and score_m >= 2:
        return "male"
    return "unisex"


# ---------- pagination ----------
def paginate_url(base_url, page_num):
    if "page=" in base_url:
        return re.sub(r"page=\d+", f"page={page_num}", base_url)
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}page={page_num}"


# ---------- main async scraper ----------
async def scrape_daraz_category(category="deodorant", max_pages=2, delay_s=(1.8, 3.5)):
    base_url = f"https://www.daraz.pk/catalog/?q={category}"
    rows = []
    observed_at = datetime.now().strftime("%Y-%m-%d")

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        for i in range(1, max_pages + 1):
            url = paginate_url(base_url, i)
            print(f"[daraz] page {i}: {url}")
            await page.goto(url, timeout=70000)
            await page.wait_for_timeout(2500)
            cards = await page.query_selector_all('[data-qa-locator="product-item"]')
            print("  found cards:", len(cards))

            for c in cards:
                try:
                    title_el = await c.query_selector('a[title], a[href] .title--wFj93')
                    title = None
                    if title_el:
                        title = await title_el.get_attribute("title")
                        if not title:
                            title = (await title_el.inner_text()).strip()

                    price_el = await c.query_selector(".ooOxS, .currency--GVKjl, .price--NVB62, .price--K8tGk")
                    price = extract_price(await price_el.inner_text()) if price_el else None

                    link_el = await c.query_selector("a[href]")
                    href = await link_el.get_attribute("href") if link_el else None
                    if href and href.startswith("/"):
                        href = "https://www.daraz.pk" + href

                    img_el = await c.query_selector("img")
                    img = await img_el.get_attribute("src") if img_el else None

                    brand = guess_brand(title)
                    size = parse_size(title)
                    size_value, size_unit = normalize_size(size)
                    gender = detect_gender(title, brand=brand)

                    rows.append({
                        "retailer": "daraz",
                        "category": category,
                        "title": title,
                        "brand": brand,
                        "size_raw": size,
                        "size_value": size_value,
                        "size_unit": size_unit,
                        "gender_target": gender,
                        "price_pkr": price,
                        "product_url": href,
                        "image_url": img,
                        "observed_at": observed_at
                    })
                except Exception:
                    continue

            await asyncio.sleep(random.uniform(*delay_s))
        await browser.close()

    df = pd.DataFrame(rows).drop_duplicates(subset=["product_url"]).reset_index(drop=True)
    return df


# ---------- orchestrator ----------
async def main():
    CATEGORIES = [
        "deodorant", "shampoo", "conditioner", "facewash", "cream", "lotion",
        "body-spray", "anti-perspirant", "razor", "soap", "handwash"
    ]
    PAGES_PER_CATEGORY = 11
    all_dfs = []

    for cat in CATEGORIES:
        print(f"\n=== scraping category: {cat} ===")
        try:
            df = await scrape_daraz_category(category=cat, max_pages=PAGES_PER_CATEGORY)
            out_path = f"daraz_{cat}_{len(df)}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(df)} rows)")
            all_dfs.append(df)
        except Exception as e:
            print("Error on", cat, e)

    big = pd.concat(all_dfs, ignore_index=True)
    big_path = f"daraz_all_{len(big)}.csv"
    big.to_csv(big_path, index=False)
    print(f"\nAll done ✅ Combined dataset saved: {big_path} ({len(big)} rows)")

if __name__ == "__main__":
    asyncio.run(main())
