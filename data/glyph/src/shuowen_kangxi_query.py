import requests
import time, random, json, os
from bs4 import BeautifulSoup
from itertools import chain
from concurrent.futures import ThreadPoolExecutor

CJK = list(chain(range(0x4E00, 0xA000), range(0x3400, 0x4DC0)))  # 27k chars
CHECKPOINT_FILE = "seal_tradi_radical.json"
SAVE_EVERY = 1000
THREADS = 5

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_radicals(codepoint):
    def fetch(url, entry):
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            div = soup.find("div", class_="zi-attrs-list")
            if not div: return None
            for p in div.find_all("p"):
                label = p.find("label")
                if label and label.text.strip() == entry:
                    return p.find("a").text.strip()
        except Exception as e:
            return None

    url1 = f"https://www.hanyuguoxue.com/shuowen/zi-{codepoint}"
    url2 = f"https://www.hanyuguoxue.com/kangxi/zi-{codepoint}"
    seal = fetch(url1, "部首")
    trad = fetch(url2, "康熙部首")
    time.sleep(random.uniform(0.8, 1.5))  # avoid rate-limiting
    return codepoint, (seal, trad)

def main():
    existing = load_checkpoint()
    done = set(map(int, existing.keys()))

    to_scrape = [cp for cp in CJK if cp not in done]
    print(f"Characters left to scrape: {len(to_scrape)}")

    total_scraped = 0
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for codepoint, radicals in executor.map(get_radicals, to_scrape):
            existing[str(codepoint)] = radicals
            total_scraped += 1

            # Save checkpoint every N new entries
            if total_scraped % SAVE_EVERY == 0:
                save_checkpoint(existing)
                print(f"✅ Saved checkpoint at {total_scraped} entries")

    # Final save
    save_checkpoint(existing)
    print("🎉 All done!")

if __name__ == "__main__":
    main()
