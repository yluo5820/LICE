import requests
import urllib.parse
from bs4 import BeautifulSoup
import time, random, json, os, re
from itertools import chain
from concurrent.futures import ThreadPoolExecutor

MIDDLE_CHINESE = "http://ccdc.fudan.edu.cn/linguae/ltcPhonology/"
OLD_CHINESE = "http://ccdc.fudan.edu.cn/linguae/ochPhonology/"
CJK = list(chain(range(0x4E00, 0xA000), range(0x3400, 0x4DC0)))  # 27k chars
CHECKPOINT_FILE = "mid_and_old_C_checkpoint.json"
SAVE_EVERY = 1000
THREADS = 5

"""
Return the html string for the query of x
"""
def get_ltc_phonology_representation(x, is_query_middle):
    base_url = MIDDLE_CHINESE if is_query_middle else OLD_CHINESE
    encoded_x = urllib.parse.quote(chr(x))
    url = f"{base_url}?representationText={encoded_x}"
    
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code}"


def extract_phonetic_info(data):
    phonetic_info = []
    for entry in data.get("data", []):  # Assuming phonetic data is under "data" key
        character = entry.get("zi", "")
        fanqie = entry.get("fanqie", "")
        phonetic = entry.get("phonetic", {})  # Nested phonetic details
        phonetic_info.append({
            "character": character,
            "fanqie": fanqie,
            "initial": phonetic.get("initial", ""),
            "final": phonetic.get("final", ""),
            "tone": phonetic.get("tone", ""),
        })
    return phonetic_info

def get_pronunciation(x, is_query_middle):
    try:
        result = get_ltc_phonology_representation(x, is_query_middle)
        soup = BeautifulSoup(result, "html.parser")

        match = re.search(r"var ltc = (\{.*\});" if is_query_middle else r"var ochs = (\[.*?\]);", soup.script.text , re.DOTALL)

        js_object = match.group(1)

        # Convert JavaScript to JSON format (replace JS-style None/False/True)
        json_string = (
            js_object.replace("None", "null")
            .replace("False", "false")
            .replace("True", "true")
            .replace("'", "\"")
        )
        # Convert to Python dictionary
        ltc_dict = json.loads(json_string)["ltcPhones"][0] if is_query_middle else json.loads(json_string)[0]
        return ltc_dict["pronunciationText"]
    except Exception as e:
        return None

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_pronounciations(codepoint):
    middle = get_pronunciation(codepoint, True)
    old = get_pronunciation(codepoint, False)
    time.sleep(random.uniform(0.8, 1.5))  # avoid rate-limiting
    return codepoint, (middle, old)

def main():
    existing = load_checkpoint()
    done = set(map(int, existing.keys()))

    to_scrape = [cp for cp in CJK if cp not in done]
    print(f"Characters left to scrape: {len(to_scrape)}")

    total_scraped = 0
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for codepoint, radicals in executor.map(get_pronounciations, to_scrape):
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
