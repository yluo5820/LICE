import requests
import urllib.parse
from bs4 import BeautifulSoup
import json
import re

MIDDLE_CHINESE = "http://ccdc.fudan.edu.cn/linguae/ltcPhonology/"
OLD_CHINESE = "http://ccdc.fudan.edu.cn/linguae/ochPhonology/"
QUERY_MIDDLE = False

def get_ltc_phonology_representation(x):
    base_url = MIDDLE_CHINESE if QUERY_MIDDLE else OLD_CHINESE
    encoded_x = urllib.parse.quote(x)
    url = f"{base_url}?representationText={encoded_x}"
    
    response = requests.get(url)
    
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

if __name__ == "__main__":
    x = "卜"
    result = get_ltc_phonology_representation(x)
    soup = BeautifulSoup(result, "html.parser")

    match = re.search(r"var ltc = (\{.*\});" if QUERY_MIDDLE else r"var ochs = (\[.*?\]);", soup.script.text , re.DOTALL)

    js_object = match.group(1)

    # Convert JavaScript to JSON format (replace JS-style None/False/True)
    json_string = (
        js_object.replace("None", "null")
        .replace("False", "false")
        .replace("True", "true")
        .replace("'", "\"")
    )
    # Convert to Python dictionary
    ltc_dict = json.loads(json_string)["ltcPhones"][0] if QUERY_MIDDLE else json.loads(json_string)[0]
    print("Pronounciation: ",ltc_dict["pronunciationText"])  # Now ltc_dict is a usable Python dictionary
    print(json.dumps(ltc_dict, indent=4, ensure_ascii=False))
