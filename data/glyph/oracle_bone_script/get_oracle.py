import requests
import json
import re

pattern = re.compile(r"""
^(
    (?P<paren_only>         （.）              ) |  # (你)
    (?P<with_paren>         (.)（(.)） ) |  # 你(好)
    (?P<slash_form>         (.)/(.) ) |    # 你/好
    (?P<plain>              .                  )     # 你
)$
""", re.VERBOSE)
URL = "https://jgw.aynu.edu.cn/home/zx/method/jgwzx.ashx?type=sortzxbybs&pageindex="

def get_oracle_bone_script():
    list_of_oracles = []
    for page_index in range(1, 21):
        response = requests.get(URL + str(page_index))
        if response.status_code == 200:
            list_of_oracles += response.json()["ALLDZ"]
        else:
            return f"Error: {response.status_code}"
    return list_of_oracles
    
def download_oracles():
    result = get_oracle_bone_script()
    with open("oracle_radicals.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2, ensure_ascii=False))

def load_oracles(file_name="oracle_radicals.json"):
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_file(data, file_name="oracle_radicals.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def remove_empty_entry(data):
    for radical_dict in data:
        radical_dict["JGWZX"] = [char for char in radical_dict["JGWZX"] if char["FTZ"] != '']
    return data

"""
Return a dict with (Chinese character, [label0, label1, ...])
Of size = 1650, counting weird, un  uncommon characters.
"""
def make_dataset(data)->dict:
    counter = 0
    char2radical = {}
    label = 0
    while label < len(data):
        chars_share_radical = data[label]["JGWZX"] #list of dicts
        for char_share_radical in chars_share_radical:
            for char in [char_share_radical["FTZ"], char_share_radical["JTZ"]]:
                if char in char2radical:
                    if label not in char2radical[char]:
                        char2radical[char].append(label)
                else:
                    char2radical[char] = [label]
            counter += 1
        label += 1
    print("counter: ", counter)
    return char2radical


def match_regex(text):
    match = pattern.match(text)
    if not match:
        return None
    elif match.group("paren_only"):
        return match.group("paren_only")[1]  # strip parentheses
    elif match.group("with_paren"):
        x = match.group(3)
        y = match.group(4)
        return x,y
    elif match.group("slash_form"):
        x = match.group(5)
        y = match.group(6)
        return x,y
    elif match.group("plain"):
        return match.group("plain")
    
def match(text):
    if len(text) == 1:
        return text
    elif len(text) == 3 and text[0] =="（":
        return text[1]
    elif len(text) == 3 and text[1] =="/":
        return text[0], text[2]
    elif len(text) == 4:
        return text[0], text[3]

if __name__ == "__main__":
    data = load_oracles("oracle_radicals_no_em.json")
    char2radical = make_dataset(data)
    save_file(char2radical, "oracle_char2radical.json")
    print(char2radical, len(char2radical))