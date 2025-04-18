from pypinyin import pinyin, lazy_pinyin, Style
from pinyin_to_ipa import pinyin_to_ipa
from itertools import chain
import json
from hanziconv import HanziConv

PINYIN_TONE = ["˧˩˧","˧˥","˥˩", "˥"]
CANTO_TONE = [ "˦˥", "˨˩", "˨˧", "˧", "˨", "˥"]
counter = [0]

CJK = list(chain(range(0x4E00, 0xA000), range(0x3400, 0x4DC0)))  # 27k chars
char2pinyin = {}
char2cantonese = {}

def load_ipa_data(file_path):
    ipa_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('{') or line.endswith('}'):
                continue  # 跳过空行和 JSON 的开始/结束括号
            try:
                key, value = line.split(':', 1)
                key = key.strip().strip('"')
                value = value.strip().strip('"').rstrip(',')  # 移除尾部的逗号
                # 如果有多个发音,只取第一个，并去除所有引号
                ipa_dict[key] = value.split(',')[0].strip().replace('"', '')
            except ValueError:
                print(f"警告: 第 {line_number} 行格式不正确: {line}")
    
    if not ipa_dict:
        raise ValueError("无法从文件中解析出任何有效的IPA数据")
    
    return ipa_dict

def cantonese_to_ipa(text, ipa_dict):
    result = []
    for char in text:
        if char in ipa_dict:
            result.append(ipa_dict[char])
        else:
            result.append(char)  # 如果字符不在字典中,保持原样
    return ' '.join(result)

def add_char_to_pinyin_dict(char, char2pinyin):
    pinyin = lazy_pinyin(chr(char), style=Style.TONE3)[0]
    if pinyin == chr(char):
        char2pinyin[chr(char)] = (None, None, None)
        return None
    # print(pinyin)
    ipa = pinyin_to_ipa(pinyin)[0]
    if len(ipa) == 3:
        # print(ipa)
        initial, mid, final = ipa
        for tone in PINYIN_TONE:
            if tone in mid:
                fianl = mid.replace(tone, "") + final
                char2pinyin[chr(char)] = (initial, fianl, tone)
                break
    elif len(ipa) == 2:
        for tone in PINYIN_TONE:
            if tone in ipa[0]: # no intial, (mid, final)
                char2pinyin[chr(char)] = (None, ipa[0].replace(tone, "") + ipa[1], tone)
                break
            else: # no final, (initial, mid)
                char2pinyin[chr(char)] = (ipa[0], ipa[1].replace(tone, ""), tone)
                break
    else: # only mid
        for tone in PINYIN_TONE:
            if tone in ipa[0]:
                char2pinyin[chr(char)] = (None, ipa[0].replace(tone, ""), tone)
                break

def add_char_to_cantonese_dict(char, ipa_dict, char2cantonese):
    yue_ipa = cantonese_to_ipa(chr(char), ipa_dict)
    if yue_ipa == None or yue_ipa == chr(char):
        yue_ipa = cantonese_to_ipa(HanziConv.toTraditional(chr(char)), ipa_dict)
        if yue_ipa == None or yue_ipa == chr(char):
            char2cantonese[chr(char)] = (None, None)
            counter[0] += 1
            return None
    for tone in CANTO_TONE:
        if tone in yue_ipa:
            ipa = yue_ipa.replace(tone, "")
            char2cantonese[chr(char)] = (ipa, tone)


if __name__ == "__main__":
    # Load IPA data
    try:
        ipa_dict = load_ipa_data('ipa_data.txt')
        print(f"成功加载了 {len(ipa_dict)} 个IPA转换规则")
    except Exception as e:
        print(f"加载IPA数据时出错: {e}")
        exit(1)

    for char in CJK:
        # Mandarin
        # add_char_to_pinyin_dict(char, char2pinyin)
        # Cantonese
        add_char_to_cantonese_dict(char, ipa_dict, char2cantonese)
    
    # Save the mapping to a file
    # with open("mandarin.json", "w", encoding="utf-8") as f:
    #     json.dump(char2pinyin, f, ensure_ascii=False, indent=2)
    print(counter[0])
    with open("cantonese.json", "w", encoding="utf-8") as f:
        json.dump(char2cantonese, f, ensure_ascii=False, indent=2)