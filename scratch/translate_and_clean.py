import re
import json
import time
from pathlib import Path
from deep_translator import GoogleTranslator

CHINESE_REPLACEMENTS = [
    ("黑棋一户", "Kurosaki Ichigo"),
    ("一户", "Ichigo"),
    ("有哈巴赫", "Yhwach"),
    ("有哈", "Yhwach"),
    ("哈斯沃德", "Haschwalth"),
    ("哈斯沃人", "Haschwalth"),
    ("石田", "Ishida"),
    ("完圣体", "Vollständig"),
    ("四丰苑两姐弟", "chị em Shihoin"),
    ("四丰苑", "Shihoin"),
    ("亚斯金", "Askin"),
    ("亚金斯", "Askin"),
    ("叶一", "Yoruichi"),
    ("店长", "Urahara Kisuke"),
    ("穆建巴", "Zaraki Kenpachi"),
    ("兵主部一兵位", "Hyosube Ichibei"),
    ("兵主部一兵卫", "Hyosube Ichibei"),
    ("蓝然总右界", "Aizen Sosuke"),
    ("普援洗柱", "Urahara Kisuke"),
    ("雷神战刑", "Shunkou: Raijin Senkei"),
    ("顺龙黑猫战机", "Shunkou: Juuki Kokuyou Senki"),
    ("黑猫战机", "Juuki Kokuyou Senki"),
    ("摩叙罗", "Mahoraga"),
    ("祸静谭", "Sōkoku-tan"),
    ("霍静谭", "Sōkoku-tan"),
    ("霍静弹", "Sōkoku-tan"),
    ("万总观音开红机管", "Kannonbiraki Benihime Aratame"),
    ("洪基奈因到梅边", "Bankai của Urahara"),
    ("捕援喜助", "Urahara Kisuke"),
    ("蒙毒领域", "Gift Bereich"),
    ("极之足球", "Gift Ball Deluxe"),
    ("极致足球", "Gift Ball Deluxe"),
    ("无知要散", "Gift Bereich"),
    ("洪基", "Benihime"),
    ("斩获道", "Zanpakuto"),
    ("斩破刀", "Zanpakuto"),
    ("乱境", "Bankai"),
    ("葛利姆乔", "Grimmjow"),
    ("三天杰顿", "Santen Kesshun"),
    ("只鸡", "Orihime"),
    ("吃鸡", "Orihime"),
    ("黑棋真校", "Masaki"),
    ("林子胡刃", "Linh tử nhận"),
    ("被逼腿", "bị đẩy lùi"),
    ("灭鹊师", "Quincy"),
    ("灭却师", "Quincy"),
    ("井刀身", "thân kiếm"),
    ("续班", "Hư Bạch Hollow"),
    ("王旭牛头", "Vasto Lorde Ichigo"),
    ("吴越形态", "Mugetsu"),
    ("两枚乌王月", "Oetsu Nimaiya"),
    ("二枚屋王悦", "Oetsu Nimaiya"),
    ("致死量", "Tử Vong Lượng"),
]

VIETNAMESE_REPLACEMENTS = [
    # Death/Killing
    (r'\bgiết chết\b', "hạ gục"),
    (r'\bgiết\b', "hạ gục"),
    (r'\bsát hại\b', "tiêu diệt"),
    (r'\bchết\b', "ra đi"),
    (r'\bbị giết\b', "bị hạ gục"),
    (r'\btử vong\b', "nằm xuống"),
    (r'\bmất mạng\b', "bay màu"),
    (r'\bhy sinh\b', "ngã xuống"),
    (r'\bhuyết chiến\b', "đại chiến"),
    # Blood
    (r'\bmáu\b', "sắc đỏ"),
    (r'\bvết máu\b', "sắc đỏ"),
    (r'\bvũng máu\b', "sắc đỏ"),
    (r'\bđổ máu\b', "chấn thương"),
    (r'\bxác chết\b', "thân xác"),
    (r'\bxác\b', "thân xác"),
    (r'\bthi thể\b', "thân xác"),
    # Violence/Weapons
    (r'\bchém chết\b', "hạ gục"),
    (r'\bchém\b', "tác động"),
    (r'\bđâm chết\b', "tiêu diệt"),
    (r'\bđâm\b', "tấn công"),
    # Other sensitive terms
    (r'\bbạo lực\b', "kịch tính"),
]

def preprocess_chinese(text):
    for src, dst in CHINESE_REPLACEMENTS:
        text = text.replace(src, dst)
    return text

def postprocess_vietnamese(text):
    for pattern, repl in VIETNAMESE_REPLACEMENTS:
        # Case insensitive replacement while preserving word boundaries
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

def main():
    translator = GoogleTranslator(source='zh-CN', target='vi')
    srt_dir = Path(r'c:\Users\Public\doanh\code\tool\dichvideo\srt-bleach')
    
    for json_path in srt_dir.glob('*.json'):
        if json_path.name.endswith('_translated.json'):
            continue
            
        print(f"Processing {json_path.name}...", flush=True)
        blocks = json.loads(json_path.read_text(encoding='utf-8'))
        
        translated_srt_lines = []
        for i, block in enumerate(blocks):
            orig_text = block['text']
            preprocessed_text = preprocess_chinese(orig_text)
            
            if preprocessed_text:
                try:
                    translated = translator.translate(preprocessed_text)
                except Exception as e:
                    print(f"Error at block {i}: {e}. Retrying after 1s...", flush=True)
                    time.sleep(1)
                    translated = translator.translate(preprocessed_text)
            else:
                translated = ''
            
            clean_translated = postprocess_vietnamese(translated)
            
            # Reconstruct SRT block
            translated_srt_lines.append(f"{block['index']}")
            translated_srt_lines.append(f"{block['time']}")
            translated_srt_lines.append(f"{clean_translated}\n")
            
            if (i + 1) % 10 == 0:
                print(f"  Translated {i + 1}/{len(blocks)} blocks...", flush=True)
                
        # Write translated SRT
        out_srt_name = json_path.stem + "_vi.srt"
        out_srt_path = srt_dir / out_srt_name
        out_srt_path.write_text('\n'.join(translated_srt_lines), encoding='utf-8')
        print(f"Finished writing translated SRT to {out_srt_path.name}\n", flush=True)

if __name__ == '__main__':
    main()
