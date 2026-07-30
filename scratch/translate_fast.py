import re
import json
import time
from pathlib import Path
from deep_translator import GoogleTranslator

DICTIONARY = [
    # Special moves and concepts
    ("万总观音开红机管", "Bankai: Kannonbiraki Benihime Aratame"),
    ("洪基奈因到梅边", "Bankai: Kannonbiraki Benihime Aratame"),
    ("雷神战刑", "Shunkou: Raijin Senkei"),
    ("顺龙黑猫战机形态", "dạng Thú Hách Miêu Cơ (Shunkou)"),
    ("顺龙黑猫战机", "dạng Thú Hách Miêu Cơ (Shunkou)"),
    ("黑猫战机形态", "dạng Thú Hách Miêu Cơ (Shunkou)"),
    ("黑猫战机", "dạng Thú Hách Miêu Cơ (Shunkou)"),
    ("半牛头蓄化形态", "dạng Bán Hóa Hư Nhất Sừng"),
    ("半牛头虚化", "dạng Bán Hóa Hư Nhất Sừng"),
    ("假面形态王旭牛头", "dạng Vasto Lorde Ichigo"),
    ("王旭牛头", "Vasto Lorde Ichigo"),
    ("王虚牛头", "Vasto Lorde Ichigo"),
    ("吴越形态", "Mugetsu"),
    ("无月形态", "Mugetsu"),
    ("无月", "Mugetsu"),
    ("月牙十字虫", "Getsuga Jujisho"),
    ("月牙十字冲", "Getsuga Jujisho"),
    ("三天杰顿", "Santen Kesshun"),
    ("三天结盾", "Santen Kesshun"),
    ("全知全能的权补", "năng lực Toàn Tri Toàn Năng (Almighty)"),
    ("全知全能", "Toàn Tri Toàn Năng (Almighty)"),
    ("完圣体", "dạng Vollständig"),
    ("蒙毒领域", "vùng độc Gift Bereich"),
    ("极之足球", "Cực Độc Cầu (Gift Ball Deluxe)"),
    ("极致足球", "Cực Độc Cầu (Gift Ball Deluxe)"),
    ("极致毒球", "Cực Độc Cầu (Gift Ball Deluxe)"),
    ("毒球", "Cực Độc Cầu"),
    ("无知要散", "vùng độc Gift Bereich"),
    ("斩获道", "Zanpakuto"),
    ("斩破刀", "Zanpakuto"),
    ("斩魄刀", "Zanpakuto"),
    ("使解", "Shikai"),
    ("乱境", "Bankai"),
    ("卍解", "Bankai"),
    
    # Characters
    ("黑棋一户", "Kurosaki Ichigo"),
    ("黑崎一护", "Kurosaki Ichigo"),
    ("一户", "Ichigo"),
    ("有哈巴赫", "Yhwach"),
    ("有哈", "Yhwach"),
    ("哈斯沃德", "Haschwalth"),
    ("哈斯沃人", "Haschwalth"),
    ("石田", "Ishida Uryuu"),
    ("四丰苑两姐弟", "hai chị em nhà Shihoin"),
    ("四枫院两姐弟", "hai chị em nhà Shihoin"),
    ("四枫院", "Shihoin"),
    ("四丰苑", "Shihoin"),
    ("亚斯金", "Askin Nakk Le Vaar"),
    ("亚金斯", "Askin Nakk Le Vaar"),
    ("葛利姆乔", "Grimmjow"),
    ("叶一", "Yoruichi Shihoin"),
    ("夜一", "Yoruichi Shihoin"),
    ("店长", "Urahara Kisuke"),
    ("穆建巴", "Zaraki Kenpachi"),
    ("更木剑八", "Zaraki Kenpachi"),
    ("兵主部一兵位", "Hyosube Ichibei (Đội trưởng Đội 0)"),
    ("兵主部一兵卫", "Hyosube Ichibei (Đội trưởng Đội 0)"),
    ("蓝然总右界", "Aizen Sosuke"),
    ("蓝染惣右介", "Aizen Sosuke"),
    ("蓝染", "Aizen Sosuke"),
    ("普援洗柱", "Urahara Kisuke"),
    ("捕援喜助", "Urahara Kisuke"),
    ("浦原喜助", "Urahara Kisuke"),
    ("只鸡", "Inoue Orihime"),
    ("织姬", "Inoue Orihime"),
    ("吃鸡", "Inoue Orihime"),
    ("黑棋真校", "Masaki Kurosaki"),
    ("黑崎真咲", "Masaki Kurosaki"),
    ("两枚乌王月", "Oetsu Nimaiya"),
    ("二枚屋王悦", "Oetsu Nimaiya"),
    ("摩叙罗", "Mahoraga"),

    # Places and other terms
    ("死神千年血战篇", "Bleach: Đại Chiến Ngàn Năm"),
    ("死神", "Shinigami"),
    ("千年血战篇", "Đại Chiến Ngàn Năm"),
    ("血战篇", "Đại Chiến Ngàn Năm"),
    ("最终季", "Mùa cuối"),
    ("经费爆炸", "chất lượng hình ảnh cực khủng"),
    ("吉英社", "Shueisha"),
    ("集英社", "Shueisha"),
    ("将灵王", "Linh Vương"),
    ("灵王宫", "Cung điện Linh Vương"),
    ("灵王", "Linh Vương"),
    ("真世界城", "Thành Wahrwelt"),
    ("致死量", "Tử Vong Lượng (Deathdealing)"),
    ("混合磷子", "Linh tử hỗn hợp"),
    ("林子胡刃", "Linh tử nhận"),
    ("被逼腿", "bị đẩy lùi"),
    ("灭鹊师", "Quincy"),
    ("灭却师", "Quincy"),
    ("井刀身", "thân kiếm"),
    ("续班", "phần Hollow (Hư)"),
    ("五大特级战力", "5 Tiềm Lực Chiến Tranh Đặc Biệt"),
    ("5大特级战力", "5 Tiềm Lực Chiến Tranh Đặc Biệt"),
    ("特级战力", "Tiềm Lực Chiến Tranh Đặc Biệt"),
    ("霍静谭", "Sōkoku-tan"),
    ("祸星谭", "Sōkoku-tan"),
    ("相克谭", "Sōkoku-tan"),
    ("霍静弹", "Sōkoku-tan"),
    ("三界崩塌", "Tam Giới sụp đổ"),
    ("三界", "Tam Giới"),
]

# Sensitive word replacements for TikTok FYF guidelines
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

def preprocess_text(text):
    """Replaces known Bleach terms with placeholders in Chinese text."""
    mapping = {}
    placeholder_id = 0
    
    # Sort dictionary by key length descending to avoid partial matches
    sorted_dict = sorted(DICTIONARY, key=lambda x: len(x[0]), reverse=True)
    
    temp_text = text
    for zh_term, vi_term in sorted_dict:
        # Check if zh_term is in text
        if zh_term in temp_text:
            placeholder = f"__TERM_{placeholder_id}__"
            mapping[placeholder] = vi_term
            # Replace all occurrences
            temp_text = temp_text.replace(zh_term, placeholder)
            placeholder_id += 1
            
    return temp_text, mapping

def restore_placeholders(translated_text, mapping):
    """Restores the placeholders in translated Vietnamese text with their correct translations."""
    temp_text = translated_text
    
    # We will use regex to find any placeholder format like __TERM_0__ or __term_0__ or with spaces
    def repl_func(match):
        ph = f"__TERM_{match.group(1)}__"
        # Find in mapping case-insensitively
        for k, v in mapping.items():
            if k.lower() == ph.lower():
                return v
        return match.group(0) # fallback
        
    temp_text = re.sub(r'__\s*term\s*[-_]?\s*(\d+)\s*__', repl_func, temp_text, flags=re.IGNORECASE)
    return temp_text

def clean_vietnamese(text):
    """Sanitizes Vietnamese text to comply with TikTok FYF standards."""
    for pattern, repl in VIETNAMESE_REPLACEMENTS:
        # Case insensitive replacement while preserving word boundaries
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

def batch_translate(texts, translator, batch_size=50):
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        print(f"    Translating batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...", flush=True)
        
        # Filter empty strings (keep them as empty)
        non_empty_indices = [idx for idx, t in enumerate(chunk) if t.strip()]
        non_empty_texts = [chunk[idx] for idx in non_empty_indices]
        
        chunk_translated = [""] * len(chunk)
        if non_empty_texts:
            try:
                translated_results = translator.translate_batch(non_empty_texts)
            except Exception as e:
                print(f"Error during batch translation: {e}. Retrying with 1s sleep...", flush=True)
                time.sleep(1)
                translated_results = translator.translate_batch(non_empty_texts)
                
            for res_idx, orig_idx in enumerate(non_empty_indices):
                chunk_translated[orig_idx] = translated_results[res_idx]
                
        translated_texts.extend(chunk_translated)
        time.sleep(0.5) # small politeness delay
    return translated_texts

def main():
    translator = GoogleTranslator(source='zh-CN', target='vi')
    srt_dir = Path(r'c:\Users\Public\doanh\code\tool\dichvideo\srt-bleach')
    
    for json_path in srt_dir.glob('*.json'):
        if json_path.name.endswith('_translated.json'):
            continue
            
        print(f"Batch processing for {json_path.name}...", flush=True)
        blocks = json.loads(json_path.read_text(encoding='utf-8'))
        
        preprocessed_texts = []
        mappings = []
        
        for block in blocks:
            orig_text = block['text']
            preprocessed, mapping = preprocess_text(orig_text)
            preprocessed_texts.append(preprocessed)
            mappings.append(mapping)
            
        print(f"  Prepared {len(blocks)} blocks. Starting batch translation...", flush=True)
        translated_results = batch_translate(preprocessed_texts, translator, batch_size=50)
        
        print(f"  Restoring placeholders and sanitizing...", flush=True)
        translated_srt_lines = []
        for i, block in enumerate(blocks):
            translated_text = translated_results[i]
            restored = restore_placeholders(translated_text, mappings[i])
            clean = clean_vietnamese(restored)
            
            translated_srt_lines.append(f"{block['index']}")
            translated_srt_lines.append(f"{block['time']}")
            translated_srt_lines.append(f"{clean}\n")
            
        # Write translated SRT
        out_srt_name = json_path.stem + "_vi.srt"
        out_srt_path = srt_dir / out_srt_name
        out_srt_path.write_text('\n'.join(translated_srt_lines), encoding='utf-8')
        print(f"Finished writing translated SRT to {out_srt_path.name}\n", flush=True)

if __name__ == '__main__':
    main()
