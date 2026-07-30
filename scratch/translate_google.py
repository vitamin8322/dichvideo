import json
import time
from pathlib import Path
from deep_translator import GoogleTranslator

def main():
    translator = GoogleTranslator(source='zh-CN', target='vi')
    srt_dir = Path(r'c:\Users\Public\doanh\code\tool\dichvideo\srt-bleach')
    
    for json_path in srt_dir.glob('*.json'):
        if json_path.name.endswith('_translated.json'):
            continue
            
        print(f"Translating {json_path.name}...")
        blocks = json.loads(json_path.read_text(encoding='utf-8'))
        
        translated_blocks = []
        for i, block in enumerate(blocks):
            text = block['text']
            if text:
                try:
                    translated = translator.translate(text)
                except Exception as e:
                    print(f"Error at block {i}: {e}. Retrying after 1s...")
                    time.sleep(1)
                    translated = translator.translate(text)
            else:
                translated = ''
            
            translated_blocks.append({
                'index': block['index'],
                'time': block['time'],
                'original': text,
                'translated': translated
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Translated {i + 1}/{len(blocks)} blocks...")
                
        out_path = json_path.parent / f"{json_path.stem}_translated.json"
        out_path.write_text(json.dumps(translated_blocks, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Finished {json_path.name} -> {out_path.name}")

if __name__ == '__main__':
    main()
