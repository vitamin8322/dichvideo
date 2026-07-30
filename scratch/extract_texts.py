import re
import json
from pathlib import Path

def parse_srt(srt_path):
    content = Path(srt_path).read_text(encoding='utf-8')
    # Normalize line endings
    content = content.replace('\r\n', '\n')
    blocks = content.strip().split('\n\n')
    parsed_blocks = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0].strip()
            time_range = lines[1].strip()
            text = ' '.join(l.strip() for l in lines[2:])
            parsed_blocks.append({
                'index': index,
                'time': time_range,
                'text': text
            })
        elif len(lines) == 2:
            # Maybe empty text
            index = lines[0].strip()
            time_range = lines[1].strip()
            parsed_blocks.append({
                'index': index,
                'time': time_range,
                'text': ''
            })
    return parsed_blocks

def main():
    srt_dir = Path(r'c:\Users\Public\doanh\code\tool\dichvideo\srt-bleach')
    for p in srt_dir.glob('*.srt'):
        if 'translated' in p.name:
            continue
        blocks = parse_srt(p)
        out_path = p.with_suffix('.json')
        out_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Parsed {p.name} -> {out_path.name} with {len(blocks)} blocks.")

if __name__ == '__main__':
    main()
