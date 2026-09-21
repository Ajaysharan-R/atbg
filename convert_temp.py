import json, os, sys

SPEAKER_MAP = {'agent': 'attacker', 'user': 'victim', 'system': 'attacker'}

def convert_record(rec):
    label = rec.get('label', 'unknown')
    is_attack = True if label == 'scam' else (False if label == 'benign' else None)
    turns = []
    for i, t in enumerate(rec.get('turns', [])):
        speaker = SPEAKER_MAP.get(str(t.get('speaker','')).lower().strip(), 'unknown')
        text = str(t.get('text', '')).strip()
        if text:
            turns.append({'turn_id': i+1, 'speaker': speaker, 'timestamp': None, 'text': text})
    return {'conversation_id': rec.get('conversation_id',''), 'source': 'real_atbg_ready', 'channel': 'unknown', 'language': 'unknown', 'is_attack': is_attack, 'scam_type': None, 'is_synthetic': False, 'turns': turns}

input_dir = r'D:\REAL_ATBG_READY\REAL_ATBG_READY\dataset\splits'
output_dir = r'dataset\real'
os.makedirs(output_dir, exist_ok=True)

for split in ['train', 'val', 'test']:
    inp = os.path.join(input_dir, f'{split}.jsonl')
    if not os.path.exists(inp):
        print(f'[{split}] not found'); continue
    out = os.path.join(output_dir, f'{split}.jsonl')
    n = na = nb = 0
    with open(inp, encoding='utf-8') as fi, open(out, 'w', encoding='utf-8') as fo:
        for line in fi:
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except: continue
            c = convert_record(rec)
            if not c['turns']: continue
            fo.write(json.dumps(c, ensure_ascii=False) + '\n')
            n += 1; na += c['is_attack'] is True; nb += c['is_attack'] is False
    print(f'[{split}] {n} conversations (attack={na}, benign={nb}) -> {out}')
