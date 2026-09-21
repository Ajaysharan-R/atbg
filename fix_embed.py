from pathlib import Path
p = Path('atbg_model/embeddings.py')
text = p.read_text(encoding='utf-8-sig')
text = text.replace(
    '_model = SentenceTransformer(_MODEL_NAME)',
    '_model = SentenceTransformer(_MODEL_NAME, device="cpu")'
)
p.write_text(text, encoding='utf-8')
print('patched')
print(p.read_text()[200:400])
