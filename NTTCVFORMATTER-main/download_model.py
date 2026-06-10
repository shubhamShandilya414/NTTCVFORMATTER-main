#!/usr/bin/env python3
"""Download HF embedding model with SSL verification disabled (corp network workaround)."""
import os
import sys
import ssl
import urllib.request
from pathlib import Path
# Disable SSL verification GLOBALLY before any HTTP libraries are imported
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
# Patch httpx to disable SSL verification
import httpx
_orig_client = httpx.Client.__init__
def _new_client(self, *a, **kw):
    kw['verify'] = False
    return _orig_client(self, *a, **kw)
httpx.Client.__init__ = _new_client
_orig_async = httpx.AsyncClient.__init__
def _new_async(self, *a, **kw):
    kw['verify'] = False
    return _orig_async(self, *a, **kw)
httpx.AsyncClient.__init__ = _new_async
try:
    import requests
    import urllib3
    urllib3.disable_warnings()
    _orig_req = requests.Session.request
    def _new_req(self, *a, **kw):
        kw['verify'] = False
        return _orig_req(self, *a, **kw)
    requests.Session.request = _new_req
except ImportError:
    pass
def main():
    project_root = Path(__file__).parent
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    target = models_dir / 'all-MiniLM-L6-v2'
    if target.exists() and any(target.iterdir()):
        files = list(target.rglob('*'))
        if any(f.name == 'config.json' or f.suffix in ['.bin', '.safetensors'] for f in files):
            print(f'Model already cached at: {target}')
            return
    print('Downloading sentence-transformers/all-MiniLM-L6-v2...')
    print(f'  Target: {target}')
    print(f'  SSL verification: DISABLED')
    print()
    # Method 1: Try via sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print('  Attempting download via sentence-transformers...')
        model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            cache_folder=str(models_dir / '.cache'),
        )
        target.mkdir(parents=True, exist_ok=True)
        model.save(str(target))
        print()
        print('SUCCESS via sentence-transformers!')
        verify(target)
        return
    except Exception as e:
        print(f'  sentence-transformers method failed: {type(e).__name__}: {str(e)[:200]}')
        print()
    # Method 2: Direct urllib download
    print('Falling back to direct urllib download...')
    direct_download(target)
def direct_download(target):
    base_url = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main'
    files = [
        'config.json',
        'config_sentence_transformers.json',
        'modules.json',
        'sentence_bert_config.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.txt',
        'model.safetensors',
        '1_Pooling/config.json',
    ]
    target.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ok = 0
    for f in files:
        url = f'{base_url}/{f}'
        out = target / f
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f'  {f}...', end=' ', flush=True)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                with open(out, 'wb') as fh:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        fh.write(chunk)
            mb = out.stat().st_size / 1024 / 1024
            print(f'OK ({mb:.1f}MB)')
            ok += 1
        except Exception as e:
            print(f'FAILED: {str(e)[:80]}')
    print()
    print(f'Downloaded {ok}/{len(files)} files')
    if ok >= 8:
        verify(target)
    else:
        print_manual(target)
def verify(target):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(str(target))
        emb = m.encode(['test'], convert_to_numpy=True)
        print(f'Model verified. Embedding shape: {emb.shape}')
        print()
        print('SUCCESS! Run the app: python -m streamlit run app.py')
    except Exception as e:
        print(f'Verification failed: {e}')
        print_manual(target)
def print_manual(target):
    print()
    print('=' * 60)
    print('MANUAL DOWNLOAD INSTRUCTIONS:')
    print('=' * 60)
    print()
    print('1. Open in browser:')
    print('   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main')
    print()
    print('2. Download all files and save to:')
    print(f'   {target}')
if __name__ == '__main__':
    main()
