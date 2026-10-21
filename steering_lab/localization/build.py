"""Localize a public IOTA snapshot without altering its application code or evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def build(app):
    app = Path(app)
    catalog = json.loads((ROOT / 'en.json').read_text())
    source = (ROOT / 'frontend-en.js').read_text().replace('__IOTA_EN_CATALOG__', json.dumps(catalog, ensure_ascii=False, separators=(',', ':')))
    css = (ROOT / 'frontend-en.css').read_text()
    version = hashlib.sha256((source + css).encode()).hexdigest()[:12]
    filename = f'frontend-en.{version}.js'
    (app / filename).write_text(source)
    stylesheet = f'frontend-en.{version}.css'
    (app / stylesheet).write_text(css)
    pages = []
    for path in sorted(app.rglob('*.html')):
        page = path.read_text()
        page = re.sub(r'<script defer src="/iota/frontend-en\.[a-f0-9]+\.js"></script>', '', page)
        page = re.sub(r'<link rel="stylesheet" href="/iota/frontend-en\.[a-f0-9]+\.css">', '', page)
        page = re.sub(r'(<html\b[^>]*\blang=")[^"]*(")', r'\g<1>en\2', page, count=1)
        # A single shared catalog covers the top-level page, lazy modules and standalone viewers.
        page = page.replace('</head>', f'<link rel="stylesheet" href="/iota/{stylesheet}"><script defer src="/iota/{filename}"></script></head>', 1)
        path.write_text(page)
        pages.append(str(path.relative_to(app)))
    manifest_path = app / 'deploy-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {f['path']: f for f in manifest['files']}
    for old in list(files):
        if re.fullmatch(r'frontend-en\.[a-f0-9]+\.(js|css)', old) and old not in (filename, stylesheet):
            files.pop(old)
            (app / old).unlink(missing_ok=True)
    for name in pages + [filename, stylesheet]:
        data = (app / name).read_bytes()
        files[name] = dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest['files'] = sorted(files.values(), key=lambda f: f['path'])
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(language='en', version=version, script=filename, html_pages=len(pages), catalog_entries=len(catalog)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    build(parser.parse_args().app)
