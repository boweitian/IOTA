"""Verify that an English frontend build preserves existing application code and data."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.current = None
        self.items = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.current = [dict(attrs), '']

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if tag == 'script' and self.current is not None:
            if not re.fullmatch(r'/iota/frontend-en\.[a-f0-9]+\.js', self.current[0].get('src', '')):
                self.items.append(self.current)
            self.current = None


def verify(before, after, manifest_path=None):
    original = json.loads((manifest_path or before / 'deploy-manifest.json').read_text())
    translated = json.loads((after / 'deploy-manifest.json').read_text())
    assert original['backend'] == translated['backend'], 'Backend changed'
    pages = 0
    preserved = 0
    for entry in original['files']:
        name = entry['path']
        left, right = (before / name).read_bytes(), (after / name).read_bytes()
        assert hashlib.sha256(left).hexdigest() == entry['sha256'], name
        if name.endswith('.html'):
            a, b = Scripts(), Scripts()
            a.feed(left.decode())
            b.feed(right.decode())
            assert a.items == b.items, f'Existing scripts or embedded data changed: {name}'
            normalized = re.sub(r'<script defer src="/iota/frontend-en\.[a-f0-9]+\.js"></script>', '', right.decode())
            normalized = re.sub(r'<link rel="stylesheet" href="/iota/frontend-en\.[a-f0-9]+\.css">', '', normalized)
            assert normalized == left.decode().replace('lang="zh-CN"', 'lang="en"'), f'Unexpected HTML change: {name}'
            pages += 1
        else:
            assert left == right, f'Existing application file changed: {name}'
            preserved += 1
    for entry in translated['files']:
        content = (after / entry['path']).read_bytes()
        assert len(content) == entry['size']
        assert hashlib.sha256(content).hexdigest() == entry['sha256'], entry['path']
    return dict(passed=True, english_pages=pages, unchanged_application_and_data_files=preserved, backend_unchanged=True, existing_inline_scripts_and_data_unchanged=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('--before-manifest', type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.before, args.after, args.before_manifest), indent=2))
