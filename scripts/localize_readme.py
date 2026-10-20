"""Render the Chinese repository-home README from its maintained translation."""
import argparse
from pathlib import Path
import posixpath
import re
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('docs/translations/README.zh-CN.md')


def render():
    def relative(url):
        parts = urlsplit(url)
        if parts.scheme or not parts.path or parts.path.startswith('/'):
            return url
        path = posixpath.normpath(posixpath.join(SOURCE.parent.as_posix(), parts.path))
        return urlunsplit(('', '', path, parts.query, parts.fragment))

    text = (ROOT / SOURCE).read_text()
    text = re.sub(r'\]\(([^)]+)\)', lambda m: '](' + relative(m[1]) + ')', text)
    return re.sub(r'(src|href)="([^"]+)"',
                  lambda m: m[1] + '="' + relative(m[2]) + '"', text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.write_text(render())
    else:
        print(render(), end='')
