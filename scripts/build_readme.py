"""Build the English homepage; Chinese stays in its own Markdown page on main."""
from pathlib import Path
import posixpath
import re
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]


def render(source):
    def relative(url):
        parts = urlsplit(url)
        if parts.scheme or not parts.path or parts.path.startswith('/'):
            return url
        path = posixpath.normpath(posixpath.join(source.parent.as_posix(), parts.path))
        return urlunsplit(('', '', path, parts.query, parts.fragment))

    text = (ROOT / source).read_text()
    text = re.sub(r'\]\(([^)]+)\)', lambda m: '](' + relative(m[1]) + ')', text)
    return re.sub(r'(src|href)="([^"]+)"',
                  lambda m: m[1] + '="' + relative(m[2]) + '"', text)


def build():
    english = render(Path('docs/translations/README.en.md'))
    (ROOT / 'README.md').write_text(english.rstrip() + '\n')
    print('Built English homepage; Chinese has a separate rendered page on main.')


if __name__ == '__main__':
    build()
