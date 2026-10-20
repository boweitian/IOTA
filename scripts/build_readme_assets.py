"""Build portable SVG pixel lettering for GitHub's sanitized README renderer."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLYPHS = {
    'A': ['01110','10001','10001','11111','10001','10001','10001'],
    'B': ['11110','10001','10001','11110','10001','10001','11110'],
    'C': ['01111','10000','10000','10000','10000','10000','01111'],
    'D': ['11110','10001','10001','10001','10001','10001','11110'],
    'E': ['11111','10000','10000','11110','10000','10000','11111'],
    'F': ['11111','10000','10000','11110','10000','10000','10000'],
    'G': ['01111','10000','10000','10111','10001','10001','01111'],
    'H': ['10001','10001','10001','11111','10001','10001','10001'],
    'I': ['11111','00100','00100','00100','00100','00100','11111'],
    'J': ['00111','00010','00010','00010','10010','10010','01100'],
    'K': ['10001','10010','10100','11000','10100','10010','10001'],
    'L': ['10000','10000','10000','10000','10000','10000','11111'],
    'M': ['10001','11011','10101','10101','10001','10001','10001'],
    'N': ['10001','11001','10101','10011','10001','10001','10001'],
    'O': ['01110','10001','10001','10001','10001','10001','01110'],
    'P': ['11110','10001','10001','11110','10000','10000','10000'],
    'Q': ['01110','10001','10001','10001','10101','10010','01101'],
    'R': ['11110','10001','10001','11110','10100','10010','10001'],
    'S': ['01111','10000','10000','01110','00001','00001','11110'],
    'T': ['11111','00100','00100','00100','00100','00100','00100'],
    'U': ['10001','10001','10001','10001','10001','10001','01110'],
    'V': ['10001','10001','10001','10001','10001','01010','00100'],
    'W': ['10001','10001','10001','10101','10101','11011','10001'],
    'X': ['10001','10001','01010','00100','01010','10001','10001'],
    'Y': ['10001','10001','01010','00100','00100','00100','00100'],
    'Z': ['11111','00001','00010','00100','01000','10000','11111'],
    '.': ['00000','00000','00000','00000','00000','00100','00100'],
    ' ': ['00000'] * 7,
}


def pixels(text, x, y, unit, advance=6):
    cells = []
    for index, letter in enumerate(text.upper()):
        for row, bits in enumerate(GLYPHS[letter]):
            for column, bit in enumerate(bits):
                if bit == '1':
                    cells.append((x + (index * advance + column) * unit, y + row * unit))
    return cells


def path(cells, unit):
    return ''.join(f'M{x},{y}h{unit}v{unit}h-{unit}z' for x, y in cells)


def svg(title, width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title" shape-rendering="crispEdges">'
            f'<title id="title">{escape(title)}</title>{body}</svg>\n')


def build():
    destination = ROOT / 'assets/brand'
    destination.mkdir(parents=True, exist_ok=True)
    title = 'An inner world for intelligence that people can understand and shape.'
    lines = ['AN INNER WORLD FOR INTELLIGENCE', 'THAT PEOPLE CAN UNDERSTAND AND SHAPE.']
    cells = []
    for number, line in enumerate(lines):
        width = ((len(line) - 1) * 7 + 5) * 3
        cells.extend(pixels(line, (900 - width) // 2, 23 + number * 42, 3, advance=7))
    lettering = path(cells, 3)
    body = ('<path d="M4 0h892v4h4v104h-4v4H4v-4H0V4h4z" fill="#071b3c"/>'
            f'<path d="{lettering}" fill="#bdd7f0"/>')
    (destination / 'moonlight-headline.svg').write_text(svg(title, 900, 112, body))
    for label in ['Demo', 'Chinese', 'English', 'Presentation', 'Blog', 'Language']:
        width = len(label) * 12 + 32
        d = path(pixels(label, 17, 11, 2), 2)
        body = (f'<path d="M3 0h{width-6}v3h3v30h-3v3H3v-3H0V3h3z" fill="#527bac"/>'
                f'<path d="M4 1h{width-8}v3h3v28h-3v3H4v-3H1V4h3z" fill="#0b284e"/>'
                f'<path d="M8 3h{width-16}v1H8z" fill="#244a78"/>'
                f'<path d="{d}" fill="#c8ddf5"/>')
        filename = 'moonlight-language-plain.svg' if label == 'Language' else f'moonlight-{label.lower()}.svg'
        (destination / filename).write_text(svg(label, width, 36, body))
    print('Built pixel headline and six navigation buttons.')


if __name__ == '__main__':
    build()
