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


def pixels(text, x, y, unit):
    cells = []
    for index, letter in enumerate(text.upper()):
        for row, bits in enumerate(GLYPHS[letter]):
            for column, bit in enumerate(bits):
                if bit == '1':
                    cells.append((x + (index * 6 + column) * unit, y + row * unit))
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
    lines = ['AN INNER WORLD', 'FOR INTELLIGENCE THAT PEOPLE', 'CAN UNDERSTAND AND SHAPE.']
    cells = []
    for number, line in enumerate(lines):
        width = (len(line) * 6 - 1) * 6
        cells.extend(pixels(line, (1000 - width) // 2, 12 + number * 62, 6))
    lettering = path(cells, 6)
    body = (f'<path d="{lettering}" fill="#193c32" transform="translate(5 7)"/>'
            f'<path d="{lettering}" fill="#60815a" transform="translate(2 3)"/>'
            f'<path d="{lettering}" fill="#c8dcb8"/>')
    highlights = [(x, y) for i, (x, y) in enumerate(cells) if i % 9 == 0]
    body += f'<path d="{path(highlights, 2)}" fill="#f0f6df"/>'
    (destination / 'headline.svg').write_text(svg(title, 1000, 210, body))
    for label in ['Demo', 'Chinese', 'English', 'Presentation']:
        width = len(label) * 18 + 40
        d = path(pixels(label, 20, 13, 3), 3)
        body = (f'<path d="M4 4h{width-8}v4h4v40h-4v4H4v-4H0V8h4z" fill="#193c32"/>'
                f'<path d="M7 7h{width-14}v3h4v32h-4v3H7v-3H3V10h4z" fill="#75935f"/>'
                f'<path d="M7 7h{width-14}v3H7v32H3V10h4z" fill="#e5efcf"/>'
                f'<path d="M7 10h{width-14}v29H7z" fill="#c8d8b7"/>'
                f'<path d="{d}" fill="#f1f5df" transform="translate(1 1)"/>'
                f'<path d="{d}" fill="#193c32"/>')
        (destination / f'button-{label.lower()}.svg').write_text(svg(label, width, 52, body))
    print('Built pixel headline and four navigation buttons.')


if __name__ == '__main__':
    build()
