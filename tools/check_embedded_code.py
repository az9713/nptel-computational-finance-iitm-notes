"""Check actual attributed Python blocks, including attributes on pre/code tags."""
from html.parser import HTMLParser
from pathlib import Path
import argparse
import ast
import pycodestyle

ROOT = Path(__file__).resolve().parents[1]

class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_pre = False
        self.current = None
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if tag == 'pre':
            self.in_pre = True
        if tag == 'code' and self.in_pre:
            attributes = dict(attrs)
            if 'language-python' in attributes.get('class', ''):
                self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == 'code' and self.current is not None:
            self.blocks.append(''.join(self.current))
            self.current = None
        if tag == 'pre':
            self.in_pre = False


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        'files', nargs='*',
        help='Chapter HTML files; default: all existing root chapter files.')
    args = arguments.parse_args()
    selected = ([Path(name).absolute() for name in args.files] if args.files else
                sorted(ROOT.glob('chapter-*.html')))
    if not selected:
        raise SystemExit('No chapter files selected; no check was performed')
    failed = False
    block_count = 0
    for path in selected:
        if (path.resolve().parent != ROOT or not path.is_file() or
                path.is_symlink()):
            raise SystemExit('Expected an ordinary root chapter file: '
                             f'{path.name}')
        name = path.name
        parser = Blocks()
        parser.feed(path.read_text(encoding='utf-8'))
        for number, source in enumerate(parser.blocks, 1):
            block_count += 1
            try:
                ast.parse(source)
            except SyntaxError as error:
                print(f'{name}: Python block {number}: syntax error: {error}')
                failed = True
                continue
            lines = (source.rstrip()+'\n').splitlines(True)
            errors = pycodestyle.Checker(lines=lines).check_all()
            print(f'{name}: Python block {number}: {errors} style errors')
            failed |= errors > 0
        if not parser.blocks:
            print(f'{name}: no declared Python blocks')
    if not block_count:
        raise SystemExit('No Python blocks found; no code was checked')
    print(f'Parsed and linted {block_count} Python blocks in {len(selected)} '
          'chapter files. Runtime execution is a separate check.')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
