"""Check actual attributed Python blocks, including attributes on pre/code tags."""
from html.parser import HTMLParser
from pathlib import Path
import ast
import sys
import pycodestyle


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
    failed = False
    for name in sys.argv[1:]:
        parser = Blocks()
        parser.feed(Path(name).read_text(encoding='utf-8'))
        for number, source in enumerate(parser.blocks, 1):
            ast.parse(source)
            errors = pycodestyle.Checker(lines=(source.rstrip()+'\n').splitlines(True)).check_all()
            print(f'{name}: Python block {number}: {errors} style errors')
            failed |= errors > 0
        if not parser.blocks:
            print(f'{name}: no declared Python blocks')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
