"""Static, offline-browser, and release gates for the public reading edition."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.ids, self.links, self.assets = [], [], []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if 'href' in attrs:
            self.links.append(attrs['href'])
        if 'src' in attrs:
            self.assets.append(attrs['src'])


def static_check(release):
    manifest = json.loads((ROOT / 'chapter-manifest.json').read_text(encoding='utf-8'))
    sources = json.loads((ROOT / 'source-manifest.json').read_text(encoding='utf-8'))
    records = {r['video_id']: r for r in sources['recordings']}
    paths = [ROOT / f for f in ('index.html', 'source-map.html', 'references.html', 'code-lab.html')]
    paths += [ROOT / c['file'] for c in manifest['chapters'] if (ROOT / c['file']).exists()]
    pages = {p.name: Page(p) for p in paths}
    problems = []
    for name, page in pages.items():
        duplicates = [key for key, count in Counter(page.ids).items() if count > 1]
        if duplicates:
            problems.append(f'{name}: duplicate ids {duplicates}')
        for link in page.links + page.assets:
            u = urlsplit(link)
            if u.scheme or u.netloc:
                if u.netloc.endswith('youtube.com') and u.path == '/watch':
                    query = parse_qs(u.query)
                    vid = query.get('v', [''])[0]
                    stamp = query.get('t', ['0'])[0]
                    if vid in records and re.fullmatch(r'\d+s?', stamp):
                        if int(stamp.rstrip('s')) > records[vid]['duration_seconds']:
                            problems.append(f'{name}: timestamp beyond recording')
                continue
            target = (page.path.parent / unquote(u.path)).resolve() if u.path else page.path
            if not target.is_relative_to(ROOT) or not target.exists():
                problems.append(f'{name}: broken local link {link}')
            elif u.fragment and target.suffix == '.html':
                other = pages.get(target.name) or Page(target)
                if unquote(u.fragment) not in other.ids:
                    problems.append(f'{name}: missing anchor {link}')
        text = page.path.read_text(encoding='utf-8')
        if re.search(r'\[(?:PROOF NEEDED|CITATION NEEDED|NUMBER NOT YET|AUTHOR DECISION)', text):
            problems.append(f'{name}: unresolved authoring placeholder')
    if release:
        for chapter in manifest['chapters']:
            path = ROOT / chapter['file']
            meta_path = path.with_suffix('.meta.json')
            if not path.exists() or not meta_path.exists():
                problems.append(f'Chapter {chapter["number"]}: missing manuscript or sidecar')
                continue
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            body = path.read_text(encoding='utf-8').split('<!-- BEGIN CHAPTER -->')[1].split('<!-- END CHAPTER -->')[0]
            digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
            reviews = meta.get('reviews', [])
            if meta.get('status') != 'accepted' or len(reviews) < 2:
                problems.append(f'Chapter {chapter["number"]}: two review rounds and acceptance required')
            elif reviews[-1].get('body_sha256') != digest or reviews[-1].get('blocking_findings') != 0:
                problems.append(f'Chapter {chapter["number"]}: latest scientific review is stale or blocking')
    return paths, problems


def browser_check(paths):
    temp = ROOT / 'work/browser-tmp'
    temp.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temp)
    from playwright.sync_api import sync_playwright
    results = []
    with sync_playwright() as p:
        # Background runtimes can omit ProgramFiles although Chrome is installed.
        # Reuse a discovered binary; profiles and temporary data stay in work/.
        candidates = []
        if os.name == 'nt':
            drive = os.environ.get('SystemDrive') or Path.home().drive or 'C:'
            bases = [Path(os.environ.get('ProgramFiles') or drive+'/Program Files'),
                     Path(os.environ.get('ProgramFiles(x86)') or drive+'/Program Files (x86)')]
            local = os.environ.get('LOCALAPPDATA')
            if local:
                bases.append(Path(local))
            for suffix in ('Google/Chrome/Application/chrome.exe',
                           'Microsoft/Edge/Application/msedge.exe'):
                candidates.extend(base / suffix for base in bases)
        attempts = []
        browser = None
        for candidate in candidates:
            if candidate.is_file():
                try:
                    browser = p.chromium.launch(executable_path=str(candidate),
                                                headless=True)
                    break
                except Exception as error:
                    attempts.append(str(error))
        if browser is None:
            for channel in ('chrome', 'msedge'):
                try:
                    browser = p.chromium.launch(channel=channel, headless=True)
                    break
                except Exception as error:
                    attempts.append(str(error))
        if browser is None:
            raise RuntimeError('Installed browser launch failed: '+' | '.join(attempts))
        context = browser.new_context(viewport={'width': 1440, 'height': 1000})
        # A genuinely offline render must not silently fetch a CDN or font.
        external = []
        def route_request(route):
            if route.request.url.startswith(('http://', 'https://')):
                external.append(route.request.url)
                route.abort()
            else:
                route.continue_()
        context.route('**/*', route_request)
        for path in paths:
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(path.as_uri(), wait_until='load')
            page.wait_for_function('window.MathJax && MathJax.startup && MathJax.startup.promise')
            page.evaluate('() => MathJax.startup.promise.then(() => true)')
            math_errors = page.locator('mjx-merror, [data-mjx-error]').count()
            math_count = page.locator('mjx-container').count()
            is_chapter = path.name.startswith('chapter-')
            if is_chapter and math_count == 0:
                errors.append('No mathematics rendered')
            if math_errors:
                errors.append(f'{math_errors} MathJax errors')
            page.locator('#theme-toggle').click()
            if not page.locator('body').evaluate('(e) => e.classList.contains("light")'):
                errors.append('Theme control failed')
            page.locator('#theme-toggle').click()
            page.locator('#text-toggle').click()
            if not page.locator('body').evaluate('(e) => e.classList.contains("large-text")'):
                errors.append('Text control failed')
            page.locator('#text-toggle').click()
            solutions = page.locator('article details')
            if solutions.count():
                page.locator('#solutions-toggle').click()
                if not page.locator('article details').evaluate_all('(xs) => xs.every(x => x.open)'):
                    errors.append('Expand-solutions control failed')
                page.locator('#solutions-toggle').click()
                page.evaluate('() => window.dispatchEvent(new Event("beforeprint"))')
                if not page.locator('article details').evaluate_all('(xs) => xs.every(x => x.open)'):
                    errors.append('Print expansion failed')
                page.evaluate('() => window.dispatchEvent(new Event("afterprint"))')
                if not page.locator('article details').evaluate_all('(xs) => xs.every(x => !x.open)'):
                    errors.append('Print restoration failed')
            if page.locator('pre code').count():
                # Inspect exact copied payload without changing the user's clipboard.
                page.evaluate('''() => Object.defineProperty(navigator, 'clipboard', {
                    configurable: true, value: {writeText: async text => {window.__copiedCode = text;}}
                })''')
                expected = page.locator('pre code').first.text_content()
                page.locator('pre button.copy-code').first.click()
                if page.evaluate('() => window.__copiedCode') != expected:
                    errors.append('Copy-code payload failed')
            if path.name == 'index.html':
                page.locator('#lecture-search').fill('rates approximations')
                shown = page.locator('[data-lecture]:visible').count()
                if shown != 1:
                    errors.append('Chapter search did not isolate the expected chapter')
                page.locator('#lecture-search').fill('')
            if is_chapter and not page.locator('#chapter-toc a').count():
                errors.append('Section navigation missing')
            page.set_viewport_size({'width': 390, 'height': 844})
            overflow = page.evaluate('() => document.documentElement.scrollWidth > innerWidth + 2')
            if overflow:
                errors.append('Mobile page overflow')
            page.screenshot(path=str(temp / (path.stem + '-mobile.png')), full_page=False)
            results.append({'file': path.name, 'math_expressions': math_count,
                            'math_errors': math_errors, 'errors': errors,
                            'passed': not errors})
            page.close()
        browser.close()
    if external:
        results.append({'file': 'offline-boundary', 'passed': False,
                        'errors': ['External resource requests occurred'], 'requests': external})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    paths, problems = static_check(args.release)
    try:
        browser = browser_check(paths) if args.browser else []
    except Exception as error:
        browser = [{'file': 'browser-run', 'passed': False,
                    'errors': [f'{type(error).__name__}: {error}']}]
    passed = not problems and all(r['passed'] for r in browser)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'release_gate': args.release,
              'files_checked': [p.name for p in paths], 'static_problems': problems,
              'file_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'asset_sha256': {p.relative_to(ROOT / 'assets').as_posix():
                               hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / 'assets').rglob('*') if p.is_file()},
              'browser': browser, 'passed': passed,
              'scope': 'Presentation, local links, recording timestamp bounds, reader behavior, and requested release evidence; scientific correctness requires separate review. Copy checks capture the Clipboard API payload without changing the OS clipboard.'}
    (ROOT / 'reader-validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for problem in problems:
        print(problem)
    for result in browser:
        print(f'{result["file"]}: {"PASS" if result["passed"] else result["errors"]}')
    print('Reader validation:', 'PASS' if passed else 'FAIL')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
