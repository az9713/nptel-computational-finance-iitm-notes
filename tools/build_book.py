"""Rebuild the public offline reader from root chapter bodies and manifests.

No private sources, network access, or nonstandard Python packages are needed.
"""
from __future__ import annotations
import hashlib
from html import escape
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = '<!-- BEGIN CHAPTER -->', '<!-- END CHAPTER -->'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def chapter_path(name):
    if not isinstance(name, str) or not re.fullmatch(r'chapter-\d{2}-[a-z0-9-]+\.html', name):
        raise ValueError('invalid chapter filename')
    original = ROOT / name
    path = original.resolve()
    if path.parent != ROOT or original.is_symlink():
        raise ValueError('chapter must be an ordinary root file')
    return path


def body_of(path):
    text = path.read_text(encoding='utf-8')
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError(f'{path.name}: exactly one chapter body required')
    return text.split(BEGIN, 1)[1].split(END, 1)[0]


def reviewed_artifact(name):
    patterns = (r'code/ch\d{2}_[a-z0-9_]+\.py',
                r'figures/ch\d{2}_[a-z0-9_]+\.py',
                r'assets/ch\d{2}-[a-z0-9-]+\.svg')
    if not isinstance(name, str) or not any(re.fullmatch(p, name)
                                           for p in patterns):
        raise ValueError('invalid reviewed artifact path')
    original = ROOT / name
    path = original.resolve()
    if original.is_symlink() or path.parent != ROOT / name.split('/')[0]:
        raise ValueError('reviewed artifact must stay in its public directory')
    return path


def wrap(title, body, *, chapter=False, turns=''):
    toc = '<aside id="chapter-toc" class="toc" aria-label="On this page"></aside>' if chapter else ''
    layout = 'layout' if chapter else 'layout no-toc'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light"><title>{escape(title)} · Computational Finance 1</title>
<link rel="stylesheet" href="assets/book.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true}},svg:{{fontCache:'local'}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script defer src="assets/mathjax-tex-svg.js"></script><script defer src="assets/book.js"></script></head>
<body><a class="skip" href="#main">Skip to content</a>
<header class="site-header"><a class="brand" href="index.html">Computational Finance 1</a>
<nav class="course-nav" aria-label="Book"><a href="index.html">Chapters</a><a href="source-map.html">Source map</a><a href="references.html">References</a><a href="code-lab.html">Code lab</a></nav></header>
<div class="{layout}">{toc}<main id="main">
<div class="reading-tools" aria-label="Reading controls"><button id="theme-toggle" aria-pressed="false">Light theme</button><button id="text-toggle" aria-pressed="false">Larger text</button><button id="solutions-toggle" aria-pressed="false">Expand solutions</button><button id="print-page">Print with solutions</button></div>
<article id="chapter-content">{BEGIN}{body}{END}
</article>{turns}</main></div>
<footer class="site-footer">Independent study edition. NPTEL course taught by Sourish Das and Bikramaditya Datta, Chennai Mathematical Institute and IIT Kanpur; coordinated by IIT Madras. Source recordings determine the topic sequence; added derivations and teaching examples are separately identified. See the source map and validation records for evidence and limits.</footer></body></html>
'''


def write_page(name, title, body):
    (ROOT / name).write_text(wrap(title, body), encoding='utf-8', newline='\n')


def main():
    manifest, source = read_json('chapter-manifest.json'), read_json('source-manifest.json')
    chapters = manifest['chapters']
    records = {r['playlist_index']: r for r in source['recordings']}
    written = []
    assignments = {}
    for c in chapters:
        if not isinstance(c['number'], int) or isinstance(c['number'], bool):
            raise ValueError('chapter number must be an integer')
        if not isinstance(c['title'], str):
            raise ValueError('chapter title must be text')
        path = chapter_path(c['file'])
        for number in c['recordings']:
            if number not in records or number in assignments:
                raise ValueError('recordings require one valid primary assignment')
            assignments[number] = c
        if path.exists():
            written.append(c)
    if set(assignments) != set(records):
        raise ValueError('every playlist recording requires a chapter assignment')
    inventory = []
    for i, c in enumerate(written):
        path = chapter_path(c['file'])
        body = body_of(path)
        links = []
        for label, neighbor in [('Previous', written[i-1] if i else None),
                                ('Next', written[i+1] if i+1 < len(written) else None)]:
            if neighbor:
                links.append(f'<a href="{escape(neighbor["file"])}">{label}: {escape(neighbor["title"])}</a>')
        turns = '<nav class="page-turn" aria-label="Chapter navigation">' + ''.join(links) + '</nav>'
        path.write_text(wrap(c['title'], body, chapter=True, turns=turns), encoding='utf-8', newline='\n')
        digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
        sidecar = path.with_suffix('.meta.json')
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        c['status'] = meta['status']
        if meta['status'] == 'accepted':
            reviews = meta.get('reviews', [])
            if len(reviews) < 2 or reviews[-1].get('body_sha256') != digest or reviews[-1].get('blocking_findings') != 0:
                raise ValueError(f'{path.name}: acceptance is missing or stale')
            for listing in meta.get('code', []):
                current_code = hashlib.sha256(reviewed_artifact(listing['path']).read_bytes()).hexdigest()
                if reviews[-1].get('code_sha256', {}).get(listing['path']) != current_code:
                    raise ValueError(f'{path.name}: accepted program has changed')
            for name, reviewed_digest in reviews[-1].get('artifact_sha256', {}).items():
                actual = hashlib.sha256(reviewed_artifact(name).read_bytes()).hexdigest()
                if actual != reviewed_digest:
                    raise ValueError(f'{path.name}: accepted figure artifact has changed')
        # A rebuild may update inventory, but cannot invent or promote a review.
        meta['body_sha256'] = digest
        sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        inventory.append({'chapter': c['number'], 'file': c['file'], 'status': meta['status'],
                          'body_sha256': digest, 'recordings': c['recordings']})
    accepted = sum(c['status'] == 'accepted' for c in inventory)
    body = f'''<h1>Computational Finance 1: an independent textbook rewrite</h1>
<p>From discounted cash flows to constrained portfolios, option-pricing algorithms, and Monte Carlo uncertainty.</p>
<p>NPTEL · Sourish Das and Bikramaditya Datta · Chennai Mathematical Institute and IIT Kanpur · IIT Madras coordination.</p>
<p>{len(records)} source recordings, {len(chapters)} planned topic chapters, {len(written)} written, {accepted} accepted after review.</p>
<h2 id="learning-route">Learning route and prerequisites</h2>
<p>Begin with calculus, linear algebra, probability, and basic programming. Chapters 1–6 develop financial objects and numerical linear algebra; 7–12 develop options, replication, and calibration; 13–18 develop calculus, stochastic pricing, and grids; 19–24 develop constrained optimization, arbitrage, and simulation. Paired R/Python labs and worked-problem recordings are mapped explicitly.</p>
<p>These are independent educational notes. Synthetic examples are identified; source statements, researched extensions, and original teaching constructions remain distinct. Read the exercises with their expandable solutions, reproduce the code, and consult the source map for recording provenance.</p>
<label class="search-label" for="lecture-search">Search chapters</label><input id="lecture-search" type="search" placeholder="Search title, topic, or status"><p id="search-status" role="status" aria-live="polite">{len(chapters)} chapters shown</p><ol class="lecture-list">'''
    for c in chapters:
        title = escape(c['title'])
        if chapter_path(c['file']).exists():
            title = f'<a href="{escape(c["file"])}">{title}</a>'
        body += f'<li data-lecture><h2>{c["number"]:02d}. {title}</h2><span class="status">{escape(c["status"])}</span><p>{escape(c["purpose"])}</p></li>'
    write_page('index.html', 'Chapter index', body+'</ol>')
    rows = []
    mapped_sections = {number: [] for number in records}
    for chapter in written:
        meta = read_json(chapter['file'].replace('.html', '.meta.json'))
        for span in meta.get('coverage', []):
            number, section = span['recording'], span['section']
            if number not in records or not re.fullmatch(r'[a-z0-9-]+', section):
                raise ValueError('invalid section coverage identity')
            if not 0 <= span['start_seconds'] <= span['end_seconds'] <= records[number]['duration_seconds']:
                raise ValueError('coverage span beyond recording')
            body = body_of(chapter_path(chapter['file']))
            if f'id="{section}"' not in body:
                raise ValueError('coverage section has no manuscript anchor')
            label = f'Chapter {chapter["number"]}: {span["topic"]} ({meta["status"]})'
            target = f'{chapter["file"]}#{section}'
            mapped_sections[number].append(f'<a href="{escape(target)}">{escape(label)}</a>')
    for number, r in records.items():
        c = assignments[number]
        target = escape(c['title'])
        if chapter_path(c['file']).exists():
            target = f'<a href="{escape(c["file"])}">{target}</a>'
        sections = '<br>'.join(mapped_sections[number]) or 'Not yet drafted'
        rows.append(f'<tr><td>{number}</td><td><a href="{escape(r["url"])}">{escape(r["title"])}</a></td><td>{r["duration_seconds"]}</td><td>{target}</td><td>{sections}</td></tr>')
    write_page('source-map.html', 'Source map', '<h1>Source map</h1><p>The playlist is the coverage spine. Recording positions preserve the supplied playlist order, including its introduction, worked problems, and language labs. A primary chapter assignment is a plan. Mapped sections report actual written coverage, including secondary topics from mixed workshops, with their review status.</p><p>Machine transcripts use local faster-whisper small.en. Recognition can misstate names and mathematics. Raw audio and transcripts remain private. Chapter sidecars identify section spans, additions, clarifications, and reviewed body digests.</p><p><a href="source-manifest.json">Source manifest</a> · <a href="completion.json">Completion inventory</a></p><table><thead><tr><th>Playlist position</th><th>Recording</th><th>Seconds</th><th>Primary chapter</th><th>Mapped sections and status</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
    code_rows = []
    for c in written:
        meta = read_json(c['file'].replace('.html', '.meta.json'))
        for code in meta.get('code', []):
            p = code['path']
            if not re.fullmatch(r'code/ch\d{2}_[a-z0-9_]+\.py', p):
                raise ValueError('invalid program path')
            code_rows.append(f'<tr><td>{c["number"]}</td><td><a href="{escape(p)}">{escape(p)}</a></td><td>{escape(code["check"])}</td></tr>')
    write_page('code-lab.html', 'Code laboratory', '<h1>Code laboratory</h1><p>Programs implement the mathematical objects derived in their chapters. Inputs are synthetic unless explicitly sourced. Download the repository and run each named Python file from its root. The code report records execution separately from scientific review.</p><p><a href="chapter-code-validation.json">Execution record</a></p><table><thead><tr><th>Chapter</th><th>Complete program</th><th>Checks</th></tr></thead><tbody>'+''.join(code_rows)+'</tbody></table>')
    refs = read_json('references.json')
    ref_items = ''.join(f'<li><a href="{escape(r["url"])}">{escape(r["title"])}</a>: {escape(r["supports"])}</li>' for r in refs)
    write_page('references.html', 'Primary references', '<h1>Primary references</h1><p>References support the particular claims identified in chapter prose. Listing a source here does not imply that its entire contents were audited. Recordings remain linked rather than redistributed.</p><ul>'+ref_items+'</ul>')
    completion = {'recordings_total': len(records), 'chapters_planned': len(chapters),
                  'chapters_written': len(written), 'chapters_accepted': accepted,
                  'complete': len(written) == len(chapters) == accepted,
                  'chapters': inventory}
    (ROOT / 'completion.json').write_text(json.dumps(completion, indent=2)+'\n', encoding='utf-8')
    (ROOT / '.nojekyll').touch()
    print(f'Built {len(written)} chapters and four navigation pages; accepted={accepted}')


if __name__ == '__main__':
    main()
