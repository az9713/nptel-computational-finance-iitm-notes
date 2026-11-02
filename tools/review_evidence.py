"""Check current scientific-review attestations without inventing acceptance.

Public builds check artifact bindings. Release checks can additionally require
the private, hash-matching reports on the authoring workstation. A fresh public
clone intentionally does not contain those reports.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re

BEGIN = b'<!-- BEGIN CHAPTER -->'
END = b'<!-- END CHAPTER -->'
SHA256 = re.compile(r'[0-9a-f]{64}')
ARTIFACT = re.compile(
    r'(?:code|figures)/ch\d{2}_[a-z0-9_]+\.py|'
    r'assets/ch\d{2}-[a-z0-9-]+\.svg')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def body_bytes(data):
    if data.count(BEGIN) != 1 or data.count(END) != 1:
        raise ValueError('exactly one pair of chapter markers required')
    first, last = data.index(BEGIN) + len(BEGIN), data.index(END)
    if first > last:
        raise ValueError('chapter markers are out of order')
    return data[first:last]


def artifact_path(root, name):
    if not isinstance(name, str) or not ARTIFACT.fullmatch(name):
        raise ValueError('invalid reviewed artifact path')
    original = root / name
    resolved = original.resolve()
    if (original.is_symlink() or resolved.parent != root / name.split('/')[0]
            or not resolved.is_file()):
        raise ValueError('reviewed artifact must be an ordinary public file')
    return resolved


def local_reports(root):
    reports = {}
    for path in sorted((root / 'work/reviews').glob('*.md')):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        reports[digest(data)] = data.decode('utf-8')
    return reports


def full_review(review):
    # Explicit false overrides the legacy scope label. Targeted closures never
    # become full rounds merely because they occur last in a history.
    return (review.get('full_scientific_review') is not False and
            str(review.get('scope', '')).startswith(
                'Full independent scientific/editorial review'))


def zero_blockers(review):
    return (type(review.get('blocking_findings')) is int and
            review['blocking_findings'] == 0 and
            review.get('verdict') == 'yes')


def accepted_evidence(meta, raw_chapter, root, *, reports=None):
    """Require two distinct full approvals bound to all current artifacts.

    reports=None checks public attestations only; a digest-to-text mapping also
    checks report presence and recorded bindings. It does not automate the
    scientific judgment or independently prove reviewer independence.
    """
    root = Path(root).resolve()
    if (meta.get('status') != 'accepted' or
            meta.get('coverage_complete') is not True):
        raise ValueError('accepted status and complete coverage required')
    body = digest(body_bytes(raw_chapter))
    if meta.get('body_sha256') != body:
        raise ValueError('sidecar body digest is stale')
    code_names = [item['path'] for item in meta.get('code', [])]
    artifact_names = meta.get('reviewed_artifacts', [])
    if (not code_names or len(set(code_names)) != len(code_names) or
            len(set(artifact_names)) != len(artifact_names)):
        raise ValueError('nonempty unique code and unique artifacts required')
    if (not any(p.startswith('figures/') for p in artifact_names) or
            not any(p.startswith('assets/') for p in artifact_names)):
        raise ValueError('original figure generator and SVG bindings required')
    codes = {p: digest(artifact_path(root, p).read_bytes())
             for p in code_names}
    artifacts = {p: digest(artifact_path(root, p).read_bytes())
                 for p in artifact_names}
    declared = set(codes) | set(artifacts)
    for name in sorted(declared):
        if not name.endswith('.py'):
            continue
        tree = ast.parse(artifact_path(root, name).read_text(encoding='utf-8'))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module.split('.')[0])
            elif isinstance(node, ast.Import):
                modules.extend(alias.name.split('.')[0]
                               for alias in node.names)
        for module in modules:
            if re.fullmatch(r'ch\d{2}_[a-z0-9_]+', module):
                dependency = f'code/{module}.py'
                if ((root / dependency).is_file() and
                        dependency not in declared):
                    raise ValueError(
                        'unbound imported scientific dependency: '
                        f'{dependency}')
    reviews = meta.get('reviews', [])

    def report_present(review):
        identity = review.get('report_sha256', '')
        if not isinstance(identity, str) or not SHA256.fullmatch(identity):
            return False
        if reports is None:
            return True
        text = reports.get(identity)
        if text is None:
            return False
        bindings = [review.get('body_sha256', '')]
        bindings += list(review.get('code_sha256', {}).values())
        bindings += list(review.get('artifact_sha256', {}).values())
        return all(isinstance(d, str) and SHA256.fullmatch(d) and d in text
                   for d in bindings)

    def direct_bindings(review):
        return (review.get('body_sha256') == body and
                all(review.get('code_sha256', {}).get(p) == d
                    for p, d in codes.items()) and
                all(review.get('artifact_sha256', {}).get(p) == d
                    for p, d in artifacts.items()))

    # The only supported infrastructure exception is an independently reviewed,
    # literal one-line creation of the ignored figure-output directory. Removal
    # that exact line must restore the historical raw generator digest. Bodies,
    # numerical programs, SVGs and all other generator bytes must still match.
    bridges = {}
    for bridge in meta.get('infrastructure_addenda', []):
        name = bridge.get('path', '')
        if (name in bridges or not name.startswith('figures/') or
                name not in artifacts):
            raise ValueError('infrastructure exception must name a generator')
        line = bridge.get('added_line', '')
        if not isinstance(line, str) or not re.fullmatch(
                r" *\(ROOT / 'work/figures'\)\.mkdir"
                r"\(parents=True, exist_ok=True\)\n", line):
            raise ValueError('unsupported infrastructure repair')
        data = artifact_path(root, name).read_bytes()
        encoded = line.encode('utf-8')
        restored = digest(data.replace(encoded, b'', 1))
        if (data.count(encoded) != 1 or
                bridge.get('after_sha256') != digest(data) or
                bridge.get('before_sha256') != restored):
            raise ValueError('infrastructure repair changes undeclared bytes')
        addenda = [r for r in reviews
                   if r.get('full_scientific_review') is False and
                   r.get('report_sha256') == bridge.get('report_sha256') and
                   zero_blockers(r) and direct_bindings(r) and
                   report_present(r)]
        if len(addenda) != 1:
            raise ValueError('infrastructure repair lacks its bound addendum')
        bridges[name] = (bridge['before_sha256'], bridge['after_sha256'])

    def current_bindings(review):
        if review.get('body_sha256') != body:
            return False
        if not all(review.get('code_sha256', {}).get(p) == d
                   for p, d in codes.items()):
            return False
        for name, current in artifacts.items():
            observed = review.get('artifact_sha256', {}).get(name)
            bridged = bridges.get(name) == (observed, current)
            if observed != current and not bridged:
                return False
        return True

    current = [r for r in reviews if current_bindings(r)]
    if any(type(r.get('blocking_findings')) is int and
           r['blocking_findings'] > 0 for r in current):
        raise ValueError('blocking findings remain on the current artifacts')
    approved = [r for r in current if full_review(r) and zero_blockers(r)
                and report_present(r)]
    identities, report_hashes = set(), set()
    for review in approved:
        identity = review.get('reviewer_id') or review.get('review_task')
        if not isinstance(identity, str) or not identity.strip():
            continue
        if review['report_sha256'] in report_hashes:
            continue
        identities.add(identity)
        report_hashes.add(review['report_sha256'])
    if len(identities) < 2 or len(report_hashes) < 2:
        raise ValueError(
            'two distinct full current zero-blocking reviews required')
    return {'body_sha256': body, 'current_full_reviews': len(identities),
            'infrastructure_addenda': len(bridges),
            'private_reports_checked': reports is not None}
