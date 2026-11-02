"""Synthetic gate regression tests, not scientific manuscript reviews."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'tools'))
from review_evidence import accepted_evidence, body_bytes, digest  # noqa: E402


class ReviewGate(unittest.TestCase):
    def setUp(self):
        (PROJECT / 'work').mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(
            dir=PROJECT / 'work', prefix='review-gate-test-')
        self.root = Path(self.directory.name).resolve()
        for name in ('code', 'figures', 'assets'):
            (self.root / name).mkdir()
        self.files = {
            'code/ch01_example.py': b'value = 1\n',
            'figures/ch01_example.py': (
                b'from pathlib import Path\nROOT = Path(".")\nx = 1\n'),
            'assets/ch01-example.svg': (
                b'<svg xmlns="http://www.w3.org/2000/svg"/>'),
        }
        for name, content in self.files.items():
            (self.root / name).write_bytes(content)
        self.raw = (b'<!-- BEGIN CHAPTER -->\r\n<p>Body</p>\r\n'
                    b'<!-- END CHAPTER -->')
        self.meta = {
            'status': 'accepted', 'coverage_complete': True,
            'body_sha256': digest(body_bytes(self.raw)),
            'code': [{'path': 'code/ch01_example.py'}],
            'reviewed_artifacts': ['figures/ch01_example.py',
                                   'assets/ch01-example.svg'],
            'reviews': []}
        self.reports = {}
        for i in (1, 2):
            codes = {'code/ch01_example.py':
                     digest(self.files['code/ch01_example.py'])}
            artifacts = {p: digest(self.files[p])
                         for p in self.meta['reviewed_artifacts']}
            text = '\n'.join([str(i), self.meta['body_sha256'],
                              *codes.values(), *artifacts.values()])
            report_hash = digest(text.encode())
            self.reports[report_hash] = text
            self.meta['reviews'].append({
                'scope': 'Full independent scientific/editorial review',
                'review_task': f'reviewer-{i}', 'verdict': 'yes',
                'blocking_findings': 0,
                'body_sha256': self.meta['body_sha256'],
                'code_sha256': codes, 'artifact_sha256': artifacts,
                'report_sha256': report_hash})

    def tearDown(self):
        self.directory.cleanup()

    def check(self, **kwargs):
        return accepted_evidence(self.meta, self.raw, self.root, **kwargs)

    def test_two_full_current_reports(self):
        state = self.check(reports=self.reports)
        self.assertEqual(state['current_full_reviews'], 2)

    def test_targeted_closure_cannot_replace_stale_full_round(self):
        self.meta['reviews'][0]['body_sha256'] = '0' * 64
        targeted = deepcopy(self.meta['reviews'][1])
        targeted.update(scope='Targeted repair closure',
                        full_scientific_review=False)
        self.meta['reviews'].append(targeted)
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_program_invalidates_both(self):
        (self.root / 'code/ch01_example.py').write_bytes(b'value = 2\n')
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_artifact_binding_is_not_approval(self):
        artifacts = self.meta['reviews'][0]['artifact_sha256']
        artifacts.pop('assets/ch01-example.svg')
        with self.assertRaises(ValueError):
            self.check()

    def test_same_reviewer_does_not_count_twice(self):
        self.meta['reviews'][1]['review_task'] = 'reviewer-1'
        with self.assertRaises(ValueError):
            self.check()

    def test_current_blocker_is_not_hidden_by_other_approvals(self):
        blocking = deepcopy(self.meta['reviews'][0])
        blocking.update(blocking_findings=1, verdict='unapproved')
        self.meta['reviews'].append(blocking)
        with self.assertRaises(ValueError):
            self.check()

    def test_private_report_required_when_requested(self):
        self.check()
        with self.assertRaises(ValueError):
            self.check(reports={})

    def test_explicit_nonfull_flag_overrides_full_scope(self):
        self.meta['reviews'][0]['full_scientific_review'] = False
        with self.assertRaises(ValueError):
            self.check()

    def test_raw_line_endings_are_bound(self):
        changed = self.raw.replace(b'\r\n', b'\n')
        self.assertNotEqual(body_bytes(changed), body_bytes(self.raw))
        with self.assertRaises(ValueError):
            accepted_evidence(self.meta, changed, self.root)

    def test_unlisted_imported_dependency_rejected(self):
        dependency = self.root / 'code/ch02_dependency.py'
        dependency.write_text('x = 1\n', encoding='utf-8')
        (self.root / 'code/ch01_example.py').write_text(
            'from ch02_dependency import x\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unbound imported'):
            self.check()

    def test_narrow_bridge_and_changed_scientific_content(self):
        name = 'figures/ch01_example.py'
        line = ("    (ROOT / 'work/figures').mkdir"
                "(parents=True, exist_ok=True)\n")
        # A syntactically valid one-line addition inside an existing loop.
        old = (b'from pathlib import Path\nROOT = Path(".")\n'
               b'for i in range(1):\n    x = 1\n')
        (self.root / name).write_bytes(old)
        for review in self.meta['reviews']:
            review['artifact_sha256'][name] = digest(old)
        current = old + line.encode()
        (self.root / name).write_bytes(current)
        addendum = deepcopy(self.meta['reviews'][0])
        addendum.update(scope='Targeted reproduction-only addendum',
                        full_scientific_review=False, report_sha256='a' * 64)
        addendum['artifact_sha256'][name] = digest(current)
        self.meta['reviews'].append(addendum)
        self.meta['infrastructure_addenda'] = [{
            'path': name, 'added_line': line,
            'before_sha256': digest(old), 'after_sha256': digest(current),
            'report_sha256': 'a' * 64}]
        state = self.check()
        self.assertEqual(state['current_full_reviews'], 2)
        self.assertEqual(state['infrastructure_addenda'], 1)
        (self.root / name).write_bytes(current.replace(b'x = 1', b'x = 2'))
        with self.assertRaises(ValueError):
            self.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)
