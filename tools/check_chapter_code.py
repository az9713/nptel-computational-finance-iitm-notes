"""Execute declared teaching programs and retain current-input evidence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter', type=int)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'chapter-manifest.json').read_text(encoding='utf-8'))
    report_path = ROOT / 'chapter-code-validation.json'
    previous = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {'runs': []}
    retained = [r for r in previous['runs'] if args.chapter and r['chapter'] != args.chapter]
    runs = []
    for chapter in manifest['chapters']:
        if args.chapter and chapter['number'] != args.chapter:
            continue
        sidecar = ROOT / chapter['file'].replace('.html', '.meta.json')
        if not sidecar.exists():
            continue
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        for item in meta.get('code', []):
            name = item['path']
            if not re.fullmatch(r'code/ch\d{2}_[a-z0-9_]+\.py', name):
                raise ValueError('invalid code path')
            original = ROOT / name
            path = original.resolve()
            if original.is_symlink() or path.parent != ROOT / 'code':
                raise ValueError('program must be an ordinary file in code/')
            temp_root = ROOT / 'work/code-runs'
            temp_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=temp_root) as directory:
                env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                           TEMP=directory, TMP=directory, MPLCONFIGDIR=directory)
                started = datetime.now(timezone.utc).isoformat()
                result = subprocess.run([sys.executable, '-I', '-X', 'utf8',
                                         str(path)], cwd=directory,
                                        env=env,
                                        capture_output=True, timeout=180)
            capture_error = None
            try:
                stdout = result.stdout.decode('utf-8')
                stderr = result.stderr.decode('utf-8')
            except UnicodeDecodeError as error:
                capture_error = str(error)
                stdout = result.stdout.decode('utf-8', errors='replace')
                stderr = result.stderr.decode('utf-8', errors='replace')
            run = {'chapter': chapter['number'], 'path': name,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'executed_at_utc': started, 'python': platform.python_version(),
                   'platform': platform.system(), 'exit_code': result.returncode,
                   'stdout': stdout, 'stderr': stderr,
                   'capture_error': capture_error,
                   'passed': result.returncode == 0 and capture_error is None,
                   'scope': 'Program assertions and observed output; not formal scientific verification.'}
            runs.append(run)
            print(f'{name}: {"PASS" if run["passed"] else "FAIL"}')
    report = {'runs': sorted(retained + runs, key=lambda r: (r['chapter'], r['path'])),
              'evidence_boundary': 'Dated executions bind program input digests; acceptance of mathematics is recorded separately.'}
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not runs:
        raise SystemExit('No programs selected')
    if not all(r['passed'] for r in runs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
