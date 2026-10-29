"""Retrieve a selected public source frame into ignored work/ for inspection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import yt_dlp

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', type=int)
    parser.add_argument('second', type=int)
    args = parser.parse_args()
    records = json.loads((ROOT / 'source-manifest.json').read_text(encoding='utf-8'))['recordings']
    record = next(r for r in records if r['playlist_index'] == args.recording)
    if not 0 <= args.second <= record['duration_seconds']:
        raise ValueError('Frame must be within the source recording')
    opts = {'quiet': True, 'cachedir': str(ROOT / 'work/cache/yt-dlp'),
            'extractor_args': {'youtube': {'player_client': ['visionos']}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(record['url'], download=False)
    choices = [f for f in info['formats'] if f.get('ext') == 'mp4'
               and f.get('vcodec', 'none').startswith('avc1')
               and 720 <= (f.get('height') or 0) <= 1080]
    if not choices:
        raise RuntimeError('No suitable source video format')
    fmt = max(choices, key=lambda f: (f.get('height', 0), f.get('tbr', 0)))
    output = ROOT / f"work/sources/frames/{args.recording:02d}-{record['video_id']}-{args.second:04d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([shutil.which('ffmpeg') or 'ffmpeg', '-hide_banner', '-loglevel', 'error',
                             '-ss', str(args.second), '-i', fmt['url'], '-frames:v', '1',
                             '-update', '1', '-y', str(output)],
                            capture_output=True, timeout=90)
    log = output.with_suffix('.log')
    log.write_bytes(result.stderr)
    if result.returncode or not output.exists():
        raise RuntimeError(f'Source frame unavailable; diagnostics saved in {log.name}')
    print(f'Captured {output.relative_to(ROOT).as_posix()} from format {fmt["format_id"]}')


if __name__ == '__main__':
    main()
