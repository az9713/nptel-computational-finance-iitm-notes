"""Acquire the playlist and captions; raw source artifacts stay in ignored work/."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import yt_dlp

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
PLAYLIST = 'https://www.youtube.com/playlist?list=PLyqSpQzTE6M_HO74zZkM7xBvLEtE1uIZ2'

def enumerate_playlist():
    with yt_dlp.YoutubeDL({'extract_flat': True, 'quiet': True, 'ignoreerrors': False,
                          'cachedir': str(WORK / 'cache' / 'yt-dlp')}) as ydl:
        info = ydl.extract_info(PLAYLIST, download=False)
    (WORK / 'sources').mkdir(parents=True, exist_ok=True)
    (WORK / 'sources' / 'playlist.raw.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    records = [{'playlist_index': i, 'video_id': e['id'], 'title': e['title'],
                'duration_seconds': e.get('duration'), 'url': f"https://www.youtube.com/watch?v={e['id']}"}
               for i, e in enumerate(info['entries'], 1)]
    manifest = {'playlist_url': PLAYLIST, 'title': info['title'], 'channel': info.get('channel'),
                'acquired_at_utc': datetime.now(timezone.utc).isoformat(),
                'recording_count': len(records), 'recordings': records}
    (ROOT / 'source-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for e in records:
        print(f"{e['playlist_index']:02d} {e['video_id']} {e['duration_seconds']} {e['title']}")

def acquire_captions(start=1, end=999):
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    manifest = json.loads((ROOT / 'source-manifest.json').read_text(encoding='utf-8'))
    target = WORK / 'sources' / 'transcripts'
    target.mkdir(parents=True, exist_ok=True)
    for e in manifest['recordings']:
        if not start <= e['playlist_index'] <= end:
            continue
        path = target / f"{e['playlist_index']:02d}-{e['video_id']}.json"
        if path.exists():
            print(f"SKIP {e['playlist_index']:02d}", flush=True)
            continue
        try:
            tracks = api.list(e['video_id'])
            try:
                track = tracks.find_manually_created_transcript(['en'])
            except Exception:
                track = tracks.find_transcript(['en'])
            data = track.fetch().to_raw_data()
            raw = {'video_id': e['video_id'], 'language': track.language_code,
                   'is_generated': track.is_generated, 'segments': data}
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding='utf-8')
            text_path = path.with_suffix('.txt')
            text_path.write_text('\n'.join(f"[{s['start']:.1f}–{s['start']+s['duration']:.1f}] {s['text']}" for s in data), encoding='utf-8')
            print(f"OK {e['playlist_index']:02d} segments={len(data)} end={max(s['start']+s['duration'] for s in data):.1f}", flush=True)
        except Exception as exc:
            print(f"FAIL {e['playlist_index']:02d} {type(exc).__name__}: {str(exc)[:220]}", flush=True)
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['enumerate', 'captions'])
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=999)
    args = parser.parse_args()
    if args.action == 'enumerate':
        enumerate_playlist()
    else:
        acquire_captions(args.start, args.end)
