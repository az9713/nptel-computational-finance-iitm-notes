"""Workspace-contained media acquisition and local machine transcription."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work'
PIPELINE = None
os.environ['HF_HOME'] = str(WORK / 'models' / 'hf')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['TEMP'] = os.environ['TMP'] = str(WORK / 'tmp')
(WORK / 'tmp').mkdir(parents=True, exist_ok=True)

def records(start, end):
    manifest = json.loads((ROOT / 'source-manifest.json').read_text(encoding='utf-8'))
    return [r for r in manifest['recordings'] if start <= r['playlist_index'] <= end]

def download_one(record):
    import yt_dlp
    target = WORK / 'sources' / 'audio'
    target.mkdir(parents=True, exist_ok=True)
    vid = record['video_id']
    if any(p.suffix in {'.webm', '.m4a', '.mp3'} for p in target.glob(vid + '.*')):
        return f"EXISTS {record['playlist_index']:02d} {vid}"
    options = {'format': os.environ.get('CF_AUDIO_FORMAT', '251/140/bestaudio/best'),
               'outtmpl': str(target / '%(id)s.f%(format_id)s.%(ext)s'),
               'quiet': True, 'noprogress': True, 'noplaylist': True,
               'retries': 5, 'socket_timeout': 25,
               'cachedir': str(WORK / 'cache' / 'yt-dlp'), 'js_runtimes': {'node': {}},
               'extractor_args': {'youtube': {'player_client': ['visionos']}}}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([record['url']])
    return f"DOWNLOADED {record['playlist_index']:02d} {vid}"

def transcribe(selected, device, batch_size):
    global PIPELINE
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    compute = 'int8_float16' if device == 'cuda' else 'int8'
    if PIPELINE is None:
        print(f'Loading small.en on {device} ({compute})', flush=True)
        model = WhisperModel('small.en', device=device, compute_type=compute,
                             download_root=str(WORK / 'models'), cpu_threads=4)
        PIPELINE = BatchedInferencePipeline(model=model)
    pipeline = PIPELINE
    out_dir = WORK / 'sources' / 'transcripts'
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in selected:
        stem = f"{record['playlist_index']:02d}-{record['video_id']}"
        dest = out_dir / (stem + '.json')
        if dest.exists():
            print(f'EXISTS {stem}', flush=True)
            continue
        files = [p for p in (WORK / 'sources' / 'audio').glob(record['video_id'] + '.*')
                 if p.suffix in {'.webm', '.m4a', '.mp3'}]
        if not files:
            print(f'WAITING_AUDIO {stem}', flush=True)
            continue
        started = time.monotonic()
        print(f'TRANSCRIBING {stem}', flush=True)
        segments, info = pipeline.transcribe(str(files[0]), language='en',
                                             beam_size=3, batch_size=batch_size,
                                             vad_filter=True)
        data = [{'start': s.start, 'end': s.end, 'text': s.text.strip(),
                 'avg_logprob': s.avg_logprob, 'no_speech_prob': s.no_speech_prob}
                for s in segments]
        result = {'video_id': record['video_id'], 'source': 'local-audio-transcription',
                  'model': 'faster-whisper small.en', 'language': 'en',
                  'is_generated': True, 'duration_seconds': info.duration,
                  'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': time.monotonic() - started, 'segments': data}
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        dest.with_suffix('.txt').write_text('\n'.join(
            f"[{s['start']:.1f}–{s['end']:.1f}] {s['text']}" for s in data), encoding='utf-8')
        print(f"TRANSCRIBED {stem} segments={len(data)} seconds={result['elapsed_seconds']:.1f}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['download', 'transcribe'])
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=999)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args()
    selected = records(args.start, args.end)
    if args.indices:
        selected = [r for r in selected if r['playlist_index'] in args.indices]
    if args.action == 'download':
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(download_one, r): r for r in selected}
            for future in as_completed(jobs):
                r = jobs[future]
                try:
                    print(future.result(), flush=True)
                except Exception as exc:
                    print(f"DOWNLOAD_FAILED {r['playlist_index']:02d}: {type(exc).__name__} {exc}", flush=True)
    else:
        if not args.watch:
            transcribe(selected, args.device, args.batch_size)
        else:
            idle_since = time.monotonic()
            while True:
                pending = [r for r in selected if not (WORK / 'sources/transcripts' /
                           f"{r['playlist_index']:02d}-{r['video_id']}.json").exists()]
                if not pending:
                    print(f'ALL_TRANSCRIBED count={len(selected)}', flush=True)
                    break
                ready = [r for r in pending if any(p.suffix in {'.webm', '.m4a', '.mp3'}
                         for p in (WORK / 'sources/audio').glob(r['video_id'] + '.*'))]
                if ready:
                    transcribe(ready, args.device, args.batch_size)
                    idle_since = time.monotonic()
                else:
                    print(f'WAITING_FOR_DOWNLOADS remaining={len(pending)}', flush=True)
                    if time.monotonic() - idle_since > 1800:
                        raise SystemExit('No new audio for 30 minutes; sources remain incomplete.')
                    time.sleep(20)
