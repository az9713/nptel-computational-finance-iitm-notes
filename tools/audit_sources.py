"""Inventory acquired source evidence without pretending it has been read."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / 'source-manifest.json').read_text(encoding='utf-8'))
    ledger = ROOT / 'work/sources/read-ledger.json'
    read = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {'recordings_read': []}
    read_ids = {r['playlist_index'] for r in read['recordings_read'] if r['status'] == 'read in full'}
    results = []
    for recording in source['recordings']:
        number, vid = recording['playlist_index'], recording['video_id']
        transcript = ROOT / f'work/sources/transcripts/{number:02d}-{vid}.json'
        audio = [p for p in (ROOT / 'work/sources/audio').glob(vid+'.*')
                 if p.suffix in {'.webm', '.m4a', '.mp3'}]
        record = {'playlist_index': number, 'video_id': vid, 'audio_acquired': bool(audio),
                  'transcript_acquired': transcript.exists(),
                  'sequential_reading': 'read in full' if number in read_ids else 'pending',
                  'flags': []}
        if transcript.exists():
            data = json.loads(transcript.read_text(encoding='utf-8'))
            segments = data['segments']
            record['transcript_sha256'] = hashlib.sha256(transcript.read_bytes()).hexdigest()
            record['transcript_model'] = data.get('model')
            record['segments'] = len(segments)
            record['first_speech_second'] = min(s['start'] for s in segments) if segments else None
            record['last_speech_second'] = max(s.get('end', s['start']+s.get('duration', 0)) for s in segments) if segments else None
            record['audio_duration_seconds'] = data.get('duration_seconds')
            if not segments:
                record['flags'].append('Empty transcript')
            elif record['last_speech_second'] < 0.85 * recording['duration_seconds']:
                record['flags'].append('Late recording coverage requires inspection')
            duration = record['audio_duration_seconds']
            if duration is not None and abs(duration-recording['duration_seconds']) > 3:
                record['flags'].append('Audio and playlist durations differ by more than three seconds')
        results.append(record)
    report = {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'recording_count': len(results), 'audio_count': sum(r['audio_acquired'] for r in results),
              'transcript_count': sum(r['transcript_acquired'] for r in results),
              'sequentially_read_count': len(read_ids), 'recordings': results,
              'scope': 'Acquisition inventory and structural transcript diagnostics; reading status is separately recorded. This is not audiovisual or mathematical verification.'}
    (ROOT / 'source-validation-summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"Sources: audio={report['audio_count']}/{len(results)}, "
          f"transcripts={report['transcript_count']}/{len(results)}, "
          f"read={report['sequentially_read_count']}/{len(results)}")
    print('Flags:', [(r['playlist_index'], r['flags']) for r in results if r['flags']])


if __name__ == '__main__':
    main()
