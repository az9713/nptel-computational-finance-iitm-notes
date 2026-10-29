# HANDOFF — Computational Finance 1

**Read this first each new session, then AGENTS.md for standing conventions.**
Checkpoint date: 2026-10-28. This is an incomplete-course checkpoint, not the final release.

## Current state

- Chapters 1–21 are written and accepted, each with two full independent scientific reviews. Chapter 19 also has a narrowly scoped figure-generator startup repair addendum; its two full scientific rounds remain intact.
- All 76 recordings have local machine transcripts. Recordings 1–74 are fully read. Recording 75's initial read was interrupted/truncated: restart it from character zero. Recording 76 has not been read.
- Browser validation passed through Chapter 21 and the four navigation pages, including a fresh run after rebuilding navigation for this checkpoint. All 21 complete programs and the embedded-code checker passed again. completion.json now correctly reports 21 accepted. Final release validation and GitHub Pages publication remain pending.
- Chapters 22–24 have not been authored. The user expects completion of all recordings and eventual verified publication, with no approval loop.
- Remote: https://github.com/az9713/nptel-computational-finance-iitm-notes, branch main. The complete Chapters 1–21 checkpoint landed in commit e672bb84179cf8ab37f226fdf13da5d1eaa4366e; local HEAD and remote main were verified equal. This handoff follow-up records that verified checkpoint. No GitHub Pages site has been enabled.
- Local ignored work/ holds essential sources, outlines, reviews, primary-reference research, video inspections, and execution evidence. It survives /clear on this workstation but is intentionally absent from GitHub. Do not delete it.

## Next task

**Read recordings 75 and 76 completely, record their scope, then finalize Chapter 22's outline before authoring.** Start with the matching files in work/sources/transcripts/; read in chunks of at most roughly 5,500 characters to avoid tool-output truncation. Update work/sources/read-ledger.json only after completing each source. Check the recording 74 video and recover the actual Metropolis–Hastings table before freezing Chapter 23.

After source reading, complete Chapter 22, then 23, then 24 using the full independent review gates. Finish whole-book checks, documentation, source-free export, commit/push, Pages publication, and live verification. Do not stop at a partial chapter milestone or mark the overall goal complete early.

## Where to read things

- WORKFLOW.md and AGENTS.md — production and verification conventions.
- BOOK-STATE.yaml, chapter-manifest.json, individual *.meta.json — live chapter state and actual source coverage.
- work/plan/architecture.md — evolving mixed-recording partitions; work/plan/chNN-outline.md — outlines for accepted chapters.
- work/sources/read-ledger.json — full-read evidence and detailed corrections for final recordings.
- work/reviews/chNN-r1.md and chNN-r2.md — full scientific review evidence. Chapter 19's additional report is work/reviews/ch19-reproduction-addendum.md.
- work/checkpoint-resume-notes.md — unresolved scientific issues and final-block details that must be carried into the remaining chapters.

## Source partitions to preserve

- Chapter 22 primary recordings: 66, 67, 68, 72, 73.
- Chapter 23 primary recordings: 69, 70, 74. Recording 69: European MC through about 1045 seconds. Recording 74: European MC 15–560 seconds and MH/utility 756–1681 seconds.
- Chapter 24 primary recordings: 71, 75, 76. Secondary Asian material: recording 69 at 1045–1792 seconds, recording 74 at 560–756 seconds. Revisit lab partitions after fully reading 75 and 76.

## Scratch and reproduction

- No persistent JavaScript globals are required. Recreate fs, child_process, crypto imports and read files from disk. Transcript strings previously held in Node memory must be reloaded.
- tools/build_book.py preserves bodies between chapter markers and generates navigation, references, code listings, and completion.json. Do not regenerate accepted prose.
- Public figures/chNN_*.py generate public SVG figures and local diagnostics. Public code/chNN_*.py contain numerical programs. These generators are durable; local work/figures and browser screenshots are diagnostic outputs.
- work/update_author_state.py and work/accept_unchanged_chapter.py are local helpers. Inspect before use; do not rerun acceptance on an already accepted chapter.
- work/fetch_video63.py, fetch_video70.py, fetch_video73.py, fetch_video74.py use yt-dlp's visionos route for low-resolution visual verification. Videos and frame provenance live in work/sources/video-inspection/. Keep caches/temp/profile paths under work/.
- Python 3.13 is installed; Chrome is installed. Prefer the validator's explicit Chrome executable configuration if Playwright's managed/channel browser is missing. Cross-program Python imports executed with -I must add the program's parent directory to sys.path.

## Remaining release issues

- Correct punctuation mojibake in references.json without silently changing accepted scientific bodies.
- Final validation reports must bind the final rebuilt files; a self-hashing publication report must exclude its own entry.
- The final public export must exclude work/, raw sources, credentials, and private local paths. Preserve their local evidence rather than deleting it.
- A fresh clone lacks private review reports; document that boundary. Do not present historical browser checks as proof of a later changed file.
