# Production and verification

The course uses a source-grounded author → independent review → repair → second independent review → browser validation workflow, adapted from the owner's NPTEL and Stanford lecture-note projects.

1. Acquire and audit source metadata and local transcripts. Read every assigned recording in full. Recover ambiguous visual inputs from actual video frames where necessary.
2. Map mixed recordings to their actual topic spans. Establish the chapter outline before drafting; distinguish source content, corrections, and original extensions.
3. Derive results and compute numerical benchmarks. Write programs and original figure generators. Author chapter sections directly in HTML, with assumptions, limitations, worked examples, and complete exercise solutions.
4. Run two full independent scientific reviews. Reports bind the manuscript body, programs, and figure artifacts. Repair findings and re-review changed scientific material. Acceptance requires current evidence with no blocking findings.
5. Build navigation and reports, execute programs and embedded Python, and validate the offline reader in a browser, including mathematics, mobile layout, controls, and printing.
6. After all planned chapters are accepted, complete whole-book integrity and publication checks, export a source-free deliverable, inspect staged files, commit/push, and verify the live publication.

`chapter-manifest.json` defines the thematic plan. `source-manifest.json` records the 76 playlist entries. Individual chapter sidecars record actual coverage and scientific acceptance. `completion.json` is generated; rebuild it after acceptance changes. Execution and browser reports have narrower scopes than scientific review.

Raw source material and detailed editorial evidence remain in ignored `work/`. Public references and timestamp links support provenance without redistributing the recordings or transcripts. Cached source material on the original workstation survives a conversation clear, but is not recoverable from a fresh Git clone.
