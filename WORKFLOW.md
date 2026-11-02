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

The builder checks two distinct full current-artifact approvals, including every
declared program, imported scientific dependency, generator and SVG. A latest
targeted closure cannot replace a stale full round. Body hashes retain exact
between-marker bytes, including line endings. Current blocking findings prevent
acceptance even if other current reports approve.

The release validator additionally requires the original private reports to be
present and match their recorded hashes and artifact bindings. Public-clone
builds verify the published attestations; they cannot independently reperform
the private editorial judgment. This boundary is intentional.

A separately identified infrastructure addendum can bridge only the supported
one-line creation of the ignored figure-output directory. Removing that exact
line must restore the historical generator hash. It cannot bridge changed
scientific bodies, numerical programs, or SVGs, and it does not count as another
full scientific round. Chapter19 records this exception explicitly.

Run the gate's synthetic regression checks with
`python -I tools/test_review_evidence.py`. They test stale/targeted histories,
duplicate reviewers, missing bindings/reports, raw body changes, current
blockers, imported dependencies, and the narrowly limited infrastructure case;
they do not establish a chapter's scientific correctness.
