# Computational Finance 1 — lecture notes

An independent, source-grounded textbook rewrite of the NPTEL course coordinated by IIT Madras, taught by Sourish Das (Chennai Mathematical Institute) and Bikramaditya Datta (IIT Kanpur).

**Work in progress: 21 of 24 planned chapters are accepted.** Each accepted chapter has two full independent scientific reviews. This repository is a checkpoint; the complete-course release and GitHub Pages publication remain pending.

- [Open the reader](index.html)
- [Source map](source-map.html) and [completion inventory](completion.json)
- [Executable programs](code-lab.html) and [primary references](references.html)
- [Original playlist](https://www.youtube.com/playlist?list=PLyqSpQzTE6M_HO74zZkM7xBvLEtE1uIZ2)
- [Official course](https://nptel.ac.in/courses/130106003)

The 76 recordings are organized into 24 thematic chapters. Notes include derivations, worked examples, original figures, reproducible programs, and six solved exercises per completed chapter. Source statements, corrections, and original extensions are distinguished in the chapter text and metadata. These are independent study notes, not an official NPTEL publication.

## Read and reproduce

Open `index.html` in a browser, or serve this directory with `python -m http.server 8000`. Mathematics assets are bundled locally. Install the pinned scientific dependencies with `python -m pip install -r requirements.txt`, then run a named program from `code/`.

From the repository root:

```text
python tools/build_book.py
python tools/check_chapter_code.py
python tools/check_embedded_code.py
python tools/validate_book.py --browser
```

Browser validation additionally needs Playwright and a compatible local Chrome installation; see the validator for its configuration. Some development and review evidence is intentionally local in ignored `work/`. Raw recordings, machine transcripts, and downloaded reference documents are not redistributed. Public metadata records acceptance and artifact digests; a fresh clone does not include private review reports.

## Remaining scope

Chapters 22–24 cover sampling and random walks, Monte Carlo pricing with uncertain parameters, and Asian options with the final numerical workflow. Whole-book integrity checks, release documentation, and live-site verification will follow. See [WORKFLOW.md](WORKFLOW.md) for production conventions and [HANDOFF.md](HANDOFF.md) for the current resume point.
