# Computational Finance 1 — lecture notes

An independent, source-grounded textbook rewrite of the NPTEL course coordinated by IIT Madras, taught by Sourish Das (Chennai Mathematical Institute) and Bikramaditya Datta (IIT Kanpur).

**All 24 chapters are written and scientifically accepted**, covering all 76 recordings with 144 solved exercises. Each chapter has two full independent reviews matching its current scientific artifacts. Final publication verification is recorded separately from chapter acceptance.

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

Install the tested validation dependencies with <code>python -m pip install -r requirements-validation.txt</code>. Browser validation additionally needs a compatible local Chrome or Edge installation; see the validator for its configuration. The embedded-code checker discovers all existing chapters when no filenames are supplied and performs syntax/style checks. Scientific review separately executes complete programs and embedded examples with their documented setup. Some development and review evidence is intentionally local in ignored <code>work/</code>. Raw recordings, machine transcripts, and downloaded reference documents are not redistributed. Public metadata records acceptance and artifact digests; a fresh clone does not include private review reports.

## Verification and evidence

The complete course includes Monte Carlo sampling, uncertain-parameter inference, Asian options, and a final comparison of numerical methods and error sources. Public execution and reader reports state their verification scopes. Source reading means sequential machine-transcript reading with selected audio/video checks; it does not mean a full audiovisual audit. Paired R/Python labs are mapped by substantive topic, and documentary language correspondence is distinguished from actual execution.

See [WORKFLOW.md](WORKFLOW.md) for production conventions, [release validation](release-validation.json) for the complete-course checks, and [HANDOFF.md](HANDOFF.md) for the latest publication state. A public clone can rebuild the reader and run programs without raw source material. The stricter development command <code>python tools/validate_book.py --browser --release</code> also requires the private reports retained on the authoring workstation; their absence from a public clone is intentional.
