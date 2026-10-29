# Project conventions

Read HANDOFF.md first, then this file. The user authorized autonomous completion of all 76 recordings, defaults consistent with their NPTEL/Stanford notes, and eventual commit/push/publication. Work only in the current project directory. No additional approval loop is required.

- Read assigned recordings fully and research primary references before fixing a chapter outline. Record coverage, source corrections, and original additions explicitly.
- The lead authors prose directly, one HTML section at a time. Preserve TeX and the BEGIN CHAPTER / END CHAPTER markers. Do not delegate prose generation.
- Compute printed numerical results before prose. Supply executable complete programs, original figures with generators, and six fully solved exercises per chapter.
- Require two full independent science-editor reviews per chapter, repair findings, and require zero blocking findings on the current artifacts. Independent review agents are authorized for these gates. Reviews must check source scope, derivations, numbers, solutions, actual figure rendering, and isolated execution of the complete and embedded programs. Bind body/code/figure digests. Interrupted review is not approval.
- Accepted body edits invalidate prior body-bound reviews. Preserve historical review evidence. A narrowly scoped infrastructure repair may receive a separately identified addendum, without pretending it is a new full scientific round.
- Rebuild with tools/build_book.py. Execute tools/check_chapter_code.py and tools/check_embedded_code.py; validate with tools/validate_book.py --browser. Only use --release after the entire course and release evidence are complete.
- Keep raw audio, transcripts, downloaded research, review reports, local paths, and credentials out of public commits. Explicitly stage intended files and inspect the staged tree before push. The ignored work/ directory is persistent local working evidence, not disposable noise.
- Use model/effort appropriate to each task. Use independent reviewers for the required gates; keep authoring under the lead. Do not mark the overall goal complete before all chapters and verified publication are complete.
