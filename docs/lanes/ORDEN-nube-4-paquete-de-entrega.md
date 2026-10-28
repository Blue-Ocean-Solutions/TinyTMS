# ORDEN nube-4 · Submission package and the receipts-backed video outline

**Lane name:** `Nube-4` · **Model:** Opus 5.5 · **Cap:** 50 turns or $15 · **Branch:** `nube/4-entrega`

## Goal
Assemble everything the hackathon submission needs from what the repository already proves. Words only; no code
beyond a small checker.

## Deliverables
1. `docs/SUBMISSION.md`: the fields a Cerebral Valley / Devpost-style form asks for, filled from this repo:
   project name (tinyTMS.IO), one-line pitch, the problem (a small carrier's dock check-in: paper, photo and voice
   that do not match), what was built during the event (list by PR number, read `git log --merges`), Meta tools
   used (SAM 3.1, Muse Spark 1.3, Muse Voice Transcribe through the Meta Model API; Muse Glimmer 30B open
   weights on Apple silicon through mlx_vlm), how to run it (three commands), license, team, and a
   "what is not here yet" section copied honestly from README.
2. `docs/VIDEO-OUTLINE.md`: a 3-minute outline in five beats (the dock problem · three witnesses · the judge says
   HOLD and the driver hears why · pull the plug: local mode on the Mac · what is real and what is next). **Every number
   in the outline links to a receipt file on the same line**; where there is no receipt yet (Saturday's real boxes,
   the driver's own voice), write `[to be measured on Saturday]` instead of a number.
3. `scripts/check_docs_numbers.py`: extends the README numbers guard to every `docs/*.md`; wire it into CI.

## Tests (first)
- The checker flags a Markdown line with `ms`/`FPS`/`%`/`tok/s` and no `receipts/` link; passes on the committed docs.

## Do not
- Do not invent metrics, prizes, judges' opinions or hardware the repo does not show. If a fact is not in the repo
  or in `evals/PROCEDENCIA.md`, leave a bracketed placeholder for a human.
- Nothing about customs authorities or government filings.
