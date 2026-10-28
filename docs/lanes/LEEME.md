# Lanes (work orders) for cloud sessions

Each `ORDEN-*.md` in this folder is a self-contained work order for one cloud session (claude.ai/code) on this
public repository. The session reads the order first, works on its own branch, and answers with a pull request
against `main`. Nothing else: no comments elsewhere, no pushes to `main` (it is protected: CI `test` required).

Contract for every lane:

1. **Read first:** this file, your `ORDEN-*.md`, `README.md`, `pyproject.toml`, and the tests you touch.
2. **Tests before code.** Write the failing test, watch it fail, then the minimal code. Run the whole suite
   (`pip install -e ".[dev]" && pytest -q`) in a fresh environment before opening the PR; the number of passed
   tests goes in the PR body.
3. **Dependencies:** every import must be declared in `pyproject.toml`. Prefer the standard library.
4. **Receipts and numbers:** no performance figure in any Markdown unless it links to a file in `receipts/` on the
   same line (`tests/test_readme_numbers.py` enforces it). Never invent a measurement; never write `time.sleep`
   to imitate one. Receipts are never edited or re-signed.
5. **Paths and secrets:** no machine paths (`/Users/`, `/private/`, `/home/`, `/tmp/`...) in receipts, test data or
   docs; no API keys anywhere; `MODEL_API_KEY` comes from the environment only and you do not have one.
6. **Scope:** this is a freight check-in tool. Nothing about customs authorities, government filings or
   enforcement. Spanish and English are both welcome in user-facing text.
7. **Hackathon rule (Meta Global AI Developer Hackathon 2026, Indie track):** everything in the demo must have been
   built during the event (2026-10-26 to 2026-11-04). Say in the PR what is new.
8. **PR body:** what you built, what you verified (commands + outcomes), what you did NOT do, and open questions.
   Commit trailer: `Co-Authored-By: <your lane name> (<model>) <noreply@anthropic.com>`.
9. **Budget:** stop and open the PR with what you have when you reach the turn or dollar cap written in the order.
