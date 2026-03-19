# Missing Papers

Papers that could not be downloaded programmatically. All are behind paywalls (SSRN 403, Springer, ACM DL) or temporarily unavailable servers.

## SSRN-gated (download manually via browser)

| # | Paper | SSRN ID | Target Filename |
|---|-------|---------|----------------|
| 18 | Abada, Harrington, Lambin, Meylahn (2025) "Where Are We Now? A Survey on Algorithmic Collusion" | 4891033 | abada_survey_2025 |
| 19 | Abada, Lambin, Tchakarov (2024) "Collusion by Mistake: Does Algorithmic Sophistication Drive Supra-competitive Profits?" EJOR | 4099361 | abada_lambin_2024 |
| 21 | Epivent & Lambin (2024) "Reward-Punishment Schemes and Algorithmic Collusion" Economics Letters | 4227229 | epivent_lambin_2024 |

## Publisher-gated

| # | Paper | Source | Target Filename |
|---|-------|--------|----------------|
| 38 | Lorscheid, Heine & Meyer (2012) "Opening the 'Black Box' of Simulations" CMOT | Springer (DOI: 10.1007/s10588-012-9131-8) | lorscheid_2012 |
| -- | Balseiro et al. (2021) "The Landscape of Auto-bidding Auctions" EC '21 | ACM DL / SSRN 3785579 | balseiro_landscape_2021 |

## Server temporarily down

| # | Paper | Source | Target Filename |
|---|-------|--------|----------------|
| 12 | Deng et al. (2024) "The Autobidder's Dilemma" NeurIPS '24 | NeurIPS proceedings (connection refused); no arXiv preprint | deng_dilemma_2024 |

## Also noted: wrong content in existing files

| File | Problem |
|------|---------|
| leme_dilemma_2024.md | Contains physics paper (Adler function), not the expected economics paper |
| balseiro_landscape_2021.md | Now contains "Non-Clairvoyant Dynamic Mechanism Design" (wrong paper by overlapping authors) |
| lambin_2024.md | Contains Oxford workshop presentation slides, not the full SSRN working paper (SSRN:4498926) |

## Instructions

For SSRN papers, visit `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=XXXXXXX` in a browser, download the PDF, save to `docs/transcriptions/FILENAME.pdf`, then run:
```bash
cd docs/transcriptions && /opt/homebrew/bin/docling FILENAME.pdf --output .
```
