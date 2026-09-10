# AGENTS.md

## Scope

This repository studies whether Taiwan-wide margin financing and short-selling behaviour explains or predicts subsequent 0050 returns. GitHub is the single source of truth.

## Rules

- Use Python 3.11 and relative paths (`pathlib`).
- Keep research logic in `src/`; notebooks only orchestrate.
- Never hard-code credentials. Colab private-repo access uses `GITHUB_TOKEN` from Colab Secrets.
- Signal data is complete after d0 close; the earliest tradable benchmark is the next valid 0050 open (O1).
- Never forward-fill holidays. Do not infer unobserved suspension intervals.
- Preserve raw and suspension-adjusted short signals.
- Do not overwrite outputs. Each run is timestamped in Asia/Taipei and records the Git commit.
- Run Stage 0 diagnostics before research. Stop on missing 0050, missing required columns, or failed reconciliation.
- Do not claim effectiveness without real data, adequate samples, FDR, parameter and annual robustness, and bias review.
- Never commit generated outputs, credentials, or large source datasets.

## Change workflow

Inspect repository/branch/status, make the smallest necessary change, run focused tests, review diff, then commit/push only when explicitly requested.
