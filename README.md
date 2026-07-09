# Diabetes — Type 1 & Type 2 (Pediatric)

Consolidated analysis for the pediatric diabetes study (Type 1 and Type 2), bringing together
work previously split across multiple platform-specific repositories.

## Repository structure

| Path | Origin | Description |
|------|--------|-------------|
| `src/`, `src_t1d/`, `omop_cohort_master/`, `data/`, `clinical_notes/` | (this repo) | Primary analysis code (OMOP cohort, T1D/T2D scripts). Scripts only — no patient data. |
| `data_T2D_Sep2025/` | (this repo) | T2D characteristic tables and crosswalk scripts. |
| `palantir/t1d/` | `Palantir_T1D_Scripts` | Type 1 diabetes scripts developed in Palantir Foundry. |
| `palantir/t2d/` | `Palantir_T2D_Scripts` | Type 2 diabetes scripts developed in Palantir Foundry. |
| `t1d-legacy-scripts/` | `T1D_project` | Earlier standalone T1D scripts, kept for reference. |

## Consolidation note (2026-07)

This repository is the **primary** for all diabetes work. The following repositories were merged
in and then archived as read-only legacy copies (renamed with a `-legacy` suffix):

- `T1D_project` → `t1d-project-legacy`
- `Palantir_T1D_Scripts` → `palantir-t1d-scripts-legacy`
- `Palantir_T2D_Scripts` → `palantir-t2d-scripts-legacy`

The pre-consolidation state of this repository is preserved on the `legacy_branch` branch.

## Data policy

Code and analysis only. No PHI or patient-level data is committed to this repository.
