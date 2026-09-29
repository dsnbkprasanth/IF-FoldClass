# Release notes — GitHub-ready package

This release package was checked for clean-clone reproducibility and documentation/data consistency.

## Changes made

- Added `scipy>=1.11` to `requirements.txt`.
- Added Python 3.11+ requirement when no Python requirement was already present.
- Updated `audit_labels.py` so the `results/` directory is created automatically.
- Corrected README descriptions to match the supplied CSV files.
- Clarified that `external_test.csv` currently contains no external-test variants.
- Removed/avoided the undocumented `dual_read` field claim.
- Clarified that `family_profile.csv` contains positions with up to 71 aligned proteins.

## Data counts in this package

- `functional_labels.csv`: 299
- `cross_IF_phenotypes.csv`: 17
- `external_test.csv`: 1
- `struct_burial.csv`: 29195
- `family_profile.csv`: 466

## Important

`external_test.csv` is intentionally not populated with fabricated data. If seven genuine external variants are available, they should be restored with their provenance before claiming external validation in a manuscript or README.
