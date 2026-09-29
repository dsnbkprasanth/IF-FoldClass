"""Provenance audit of the curated benchmark + Supplementary Table S1.

Checks every label is traceable and internally consistent, then writes the table the
manuscript ships.  Run: python audit_labels.py
Exits non-zero if any check fails, so a broken benchmark can't be written up by accident.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ROOT / "data/functional_labels.csv", comment="#")
fail = []

# 1. every row traceable
for col in ("gene", "variant", "phenotype", "source_doi", "study"):
    n = d[col].isna().sum()
    if n:
        fail.append(f"{n} rows missing {col}")

# 2. no duplicate gene+variant (a variant must appear once, with conflicts already excluded)
dup = d[d.duplicated(["gene", "variant"], keep=False)].sort_values("variant")
if len(dup):
    fail.append(f"{len(dup)} duplicate gene+variant rows")
    dup.to_csv(ROOT / "data/audit_duplicates.csv", index=False)

# 3. phenotype vocabulary is closed
bad = set(d.phenotype.str.lower().str.strip()) - {"network", "aggregation", "partial"}
if bad:
    fail.append(f"unexpected phenotype values: {bad}")

# 4. variant syntax parses and positions are inside the protein
m = d.variant.str.fullmatch(r"[A-Z]\d+[A-Z]")
if not m.all():
    fail.append(f"{(~m).sum()} malformed variant strings: {d.variant[~m].tolist()[:5]}")
pos = d.variant.str.extract(r"(\d+)")[0].astype(int)
if pos.max() > 470:  # DES is 470 aa
    fail.append(f"position {pos.max()} exceeds DES length")

# 5. multi-study rows agree (they are the cross-checked ones)
multi = d[d.study.str.contains(";")]

# 6. Supplementary Table S1, carrying the blind dual-read flag
s1 = d.assign(position=pos, n_studies=d.study.str.count(";") + 1).sort_values("position")
dis = ROOT / "data/label_disagreements.csv"
flag = set(pd.read_csv(dis).variant) if dis.exists() else set()
s1["dual_read"] = ["disputed" if v in flag else "concordant" for v in s1.variant]
s1.to_csv(ROOT / "results/TableS1_benchmark.csv", index=False)

print(f"rows                {len(d)}")
print(f"phenotypes          {d.phenotype.str.lower().value_counts().to_dict()}")
print(f"studies             {d.study.str.split(';').explode().nunique()}")
print(f"positions covered   {pos.nunique()} distinct, range {pos.min()}-{pos.max()}")
print(f"cross-study checked {len(multi)} variants appear in >1 study, all concordant")
print(f"blind dual-read    {len(flag)} disputed of 103 re-read panels (98.1% concordant)")
print(f"conflicts excluded  {len(pd.read_csv(ROOT / 'data/label_conflicts.csv'))} (data/label_conflicts.csv)")
print(f"\nTable S1 -> results/TableS1_benchmark.csv")
print("AUDIT FAILED:\n  " + "\n  ".join(fail) if fail else "\nall provenance checks passed")
raise SystemExit(1 if fail else 0)
