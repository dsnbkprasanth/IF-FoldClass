"""Build data/family_profile.csv: IF-family residue frequencies per vimentin-aligned column.

Reuses the vimentin-anchored alignment already stored in data/struct_burial.csv (acc,pos -> vim_pos),
so no new alignment is computed. Run once after `ifclass.py download`:  python make_family_profile.py
"""
import collections, math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
AA = "ACDEFGHIKLMNPQRSTVWY"

seq = pd.read_csv(ROOT / "data/raw/uniprot_if.tsv", sep="\t").set_index("acc").seq.to_dict()
b = pd.read_csv(ROOT / "data/struct_burial.csv")[["acc", "pos", "vim_pos"]].dropna()
b = b[[a in seq and 1 <= p <= len(seq[a]) for a, p in zip(b.acc, b.pos)]]
b["aa"] = [seq[a][int(p) - 1] for a, p in zip(b.acc, b.pos)]

rows = []
for vp, g in b.groupby("vim_pos"):
    c = collections.Counter(g.aa)
    n = sum(c.values())
    f = {a: c.get(a, 0) / n for a in AA}
    rows.append(dict(vim_pos=vp, n_fam=n,
                     fam_entropy=-sum(v * math.log(v) for v in f.values() if v > 0),
                     **{f"f_{a}": f[a] for a in AA}))

out = pd.DataFrame(rows).sort_values("vim_pos")
out.to_csv(ROOT / "data/family_profile.csv", index=False)
print(f"{len(out)} aligned columns, median {int(out.n_fam.median())} IF proteins per column, "
      f"entropy {out.fam_entropy.min():.2f}-{out.fam_entropy.max():.2f}")
assert out.n_fam.min() >= 1 and out[[f"f_{a}" for a in AA]].sum(axis=1).round(6).eq(1).all()
