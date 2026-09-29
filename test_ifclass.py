"""Quick self-check of the parsing / heptad logic. Run: python test_ifclass.py"""
import numpy as np
from ifclass import clinvar_label, coil_segments, domain_map, heptad, parse_pchange

assert parse_pchange("NM_001927.4(DES):c.1360C>T (p.Arg454Trp)") == "R454W"
assert parse_pchange("p.Arg454Trp") == "R454W"
assert parse_pchange("p.Arg454Ter") is None          # nonsense
assert parse_pchange("p.Arg454TrpfsTer5") is None    # frameshift
assert parse_pchange("p.Glu353dup") is None
assert clinvar_label("Likely pathogenic") == 1 and clinvar_label("Benign/Likely benign") == 0
assert np.isnan(clinvar_label("Conflicting classifications of pathogenicity"))
assert np.isnan(clinvar_label("Uncertain significance"))

# heptad: LxxLxxx repeat -> every L lands on a or d
seq = "MS" + "LKELEKA" * 6 + "GG"
segs = [(2, 44)]
reg = heptad(seq, segs)
assert all(reg[i] in "ad" for i in range(2, 44) if seq[i] == "L"), reg
assert reg[0] == "none"

feats = [dict(type="Domain", desc="IF rod", start=3, end=44), dict(type="Region", desc="Coil 1A", start=3, end=20)]
dom = domain_map(feats, 46)
assert dom[0] == "head" and dom[5] == "coil1A" and dom[30] == "rod" and dom[45] == "tail"
assert coil_segments(dom, feats) == [(2, 20)]
print("all checks passed")
