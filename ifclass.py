"""IF-FoldClass: structure/PLM-aware ML for intermediate-filament (IF) missense variants.

Steps (run in order, or `all`):
  python ifclass.py download   # UniProt, AlphaMissense, ClinVar, gnomAD, AlphaFold  (~2 GB)
  python ifclass.py features   # ESM-2 + heptad + domain + pLDDT for EVERY possible missense (~650k rows)
  python ifclass.py train      # Stage A (pathogenicity, ~thousands labels, leave-gene-out CV)
                               # Stage B (filament-assembly phenotype, your curated labels)
                               # Stage C (score all variants) + figures
"""
import argparse, gzip, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RAW, PROC, RES = ROOT / "data/raw", ROOT / "data/processed", ROOT / "results"
FUNC_LABELS = ROOT / "data/functional_labels.csv"
CROSS_IF = ROOT / "data/cross_IF_phenotypes.csv"  # non-DES phenotypes, test only
FAMPROF = ROOT / "data/family_profile.csv"  # IF-family residue frequencies per vimentin-aligned column
STRUCT = ROOT / "data/struct_burial.csv"  # per-residue interface burial from the 8RVE cryo-EM vimentin
# filament (Eibauer 2024), mapped onto each IF protein by homology. bur_dimer = buried by the coiled-coil
# partner; bur_higher = buried by every other chain in the filament lattice (tetramer/lateral/longitudinal).

AM_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz"
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
GNOMAD_API = "https://gnomad.broadinstitute.org/api"
GNOMAD_BENIGN_AF = 1e-4  # ponytail: AF cut-off for benign proxy; tune / sensitivity-test in paper

AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = dict(Ala="A", Arg="R", Asn="N", Asp="D", Cys="C", Gln="Q", Glu="E", Gly="G", His="H", Ile="I",
             Leu="L", Lys="K", Met="M", Phe="F", Pro="P", Ser="S", Thr="T", Trp="W", Tyr="Y", Val="V")
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
VOL = dict(zip(AA, [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6]))
CHG = {a: (1 if a in "KR" else -1 if a in "DE" else 0) for a in AA}
HELIX = dict(zip(AA, [0, .68, .69, .40, .54, 1.0, .61, .41, .26, .21, .24, .65, 3.16, .39, .21, .50, .66, .61, .49, .53]))  # Pace & Scholtz 1998
HYD_CORE = set("LIVMAF")
DOMAINS = ["head", "coil1A", "linker1", "coil1B", "linker12", "coil2", "rod", "tail", "other"]
HEPTAD = list("abcdefg") + ["none"]
CAT = {"domain": DOMAINS, "heptad": HEPTAD, "wt": list(AA), "mut": list(AA)}
N_PC = 32
# Stage B labels cluster by position, so position features become a shortcut (learned "rod residue >128 = network",
# failed on coil 2). MECH = substitution-level features only, no rel_pos / rel_rod_pos / domain / ESM PCs.
MECH = ["esm_llr", "esm_wt_logp", "esm_entropy", "plddt", "d_hyd", "d_vol", "d_chg", "d_helix",
        "to_pro", "from_gly", "chg_reversal", "heptad", "wt", "mut", "bur_dimer", "bur_higher",
        "fam_entropy", "fam_freq_wt", "fam_freq_mut", "fam_logodds"]

PCH = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?![a-z])")


# ----------------------------------------------------------------------------- helpers
def parse_pchange(s):
    """'NM_..(DES):c.1360C>T (p.Arg454Trp)' or 'p.Arg454Trp' -> 'R454W'; None for non-missense."""
    m = PCH.search(s or "")
    if not m:
        return None
    w, mu = THREE.get(m[1]), THREE.get(m[3])
    return f"{w}{m[2]}{mu}" if w and mu and w != mu else None


def clinvar_label(sig):
    s = str(sig).lower()
    if "conflicting" in s or "uncertain" in s:
        return np.nan
    if "pathogenic" in s and "benign" not in s:
        return 1.0
    if "benign" in s and "pathogenic" not in s:
        return 0.0
    return np.nan


def domain_map(feats, L):
    """UniProt features -> per-residue IF domain label."""
    lab = np.array(["other"] * L, dtype=object)
    rod = [f for f in feats if f["type"] == "Domain" and "rod" in f["desc"].lower()]
    for f in rod:
        lab[f["start"] - 1:f["end"]] = "rod"
    for f in feats:
        if f["type"] != "Region":
            continue
        d = f["desc"].lower().replace(" ", "")
        name = ("head" if "head" in d else "tail" if "tail" in d else "coil1A" if "coil1a" in d else
                "coil1B" if "coil1b" in d else "coil2" if d.startswith("coil2") else
                "linker12" if "linker12" in d or "linkerl12" in d else "linker1" if d.startswith("linker") else None)
        if name:
            lab[f["start"] - 1:f["end"]] = name
    if rod:
        s, e = min(f["start"] for f in rod), max(f["end"] for f in rod)
        lab[:s - 1][lab[:s - 1] == "other"] = "head"
        lab[e:][lab[e:] == "other"] = "tail"
    return lab


def coil_segments(dom, feats):
    """Contiguous coil stretches (0-based [s, e)); falls back to UniProt 'Coiled coil' / rod."""
    is_coil = np.isin(dom, ["coil1A", "coil1B", "coil2"])
    if not is_coil.any():
        is_coil = np.zeros(len(dom), bool)
        for f in feats:
            if f["type"] == "Coiled coil":
                is_coil[f["start"] - 1:f["end"]] = True
        if not is_coil.any():
            is_coil = dom == "rod"
    segs, i = [], 0
    while i < len(is_coil):
        if is_coil[i]:
            j = i
            while j < len(is_coil) and is_coil[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def heptad(seq, segs):
    """Assign heptad a–g per coil segment: phase that puts most L/I/V/M/A/F at a & d.
    ponytail: one register per segment, ignores stutters/skips (coil 2 stutter); swap in
    DeepCoil/CCHMM registers if reviewers ask."""
    reg = np.array(["none"] * len(seq), dtype=object)
    for s, e in segs:
        ph = max(range(7), key=lambda p: sum(seq[i] in HYD_CORE for i in range(s, e) if (i - s + p) % 7 in (0, 3)))
        for i in range(s, e):
            reg[i] = "abcdefg"[(i - s + ph) % 7]
    return reg


def get(url, method="get", tries=5, **kw):
    """HTTP with retry on network errors / 429 / 5xx; other 4xx raise at once."""
    for k in range(tries):
        try:
            r = requests.request(method, url, timeout=300, **kw)
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
                return r
            print(f"  retry {k + 1}/{tries} {url}: HTTP {r.status_code}")
        except requests.ConnectionError as e:
            print(f"  retry {k + 1}/{tries} {url}: {e}")
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def fetch(url, dest):
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        bar = tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True, desc=dest.name)
        with open(tmp, "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
                bar.update(len(c))
        bar.close()
    tmp.rename(dest)
    return dest


def load_proteins():
    p = pd.read_csv(RAW / "uniprot_if.tsv", sep="\t")
    p["features"] = p["features"].map(json.loads)
    return p


# ----------------------------------------------------------------------------- 1. download
def dl_uniprot():
    out = RAW / "uniprot_if.tsv"
    if out.exists():
        return
    q = "xref:pfam-PF00038 AND organism_id:9606 AND reviewed:true"  # all human IF-rod proteins
    r = get("https://rest.uniprot.org/uniprotkb/search",
            params=dict(query=q, format="json", size=500, fields="accession,gene_primary,sequence,ft_domain,ft_region,ft_coiled"))
    rows = []
    for e in r.json()["results"]:
        g = (e.get("genes") or [{}])[0].get("geneName", {}).get("value", e["primaryAccession"])
        feats = []
        for f in e.get("features", []):
            s, t = f["location"]["start"].get("value"), f["location"]["end"].get("value")
            if s and t:
                feats.append(dict(type=f["type"], desc=f.get("description", ""), start=s, end=t))
        rows.append(dict(acc=e["primaryAccession"], gene=g, seq=e["sequence"]["value"], features=json.dumps(feats)))
    df = pd.DataFrame(rows)
    assert len(df) >= 40, f"UniProt returned only {len(df)} IF proteins, check query"
    df.to_csv(out, sep="\t", index=False)
    print(f"UniProt: {len(df)} human IF proteins, {df.seq.str.len().sum():,} residues")


def dl_alphamissense(accs):
    out = RAW / "alphamissense_if.tsv"
    if out.exists():
        return
    gz = fetch(AM_URL, RAW / "AlphaMissense_aa_substitutions.tsv.gz")  # ~1.2 GB
    accs, n = set(accs), 0
    with gzip.open(gz, "rt") as f, open(out, "w") as o:
        o.write("acc\tvariant\tam_score\tam_class\n")
        for line in tqdm(f, desc="AlphaMissense filter", unit=" lines", mininterval=5):
            if line[0] != "#" and line.split("\t", 1)[0] in accs:
                o.write(line)
                n += 1
    print(f"AlphaMissense: {n:,} IF substitutions")


def dl_clinvar(genes):
    out = RAW / "clinvar_if.tsv"
    if out.exists():
        return
    gz = fetch(CLINVAR_URL, RAW / "variant_summary.txt.gz")  # ~400 MB
    genes, keep = set(genes), []
    for ch in pd.read_csv(gz, sep="\t", dtype=str, chunksize=500_000, low_memory=False):
        sig = next(c for c in ("ClinicalSignificance", "GermlineClassification") if c in ch.columns)
        ch = ch[(ch.Assembly == "GRCh38") & (ch.Type == "single nucleotide variant") & ch.GeneSymbol.isin(genes)
                & ~ch.ReviewStatus.str.contains("no assertion", na=False)]
        keep.append(ch[["GeneSymbol", "Name", sig, "ReviewStatus"]].set_axis(["gene", "name", "sig", "review"], axis=1))
    cv = pd.concat(keep)
    cv["variant"] = cv.name.map(parse_pchange)
    cv["y"] = cv.sig.map(clinvar_label)
    cv = cv.dropna(subset=["variant"])
    # same protein change from different SNVs: keep only if labels agree
    agg = cv.groupby(["gene", "variant"]).agg(y=("y", lambda s: s.iloc[0] if s.nunique() == 1 and s.notna().all() else np.nan),
                                              sig=("sig", "first")).reset_index()
    agg.to_csv(out, sep="\t", index=False)
    print(f"ClinVar: {len(agg):,} IF missense ({int((agg.y == 1).sum())} P/LP, {int((agg.y == 0).sum())} B/LB, rest VUS)")


def dl_gnomad(genes):
    out = RAW / "gnomad_if.tsv"
    if out.exists():
        return
    q = """query($g:String!){gene(gene_symbol:$g, reference_genome:GRCh38){
             variants(dataset:gnomad_r4){consequence hgvsp exome{ac an} genome{ac an}}}}"""
    rows = []
    for g in tqdm(genes, desc="gnomAD v4"):
        try:
            js = get(GNOMAD_API, method="post", json=dict(query=q, variables=dict(g=g))).json()
        except Exception as e:
            print(f"  gnomAD skip {g}: {e}")
            continue
        vs = ((js.get("data") or {}).get("gene") or {}).get("variants") or []
        for v in vs:
            if v.get("consequence") != "missense_variant":
                continue
            ac = sum((v.get(k) or {}).get("ac", 0) or 0 for k in ("exome", "genome"))
            an = sum((v.get(k) or {}).get("an", 0) or 0 for k in ("exome", "genome"))
            var = parse_pchange(v.get("hgvsp"))
            if var and an:
                rows.append(dict(gene=g, variant=var, af=ac / an))
        time.sleep(6)  # public API rate limit
    df = pd.DataFrame(rows, columns=["gene", "variant", "af"]).groupby(["gene", "variant"], as_index=False).af.max()
    df.to_csv(out, sep="\t", index=False)
    print(f"gnomAD: {len(df):,} IF missense variants")


def dl_alphafold(accs):
    d = RAW / "alphafold"
    d.mkdir(exist_ok=True)
    for acc in tqdm(accs, desc="AlphaFold"):
        if (d / f"{acc}.pdb").exists():
            continue
        try:
            meta = get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}").json()[0]
            fetch(meta["pdbUrl"], d / f"{acc}.pdb")
        except Exception as e:
            print(f"  no AlphaFold model for {acc}: {e}")


def cmd_download(a):
    RAW.mkdir(parents=True, exist_ok=True)
    dl_uniprot()
    p = load_proteins()
    dl_alphafold(p.acc)
    dl_clinvar(p.gene)
    dl_gnomad(list(p.gene))
    dl_alphamissense(p.acc)


# ----------------------------------------------------------------------------- 2. features
def plddt(acc, L):
    f = RAW / "alphafold" / f"{acc}.pdb"
    out = np.full(L, np.nan, np.float32)
    if f.exists():
        for line in open(f):
            if line.startswith("ATOM") and line[12:16] == " CA ":
                i = int(line[22:26]) - 1
                if 0 <= i < L:
                    out[i] = float(line[60:66])
    return out


def esm_scores(seqs, model_name, device):
    """WT-marginal ESM-2 log-probs (L x 20) + last-layer residue embeddings, 1022-aa sliding windows.
    ponytail: WT-marginal, not masked-marginal (L forward passes/protein); add masked if GPU time allows."""
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"ESM-2 {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).eval().to(device)
    aa_ids = tok.convert_tokens_to_ids(list(AA))
    W, out = 1022, {}
    for acc, seq in tqdm(seqs.items(), desc="ESM-2"):
        L = len(seq)
        starts = [0] if L <= W else list(range(0, L - W, W // 2)) + [L - W]
        logp, emb, best = np.zeros((L, 20), np.float32), None, np.full(L, -1)
        for st in starts:
            sub = seq[st:st + W]
            ids = tok(sub, return_tensors="pt").to(device)
            with torch.no_grad():
                o = model(**ids, output_hidden_states=True)
            lp = torch.log_softmax(o.logits[0, 1:-1].float(), -1)[:, aa_ids].cpu().numpy()
            h = o.hidden_states[-1][0, 1:-1].float().cpu().numpy()
            emb = np.zeros((L, h.shape[1]), np.float32) if emb is None else emb
            for j in range(len(sub)):
                c = min(j, len(sub) - 1 - j)  # prefer the window where residue is most central
                if c > best[st + j]:
                    best[st + j], logp[st + j], emb[st + j] = c, lp[j], h[j]
        out[acc] = (logp, emb)
    return out


def cmd_features(a):
    from sklearn.decomposition import PCA
    PROC.mkdir(parents=True, exist_ok=True)
    p = load_proteins()
    esm = esm_scores(dict(zip(p.acc, p.seq)), a.esm, a.device)
    allemb = np.vstack([e for _, e in esm.values()])
    pca = PCA(N_PC, random_state=0).fit(allemb[np.random.default_rng(0).permutation(len(allemb))[:50_000]])
    print(f"ESM embedding PCA-{N_PC}: {pca.explained_variance_ratio_.sum():.2%} variance")

    frames = []
    for r in tqdm(p.itertuples(), total=len(p), desc="variant table"):
        seq, L = r.seq, len(r.seq)
        dom = domain_map(r.features, L)
        hep = heptad(seq, coil_segments(dom, r.features))
        pl = plddt(r.acc, L)
        logp, emb = esm[r.acc]
        pcs = pca.transform(emb).astype(np.float32)
        ent = -(np.exp(logp) * logp).sum(1)
        rod = np.where(np.isin(dom, ["rod", "coil1A", "linker1", "coil1B", "linker12", "coil2"]))[0]
        rs, re_ = (rod.min(), rod.max()) if len(rod) else (0, L - 1)
        pos_idx, mut_aa = [], []
        for i, w in enumerate(seq):
            if w in KD:
                for m in AA:
                    if m != w:
                        pos_idx.append(i)
                        mut_aa.append(m)
        i = np.array(pos_idx)
        wt = np.array([seq[k] for k in i])
        mu = np.array(mut_aa)
        wi = np.array([AA.index(x) for x in wt])
        mi = np.array([AA.index(x) for x in mu])
        f = pd.DataFrame(dict(
            acc=r.acc, gene=r.gene, pos=i + 1, wt=wt, mut=mu,
            domain=dom[i], heptad=hep[i],
            rel_rod_pos=np.clip((i - rs) / max(re_ - rs, 1), -0.5, 1.5).astype(np.float32),
            rel_pos=(i / L).astype(np.float32), prot_len=L,
            plddt=pl[i],
            esm_llr=logp[i, mi] - logp[i, wi], esm_wt_logp=logp[i, wi], esm_entropy=ent[i],
            d_hyd=[KD[m] - KD[w] for w, m in zip(wt, mu)],
            d_vol=[VOL[m] - VOL[w] for w, m in zip(wt, mu)],
            d_chg=[CHG[m] - CHG[w] for w, m in zip(wt, mu)],
            d_helix=[HELIX[m] - HELIX[w] for w, m in zip(wt, mu)],
            to_pro=(mu == "P").astype(np.int8), from_gly=(wt == "G").astype(np.int8),
            chg_reversal=np.array([CHG[w] * CHG[m] == -1 for w, m in zip(wt, mu)], np.int8),
        ))
        f["variant"] = f.wt + f.pos.astype(str) + f.mut
        f = pd.concat([f, pd.DataFrame(pcs[i], columns=[f"esm_pc{k}" for k in range(N_PC)])], axis=1)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    for c, cats in CAT.items():
        df[c] = pd.Categorical(df[c], categories=cats)

    # attach labels / comparators
    cv = pd.read_csv(RAW / "clinvar_if.tsv", sep="\t")[["gene", "variant", "y"]].rename(columns={"y": "clinvar_y"})
    gn = pd.read_csv(RAW / "gnomad_if.tsv", sep="\t")
    am = pd.read_csv(RAW / "alphamissense_if.tsv", sep="\t")[["acc", "variant", "am_score"]]
    df = df.merge(cv, on=["gene", "variant"], how="left").merge(gn, on=["gene", "variant"], how="left") \
           .merge(am, on=["acc", "variant"], how="left")
    df["y_path"] = df.clinvar_y
    benign_proxy = df.y_path.isna() & (df.af >= GNOMAD_BENIGN_AF)
    df.loc[benign_proxy, "y_path"] = 0.0
    df["label_src"] = np.where(df.clinvar_y.notna(), "clinvar", np.where(benign_proxy, "gnomad", ""))
    df.to_parquet(PROC / "if_variants.parquet", index=False)
    lab = df.y_path.notna()
    print(f"\nVariant table: {len(df):,} variants x {df.shape[1]} cols across {df.gene.nunique()} IF genes")
    print(f"Stage-A labels: {int(lab.sum()):,} ({int((df.y_path == 1).sum()):,} pathogenic / {int((df.y_path == 0).sum()):,} benign)")


# ----------------------------------------------------------------------------- 3. train
def feat_cols(df):
    # prot_len dropped: it identifies the gene, so the model learned gene-level label prevalence (shortcut)
    skip = {"acc", "gene", "pos", "variant", "clinvar_y", "af", "am_score", "y_path", "label_src", "y_func", "prot_len"}
    return [c for c in df.columns if c not in skip]


def lgbm(**kw):
    import lightgbm as lgb
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, class_weight="balanced", verbose=-1, random_state=0)
    return lgb.LGBMClassifier(**{**p, **kw})


def lgbm_small():
    """Stage B has ~40-300 labels: default min_child_samples=20 never splits -> constant 0.5 predictions."""
    return lgbm(n_estimators=300, num_leaves=7, min_child_samples=3, subsample=1.0)


def logreg():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))


def dummies(X):
    return pd.get_dummies(X, columns=list(CAT), dtype=np.float32)


def oof(make, X, y, groups, n_splits=5):
    """Out-of-fold predictions, folds split by group (gene or domain) so no leakage."""
    from sklearn.model_selection import GroupKFold
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(n_splits, len(set(groups)))).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


def cv_threshold(s, y, groups, n_splits=5):
    """Held-out calls using a cut-off chosen on training groups only (maximises balanced accuracy)."""
    from sklearn.model_selection import GroupKFold
    out = np.zeros(len(y), bool)
    for tr, te in GroupKFold(n_splits=min(n_splits, len(set(groups)))).split(s, y, groups):
        grid = np.unique(np.quantile(s[tr], np.linspace(0.01, 0.99, 99)))
        ba = [((s[tr][y[tr] == 1] > t).mean() + (s[tr][y[tr] == 0] <= t).mean()) / 2 for t in grid]
        out[te] = s[te] > grid[int(np.argmax(ba))]
    return out


def bal_acc_ci(y, c, n=1000):
    rng = np.random.default_rng(0)
    f = lambda i: ((c[i][y[i] == 1]).mean() + (~c[i][y[i] == 0]).mean()) / 2
    bs = [f(i) for i in (rng.integers(0, len(y), len(y)) for _ in range(n)) if 0 < y[i].sum() < len(i)]
    return f(np.arange(len(y))), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_ci(y, s, n=1000):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    rng = np.random.default_rng(0)
    auc, ap = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            auc.append(roc_auc_score(y[i], s[i]))
            ap.append(average_precision_score(y[i], s[i]))
    return dict(n=int(ok.sum()), auroc=roc_auc_score(y, s), auroc_lo=np.percentile(auc, 2.5), auroc_hi=np.percentile(auc, 97.5),
                auprc=average_precision_score(y, s), auprc_lo=np.percentile(ap, 2.5), auprc_hi=np.percentile(ap, 97.5))


def compare(y, scores, name):
    rows = [dict(predictor=k, **boot_ci(y, np.asarray(v, float))) for k, v in scores.items()]
    t = pd.DataFrame(rows).round(3)
    t.to_csv(RES / f"metrics_{name}.csv", index=False)
    print(f"\n== {name} ==\n{t.to_string(index=False)}")
    return t


def roc_plot(y, scores, path, title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    plt.figure(figsize=(5, 5))
    for k, s in scores.items():
        s = np.asarray(s, float)
        ok = ~np.isnan(s)
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        plt.plot(fpr, tpr, label=f"{k} (AUROC {roc_auc_score(y[ok], s[ok]):.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(title); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def cmd_train(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "if_variants.parquet")
    if STRUCT.exists():  # cryo-EM interface burial; NaN outside the modelled vimentin filament
        df = df.merge(pd.read_csv(STRUCT)[["acc", "pos", "vim_pos", "bur_dimer", "bur_higher"]], on=["acc", "pos"], how="left")
    else:
        df["bur_dimer"] = df["bur_higher"] = df["vim_pos"] = np.nan
    # IF-family conservation: ESM-2 is trained on all proteins, so nothing else in the feature set is
    # family-specific. Frequencies come from the same vimentin-anchored alignment used for burial.
    df["fam_entropy"] = df["fam_freq_wt"] = df["fam_freq_mut"] = df["fam_logodds"] = np.nan
    if FAMPROF.exists() and df.vim_pos.notna().any():
        fp = pd.read_csv(FAMPROF).set_index("vim_pos")
        ent = df.vim_pos.map(fp.fam_entropy)
        def freq(aa_col):
            idx = pd.MultiIndex.from_arrays([df.vim_pos, "f_" + df[aa_col].astype(str)])
            st = fp[[c for c in fp.columns if c.startswith("f_")]].stack()
            return pd.Series(st.reindex(idx).values, index=df.index)
        fw, fm = freq("wt"), freq("mut")
        eps = 1e-3
        df["fam_entropy"], df["fam_freq_wt"], df["fam_freq_mut"] = ent.values, fw.values, fm.values
        df["fam_logodds"] = np.log((fm + eps) / (fw + eps)).values
        print(f"family profile: {int(df.fam_entropy.notna().sum()):,}/{len(df):,} variants covered")
    F = feat_cols(df)
    print(f"{len(df):,} variants, {len(F)} features")

    func = None
    if FUNC_LABELS.exists():
        func = pd.read_csv(FUNC_LABELS, comment="#").dropna(subset=["gene", "variant", "phenotype"])
        func["y_func"] = (func.phenotype.str.lower().str.strip() != "network").astype(float)
        func = func.merge(df, on=["gene", "variant"], how="inner")
        print(f"Functional labels matched: {len(func)}")
    func_keys = set(zip(func.gene, func.variant)) if func is not None else set()

    # ---- Stage A: pathogenicity across all IF genes, leave-gene-out CV
    A = df[df.y_path.notna()].reset_index(drop=True)
    yA, gA = A.y_path.values.astype(int), A.gene.values
    pA_lgb = oof(lgbm, A[F], yA, gA)
    pA_lr = oof(logreg, dummies(A[F]), yA, gA)
    sA = {"IF-FoldClass (LightGBM)": pA_lgb, "Logistic regression": pA_lr,
          "ESM-2 zero-shot (-LLR)": -A.esm_llr.values, "AlphaMissense": A.am_score.values}
    compare(yA, sA, "stageA_pathogenicity_leave-gene-out")
    roc_plot(yA, sA, RES / "fig_roc_stageA.png", "Stage A: IF pathogenicity (leave-gene-out)")
    # within-gene AUROC: shows the model ranks variants inside a gene, not just which gene is "disease"
    from sklearn.metrics import roc_auc_score
    pg = [dict(gene=g, n=len(s), n_path=int(s.y_path.sum()), auroc_ifclass=roc_auc_score(s.y_path, s.p),
               auroc_alphamissense=roc_auc_score(s.y_path[s.am_score.notna()], s.am_score.dropna()),
               auroc_esm=roc_auc_score(s.y_path, -s.esm_llr))
          for g, s in A.assign(p=pA_lgb).groupby("gene") if s.y_path.nunique() == 2 and len(s) >= 30]
    pd.DataFrame(pg).round(3).to_csv(RES / "stageA_within_gene_auroc.csv", index=False)
    print(pd.DataFrame(pg).round(3).to_string(index=False))
    A.assign(oof_lgbm=pA_lgb, oof_logreg=pA_lr)[["gene", "variant", "label_src", "y_path", "oof_lgbm", "oof_logreg", "am_score", "esm_llr"]] \
        .to_csv(RES / "stageA_oof_predictions.csv", index=False)

    # final Stage A model; exclude functional-test variants so Stage B transfer eval is clean
    trainA = A[[k not in func_keys for k in zip(A.gene, A.variant)]]
    mA = lgbm().fit(trainA[F], trainA.y_path.astype(int))
    df["ifclass_path_score"] = mA.predict_proba(df[F])[:, 1]

    # interpretability: LightGBM native SHAP contributions
    contrib = mA.predict(A[F].sample(min(len(A), 5000), random_state=0), pred_contrib=True)
    contrib = pd.DataFrame(np.abs(contrib[:, :-1]), columns=F).mean()
    grp = contrib.groupby(lambda c: "ESM-2 embedding (PCs)" if c.startswith("esm_pc") else c).sum().sort_values()
    grp.to_csv(RES / "feature_importance_stageA.csv")
    plt.figure(figsize=(6, 6)); grp.tail(20).plot.barh(); plt.xlabel("mean |SHAP| (log-odds)")
    plt.title("Stage A feature attribution"); plt.tight_layout(); plt.savefig(RES / "fig_importance.png", dpi=300); plt.close()

    # heptad x domain mechanistic map (all saturation variants)
    coil = df[df.heptad != "none"]
    hm = coil.pivot_table(index="domain", columns="heptad", values="ifclass_path_score", aggfunc="mean", observed=True)
    hm = hm.drop(columns="none", errors="ignore").dropna(how="all")
    hm.to_csv(RES / "heptad_domain_mean_score.csv")
    plt.figure(figsize=(6, 3)); plt.imshow(hm.values, cmap="magma", aspect="auto")
    plt.xticks(range(hm.shape[1]), hm.columns); plt.yticks(range(hm.shape[0]), hm.index); plt.colorbar(label="mean score")
    plt.title("Predicted damage by heptad position (all IF coils)"); plt.tight_layout(); plt.savefig(RES / "fig_heptad.png", dpi=300); plt.close()

    # ---- Stage B: filament-assembly phenotype (curated functional benchmark)
    if func is not None and len(func) >= 20 and func.y_func.nunique() == 2:
        from scipy.stats import binomtest, mannwhitneyu
        from sklearn.metrics import roc_auc_score
        yB = func.y_func.values.astype(int)
        func["ifclass_path_score"] = mA.predict_proba(func[F])[:, 1]  # mA never saw these variants
        FB, FM = F + ["ifclass_path_score"], MECH + ["ifclass_path_score"]
        study = func.study.astype(str).str.split(";").str[0].values if "study" in func else func.source_doi.values
        schemes = {"position": func.pos.values, "domain": func.domain.astype(str).values, "study": study}
        FS = FM + ["am_score"]  # does cell-structure information add anything on top of AlphaMissense?
        FN = [c for c in FM if c not in ("bur_dimer", "bur_higher")]  # ablation: same model, no crystal-burial features
        FD = [c for c in FM if c != "bur_higher"]    # dimer (coiled-coil core) burial only
        FL = [c for c in FM if c != "bur_dimer"]  # lattice (non-dimer) burial only
        FNF = [c for c in FM if not c.startswith("fam_")]  # ablation: no IF-family conservation features
        models = {"S AlphaMissense+structure": FS, "M mechanism-only stacked": FM,
                  "M-noStruct ablation": FN, "M-noFamily ablation": FNF, "M-dimerOnly": FD, "M-latticeOnly": FL,
                  "AB stacked": FB, "B supervised": F}
        # Was family-wide clinical pretraining the active ingredient? Retrain Stage A (a) with DES removed
        # entirely and (b) on DES alone, then score the DES phenotype benchmark zero-shot.
        noDES = trainA[trainA.gene != "DES"]
        desOnly = trainA[trainA.gene == "DES"]
        pNo = lgbm().fit(noDES[F], noDES.y_path.astype(int)).predict_proba(func[F])[:, 1]
        pDes = (lgbm_small().fit(desOnly[F], desOnly.y_path.astype(int)).predict_proba(func[F])[:, 1]
                if desOnly.y_path.nunique() == 2 else np.full(len(func), 0.5))
        # size-matched control: is the DES-only deficit about family breadth, or just fewer labels?
        rs = np.random.default_rng(3)
        nd = noDES.reset_index(drop=True)
        sm = []
        for _ in range(200):
            sub = nd.iloc[rs.choice(len(nd), min(len(desOnly), len(nd)), replace=False)]
            if sub.y_path.nunique() == 2:
                sm.append(roc_auc_score(yB, lgbm_small().fit(sub[F], sub.y_path.astype(int)).predict_proba(func[F])[:, 1]))
        aDes = roc_auc_score(yB, pDes)
        print(f"\nStage A pretraining ablation: all IF n={len(trainA)}, without DES n={len(noDES)}, DES only n={len(desOnly)}")
        print(f"  size-matched non-DES (n={len(desOnly)}, {len(sm)} draws): AUROC {np.mean(sm):.3f} "
              f"[{np.percentile(sm, 2.5):.3f}-{np.percentile(sm, 97.5):.3f}] vs DES-only {aDes:.3f} "
              f"-> one-sided p (draws <= DES-only) = {np.mean(np.array(sm) <= aDes):.3f}")
        pd.DataFrame(dict(size_matched_auroc=sm)).to_csv(RES / "stageA_pretraining_size_matched.csv", index=False)
        zero = {"A transfer (zero-shot)": func.ifclass_path_score.values,
                "A transfer, pretrained WITHOUT DES": pNo,
                "A transfer, pretrained on DES only": pDes,
                "ESM-2 zero-shot (-LLR)": -func.esm_llr.values, "AlphaMissense": func.am_score.values}
        oofs = {}
        for sch, grp in schemes.items():  # three increasingly strict splits
            oofs[sch] = {f"IF-FoldClass-{k} (leave-{sch}-out)": oof(lgbm_small, func[c], yB, grp) for k, c in models.items()}
            compare(yB, {**oofs[sch], **zero}, f"stageB_leave-{sch}-out")
        sB = {**oofs["study"], **zero}  # headline = leave-study-out (strictest)
        roc_plot(yB, sB, RES / "fig_roc_stageB.png", "Stage B: network vs aggregation (leave-study-out)")

        # thresholded calls + exact McNemar vs AlphaMissense (published cut-off 0.564)
        call = {k: np.nan_to_num(v, nan=0) > (0.564 if k == "AlphaMissense" else 0.5) for k, v in sB.items() if "ESM" not in k}
        # fair baselines: cut-off re-learned on training studies (max balanced accuracy), applied to held-out study
        for k in ("AlphaMissense", "ESM-2 zero-shot (-LLR)", "A transfer (zero-shot)"):
            call[f"{k} [recalibrated cut-off]"] = cv_threshold(np.nan_to_num(sB[k], nan=np.nanmedian(sB[k])), yB, study)
        am_ok = call["AlphaMissense"] == yB
        amr_ok = call["AlphaMissense [recalibrated cut-off]"] == yB
        tt = []
        for k, c in call.items():
            ok = c == yB
            b, cc = int((ok & ~am_ok).sum()), int((~ok & am_ok).sum())
            b2, c2 = int((ok & ~amr_ok).sum()), int((~ok & amr_ok).sum())
            ba, lo, hi = bal_acc_ci(yB, c)
            tt.append(dict(predictor=k, sensitivity=c[yB == 1].mean(), specificity=(~c[yB == 0]).mean(),
                           balanced_acc=ba, bal_acc_lo=lo, bal_acc_hi=hi,
                           wins_vs_AM=b, losses_vs_AM=cc,
                           mcnemar_p=binomtest(b, b + cc).pvalue if b + cc else 1.0,
                           wins_vs_AMrecal=b2, losses_vs_AMrecal=c2,
                           mcnemar_p_vs_AMrecal=binomtest(b2, b2 + c2).pvalue if b2 + c2 else 1.0,
                           false_pos=", ".join(func.variant[(yB == 0) & c])))
        tt = pd.DataFrame(tt)
        pc = [c for c in tt if c.startswith("mcnemar_p")]  # keep exact p; .round(4) printed 3e-12 as 0.0
        tt[[c for c in tt if c not in pc]] = tt[[c for c in tt if c not in pc]].round(4)
        tt[pc] = tt[pc].map(lambda v: f"{v:.2e}")
        tt.to_csv(RES / "stageB_threshold_errors.csv", index=False)
        print(f"\n== Thresholded calls (leave-study-out) ==\n{tt.drop(columns='false_pos').to_string(index=False)}")

        # per-domain: where do generic predictors fail?
        dom = []
        for d, idx in func.groupby(func.domain.astype(str)).groups.items():
            i = func.index.get_indexer(idx)
            for k, c in call.items():
                dom.append(dict(domain=d, predictor=k, n=len(i), n_agg=int(yB[i].sum()),
                                sensitivity=c[i][yB[i] == 1].mean() if yB[i].sum() else np.nan,
                                specificity=(~c[i][yB[i] == 0]).mean() if (yB[i] == 0).any() else np.nan))
        dom = pd.DataFrame(dom).round(3)
        dom.to_csv(RES / "stageB_per_domain.csv", index=False)
        print(f"\n== Per-domain calls ==\n{dom.pivot_table(index=['domain', 'n', 'n_agg'], columns='predictor', values='specificity').round(2).to_string()}")

        # mechanism: proline tolerance by heptad position (a/d core vs surface)
        pro = func[func.mut == "P"].assign(core=lambda x: x.heptad.astype(str).isin(["a", "d"]),
                                          am_call=lambda x: x.am_score > 0.564)
        if len(pro):
            ct = pd.crosstab([pro.domain.astype(str), pro.heptad.astype(str)], pro.phenotype)
            ct.to_csv(RES / "proline_heptad_crosstab.csv")
            cc = pd.crosstab(pro.core, pro.y_func.astype(int))
            print(f"\n== Proline substitutions: phenotype by heptad ==\n{ct.to_string()}")
            print(f"a/d core prolines aggregating: {pro[pro.core].y_func.mean():.2f} (n={pro.core.sum()}); "
                  f"other positions: {pro[~pro.core].y_func.mean():.2f} (n={(~pro.core).sum()}); "
                  f"AlphaMissense calls {pro.am_call.mean():.2f} of all prolines pathogenic")
            if cc.shape == (2, 2):
                from scipy.stats import fisher_exact
                print(f"Fisher exact (core vs non-core aggregation): p = {fisher_exact(cc.values)[1]:.2e}")

        # paired bootstrap on AUROC differences (the decisive stat: is any gain real?)
        rng2 = np.random.default_rng(1)
        idx = [rng2.integers(0, len(yB), len(yB)) for _ in range(2000)]
        idx = [i for i in idx if 0 < yB[i].sum() < len(i)]
        def dauc(a, b, mask=None):
            m = np.ones(len(yB), bool) if mask is None else np.asarray(mask)
            d = []
            for i in idx:
                j = i[m[i]]
                if len(j) and 0 < yB[j].sum() < len(j):
                    d.append(roc_auc_score(yB[j], np.asarray(a)[j]) - roc_auc_score(yB[j], np.asarray(b)[j]))
            d = np.array(d)
            # two-sided: take the smaller tail in EITHER direction, else a negative delta reports p=1.0
            return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), 2 * min((d <= 0).mean(), (d >= 0).mean())
        amv = np.nan_to_num(sB["AlphaMissense"], nan=np.nanmedian(sB["AlphaMissense"]))
        # burial is NaN outside solved vimentin fragments, and that missingness tracks domain (head has no
        # structure). Restricting to structure-covered variants makes the flag constant, so any surviving
        # gain is real structural signal rather than a domain shortcut.
        cov = func.bur_dimer.notna().values
        tet = func.bur_higher.notna().values
        fsh = func.copy()  # same missingness, scrambled values: isolates information content from coverage
        rngb = np.random.default_rng(7)
        for c in ("bur_dimer", "bur_higher"):
            v = fsh[c].values.copy()
            ok = ~np.isnan(v)
            v[ok] = rngb.permutation(v[ok])
            fsh[c] = v
        pShuf = oof(lgbm_small, fsh[FM], yB, study)
        # same control for the family features: shuffle their values among rows where they are defined
        ffs, fcov = func.copy(), func.fam_entropy.notna().values
        for c in ("fam_entropy", "fam_freq_wt", "fam_freq_mut", "fam_logodds"):
            v = ffs[c].values.copy(); v[fcov] = rngb.permutation(v[fcov]); ffs[c] = v
        pFamShuf = oof(lgbm_small, ffs[FM], yB, study)
        mM, mN = sB["IF-FoldClass-M mechanism-only stacked (leave-study-out)"], sB["IF-FoldClass-M-noStruct ablation (leave-study-out)"]
        esmv = -func.esm_llr.values
        pairs = [# does training on IF clinical labels add anything beyond the protein language model itself?
                 ("Stage A vs ESM-2 zero-shot alone", func.ifclass_path_score.values, esmv, None),
                 ("Stage A without DES vs ESM-2 alone", pNo, esmv, None),
                 ("Stage A vs AlphaMissense", func.ifclass_path_score.values, amv, None),
                 ("family pretraining: all-IF vs DES-only", func.ifclass_path_score.values, pDes, None),
                 ("family pretraining without DES vs DES-only", pNo, pDes, None),
                 ("dropping DES from pretraining costs", func.ifclass_path_score.values, pNo, None),
                 # PRE-SPECIFIED PRIMARY COMPARISON for the family-conservation addition:
                 ("family conservation adds (M vs M-noFamily)", mM,
                  sB["IF-FoldClass-M-noFamily ablation (leave-study-out)"], None),
                 ("M(with family) vs ESM-2 alone", mM, esmv, None),
                 ("family VALUES beat shuffled family values", mM, pFamShuf, None),
                 ("structure features add (M vs M-noStruct), all variants", mM, mN, None),
                 (f"structure features add, structure-covered only (n={int(cov.sum())})", mM, mN, cov),
                 ("M vs AlphaMissense", mM, amv, None),
                 (f"M vs AlphaMissense, structure-covered only", mM, amv, cov),
                 ("S vs AlphaMissense", sB["IF-FoldClass-S AlphaMissense+structure (leave-study-out)"], amv, None),
                 ("dimer-core burial alone adds (covered)", sB["IF-FoldClass-M-dimerOnly (leave-study-out)"],
                  sB["IF-FoldClass-M-noStruct ablation (leave-study-out)"], cov),
                 ("lattice burial alone adds (covered)", sB["IF-FoldClass-M-latticeOnly (leave-study-out)"],
                  sB["IF-FoldClass-M-noStruct ablation (leave-study-out)"], cov),
                 # Controls kept from the crystal-fragment version: they now act on near-complete
                 # coverage, so they confirm the signal is the burial VALUE rather than which residues are modelled.
                 # (a) restrict to residues where it is defined, (b) shuffle values within covered rows.
                 (f"lattice burial adds, only where defined (n={int(tet.sum())})",
                  sB["IF-FoldClass-M-latticeOnly (leave-study-out)"],
                  sB["IF-FoldClass-M-noStruct ablation (leave-study-out)"], tet),
                 ("burial VALUES beat shuffled values (covered)", mM, pShuf, cov),
                 ("burial VALUES beat shuffled values, lattice-defined only", mM, pShuf, tet)]
        dt = pd.DataFrame([dict(comparison=n, delta_auroc=a, lo=b, hi=c, p_two_sided=min(d, 1.0))
                           for n, x, y, m in pairs for a, b, c, d in [dauc(x, y, m)]]).round(4)
        print(f"\nstructure coverage: {int(cov.sum())}/{len(cov)} labelled variants; "
              f"aggregation among covered {yB[cov].mean():.2f}, uncovered {yB[~cov].mean():.2f}")
        # Benjamini-Hochberg across the whole delta table (the primary comparison is pre-specified
        # in flow.md §9 and also reported uncorrected).
        from scipy.stats import false_discovery_control
        dt["p_BH"] = false_discovery_control(dt.p_two_sided.clip(1e-12, 1.0), method="bh").round(4)
        dt.to_csv(RES / "stageB_paired_auroc_deltas.csv", index=False)
        print(f"\n== Paired bootstrap AUROC differences (leave-study-out) ==\n{dt.to_string(index=False)}")

        mB = lgbm_small().fit(func[FB], yB)
        mM = lgbm_small().fit(func[FM], yB)
        con = pd.DataFrame(np.abs(mM.predict(func[FM], pred_contrib=True)[:, :-1]), columns=FM).mean().sort_values()
        con.to_csv(RES / "feature_importance_stageB_mechanism.csv")
        plt.figure(figsize=(6, 5)); con.tail(15).plot.barh(); plt.xlabel("mean |SHAP| (log-odds)")
        plt.title("Stage B mechanism model"); plt.tight_layout(); plt.savefig(RES / "fig_importance_stageB.png", dpi=300); plt.close()

        c2 = func[func.bur_dimer.notna()].assign(lab=lambda x: np.where(x.y_func == 1, "aggregation", "network"),
                                                 pro=lambda x: np.where(x.mut == "P", "proline", "other"))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        for k, (nm, g) in enumerate(c2.groupby("pro")):
            for lab, gg in g.groupby("lab"):
                ax[k].scatter(gg.bur_dimer, gg.bur_higher, s=18, alpha=.7, label=f"{lab} (n={len(gg)})")
            ax[k].set_title(nm); ax[k].set_xlabel("dimer-core burial"); ax[k].legend(fontsize=7)
        ax[0].set_ylabel("tetramer-interface burial")
        plt.tight_layout(); plt.savefig(RES / "fig_burial_phenotype.png", dpi=300); plt.close()
        for nm, g in c2.groupby("pro"):  # note: locals must not shadow the argparse namespace `a`
            d1, d0 = g.bur_dimer[g.y_func == 1], g.bur_dimer[g.y_func == 0]
            t, u = g.bur_higher[g.y_func == 1].dropna(), g.bur_higher[g.y_func == 0].dropna()
            pt = mannwhitneyu(t, u).pvalue if len(t) and len(u) else float("nan")
            print(f"  {nm:8s} n={len(g):3d}  dimer burial agg {d1.mean():.2f} vs net {d0.mean():.2f} "
                  f"(MWU p={mannwhitneyu(d1, d0).pvalue:.3g});  lattice agg {t.mean():.2f} vs net {u.mean():.2f} "
                  f"(n={len(t)}/{len(u)}, p={pt:.3g})")
        df["ifclass_aggregation_prob"] = mB.predict_proba(df[FB])[:, 1]
        df["ifclass_mechanism_prob"] = mM.predict_proba(df[FM])[:, 1]

        # label-permutation test on the strictest split
        key = "IF-FoldClass-AB stacked (leave-study-out)"
        obs = roc_auc_score(yB, sB[key])
        rng = np.random.default_rng(0)
        null = [roc_auc_score(yp, oof(lgbm_small, func[FB], yp, study)) for yp in (rng.permutation(yB) for _ in range(a.n_perm))]
        pval = (1 + sum(n >= obs for n in null)) / (1 + len(null))
        print(f"\nPermutation test ({key}, {a.n_perm} shuffles): AUROC {obs:.3f}, null mean {np.mean(null):.3f}, p = {pval:.4f}")
        pd.DataFrame(dict(null_auroc=null)).to_csv(RES / "stageB_permutation_null.csv", index=False)

        # cross-IF transfer: DES-phenotype-only models (no ClinVar input) vs ClinVar in other IF genes
        mBo, mMo = lgbm_small().fit(func[F], yB), lgbm_small().fit(func[MECH], yB)
        o = df[(df.gene != "DES") & df.clinvar_y.notna()]
        ot = [dict(gene=g, n=len(x), auroc_des_phenotype_model=roc_auc_score(x.clinvar_y, mBo.predict_proba(x[F])[:, 1]),
                   auroc_des_mechanism_model=roc_auc_score(x.clinvar_y, mMo.predict_proba(x[MECH])[:, 1]),
                   auroc_alphamissense=roc_auc_score(x.clinvar_y[x.am_score.notna()], x.am_score.dropna()))
              for g, x in o.groupby("gene") if x.clinvar_y.nunique() == 2 and min((x.clinvar_y == 1).sum(), (x.clinvar_y == 0).sum()) >= 5]
        if ot:
            ot = pd.DataFrame(ot).round(3)
            ot.to_csv(RES / "cross_IF_transfer.csv", index=False)
            print(f"\n== Cross-IF transfer: DES-phenotype model vs ClinVar in other IF genes ==\n{ot.to_string(index=False)}")

        # cross-IF phenotype test: published aggregation phenotypes in other IFs + gnomAD-common variants as negatives
        if CROSS_IF.exists():
            cx = pd.read_csv(CROSS_IF, comment="#").merge(df, on=["gene", "variant"], how="inner")
            neg = df[df.gene.isin(cx.gene) & (df.af >= GNOMAD_BENIGN_AF) & ~df.variant.isin(cx.variant)].assign(phenotype="population_common")
            cx = pd.concat([cx, neg], ignore_index=True)
            cx["y"] = (~cx.phenotype.isin(["network", "population_common"])).astype(int)
            cx["ifclass_path_score"] = mA.predict_proba(cx[F])[:, 1]
            cx["mech_prob"], cx["stacked_prob"] = mM.predict_proba(cx[FM])[:, 1], mB.predict_proba(cx[FB])[:, 1]
            cx[["gene", "variant", "domain", "heptad", "phenotype", "mech_prob", "stacked_prob", "ifclass_path_score", "am_score"]] \
                .round(3).to_csv(RES / "cross_IF_phenotype_predictions.csv", index=False)
            r = [dict(predictor=k, n=len(cx), n_agg=int(cx.y.sum()), auroc=roc_auc_score(cx.y[cx[c].notna()], cx[c].dropna()),
                      sens=(cx[c][cx.y == 1] > t).mean(), spec=(cx[c][cx.y == 0] <= t).mean())
                 for k, c, t in [("mechanism-only", "mech_prob", .5), ("stacked", "stacked_prob", .5),
                                 ("Stage A", "ifclass_path_score", .5), ("AlphaMissense", "am_score", .564)]]
            print(f"\n== Cross-IF phenotype test (GFAP/NEFL, DES-trained models) ==\n{pd.DataFrame(r).round(3).to_string(index=False)}")
            pd.DataFrame(r).round(3).to_csv(RES / "cross_IF_phenotype_metrics.csv", index=False)
    else:
        print(f"\nStage B skipped: need >=20 curated rows with both phenotypes in {FUNC_LABELS.name} "
              f"(have {0 if func is None else len(func)}). Fill it from the DES atlas papers (see README).")

    # ---- Stage C: release scores for every possible IF missense
    keep = ["acc", "gene", "pos", "wt", "mut", "variant", "domain", "heptad", "plddt", "esm_llr", "am_score",
            "clinvar_y", "af", "ifclass_path_score"] + [c for c in ("ifclass_aggregation_prob", "ifclass_mechanism_prob") if c in df]
    df[keep].to_parquet(RES / "ifclass_scores_all_IF_variants.parquet", index=False)
    des = df[df.gene == "DES"]
    if len(des):
        des[keep].to_csv(RES / "ifclass_scores_DES.csv", index=False)
        col = "ifclass_aggregation_prob" if "ifclass_aggregation_prob" in des else "ifclass_path_score"
        mat = des.pivot_table(index="mut", columns="pos", values=col, observed=True)
        plt.figure(figsize=(14, 3.5)); plt.imshow(mat.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        plt.yticks(range(len(mat.index)), mat.index, fontsize=6); plt.xlabel("DES residue"); plt.colorbar(label=col)
        plt.title("DES saturation map"); plt.tight_layout(); plt.savefig(RES / "fig_DES_saturation.png", dpi=300); plt.close()
    print(f"\nDone. Results in {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "features", "train", "all"])
    ap.add_argument("--esm", default="facebook/esm2_t33_650M_UR50D",
                    help="weaker PC: facebook/esm2_t12_35M_UR50D ; big GPU: facebook/esm2_t36_3B_UR50D")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-perm", type=int, default=200, help="label shuffles for Stage B permutation test")
    a = ap.parse_args()
    for step, fn in [("download", cmd_download), ("features", cmd_features), ("train", cmd_train)]:
        if a.step in (step, "all"):
            fn(a)
