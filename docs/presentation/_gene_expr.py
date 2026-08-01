"""
Per-spot gene-expression extractor for presentation figures.

Pulls a handful of marker genes straight out of the REAL lab count matrices
(dataset/ST/scVI_counts/{sample}.csv, genes-as-rows x spots-as-cols, raw integer
counts) without loading the whole 140 MB matrix into memory: we scan the file once
and keep only the rows whose gene symbol is in the requested set, plus the header
row of spot barcodes.

Expression is returned CP10k-log1p normalised per spot (same recipe Stage-1A used),
using each spot's total counts (nCount) from the *_qc_metrics.csv, so values are
comparable spot-to-spot and sample-to-sample.

Everything is keyed by `barcode` (e.g. PDACP_10_AAAC...-1 / PDACH_...), which joins
to leaving_program_scores.csv and to the WSI pixel coords via _tissue.wsi_spot_xy.
"""
import os, csv, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(ROOT))
COUNTS = os.path.join(PROJ, "dataset", "ST", "scVI_counts")
CACHE = os.path.join(ROOT, "_gene_cache")
os.makedirs(CACHE, exist_ok=True)


def _qc_ncount(sample):
    """barcode -> total counts (library size) for CP10k normalisation."""
    p = os.path.join(COUNTS, f"{sample}_qc_metrics.csv")
    q = pd.read_csv(p)
    return dict(zip(q["barcode"].astype(str), q["nCount"].astype(float)))


def extract(sample, genes, force=False):
    """Return DataFrame indexed by barcode with one CP10k-log1p column per gene that
    is present in the matrix. Missing genes are simply skipped (check .columns)."""
    genes = list(dict.fromkeys(genes))
    key = os.path.join(CACHE, f"{sample}__{'_'.join(sorted(genes))[:80]}_{len(genes)}.csv")
    if os.path.exists(key) and not force:
        return pd.read_csv(key, index_col=0)

    path = os.path.join(COUNTS, f"{sample}.csv")
    want = set(genes)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        barcodes = header[1:]
        raw = {}
        for row in reader:
            g = row[0]
            if g in want:
                raw[g] = np.asarray(row[1:], dtype=np.float32)
                if len(raw) == len(want):
                    break

    ncount = _qc_ncount(sample)
    lib = np.asarray([ncount.get(b, np.nan) for b in barcodes], dtype=np.float32)
    lib[lib <= 0] = np.nan
    out = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    for g in genes:
        if g in raw:
            cp10k = raw[g] / lib * 1e4
            out[g] = np.log1p(cp10k)
    out.to_csv(key)
    return out


def presence_report(genes, samples=None):
    """Quick which-genes-are-present-per-sample table (mean CP10k-log1p, %spots>0)."""
    if samples is None:
        samples = ["IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11",
                   "IU_PDA_HM11", "IU_PDA_HM13"]
    recs = []
    for s in samples:
        df = extract(s, genes)
        for g in genes:
            if g in df.columns:
                v = df[g].values
                recs.append(dict(sample=s, gene=g, present=True,
                                 mean=float(np.nanmean(v)), pct_pos=float(np.mean(v > 0))))
            else:
                recs.append(dict(sample=s, gene=g, present=False, mean=0.0, pct_pos=0.0))
    return pd.DataFrame(recs)


if __name__ == "__main__":
    PANEL = ["ALB", "TTR", "APOA1", "HP",                                   # hepatocyte
             "SERPINE1", "S100A4", "POSTN", "MMP9", "MMP14", "COL1A1",       # EMT/invasion
             "FN1", "LAMC2", "TNC", "VIM", "SNAI2", "SPARC",
             "EPCAM", "CDH1", "KRT19", "KRT8",                               # epithelial
             "KRAS", "TP53", "SMAD4"]                                        # drivers
    rep = presence_report(PANEL)
    piv = rep.pivot(index="gene", columns="sample", values="pct_pos")
    pd.set_option("display.width", 200, "display.max_columns", 12)
    print("=== fraction of spots with detected expression (>0) ===")
    print(piv.reindex(PANEL).round(2).to_string())
    piv.to_csv(os.path.join(CACHE, "presence_pctpos.csv"))
    rep.to_csv(os.path.join(CACHE, "presence_full.csv"), index=False)
    print("\nsaved presence report to", CACHE)
