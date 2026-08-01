"""
Download a SUBSET of HEST-1k (MahmoodLab) — never the whole thing.

Why a subset
------------
HEST-1k is 1,876 GB in total. Almost all of it is data we do not need:

    wsis/          1057 GB   whole-slide pyramidal TIFFs  -> SKIP
    transcripts/    370 GB   Xenium per-transcript tables -> SKIP
    patches/        300 GB   224x224 H&E patches (.h5)    -> KEEP (subset)
    st/              78 GB   expression (.h5ad)           -> KEEP (subset)
    metadata/         0 GB   per-sample JSON              -> KEEP (all, tiny)

Skipping `wsis/` costs nothing: HEST already ships 224x224 patches at the spot
coordinates in `patches/*.h5`, which is exactly the input our UNI2-h extractor
takes.  Re-extracting them from the WSIs would be a terabyte of download to
reproduce a file we are given.

ACCESS
------
HEST is a GATED dataset (CC BY-NC-SA 4.0).  Listing its files works for anyone,
but downloading returns 403 until your HuggingFace account is on the authorised
list.  Do this once, in a browser:

    1. sign in at https://huggingface.co/datasets/MahmoodLab/hest
    2. accept the terms / request access on that page
    3. make sure the CLI is authenticated as the SAME account:
           huggingface-cli login          (or set HF_TOKEN)
       check with:  python -c "from huggingface_hub import whoami; print(whoami()['name'])"

Usage
-----
    python 05_HEST/download_hest_subset.py --list-organs
    python 05_HEST/download_hest_subset.py --organ Pancreas --dry-run
    python 05_HEST/download_hest_subset.py --organ Pancreas
    python 05_HEST/download_hest_subset.py --oncotree PAAD --with-thumbnails
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "dataset", "external", "HEST")
REPO = "MahmoodLab/hest"
META_CSV = "HEST_v1_1_0.csv"          # the catalogue; other versions also exist at repo root


def die(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def check_access():
    from huggingface_hub import whoami, get_token
    from huggingface_hub.errors import GatedRepoError
    from huggingface_hub import HfApi
    tok = get_token()
    if not tok:
        die("no HuggingFace token found. Run `huggingface-cli login` first.")
    try:
        who = whoami(tok)["name"]
    except Exception as e:
        die(f"token rejected: {e}")
    api = HfApi()
    try:                                   # a 1-byte read is enough to test the gate
        api.hf_hub_download(REPO, META_CSV, repo_type="dataset",
                            local_dir=DEST, token=tok)
    except GatedRepoError:
        die(f"logged in as '{who}', but that account is NOT authorised for {REPO}.\n"
            f"       Accept the terms at https://huggingface.co/datasets/{REPO} "
            f"while signed in as '{who}', then re-run.")
    except Exception as e:
        die(f"could not fetch the catalogue: {type(e).__name__}: {e}")
    print(f"access OK as '{who}'")
    return os.path.join(DEST, META_CSV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ", help="e.g. Pancreas  (matches the 'organ' column)")
    ap.add_argument("--oncotree", help="e.g. PAAD  (matches 'oncotree_code')")
    ap.add_argument("--ids", nargs="*", help="explicit sample ids, e.g. TENX123 INT5")
    ap.add_argument("--ids-file", help="file with one sample id per line")
    ap.add_argument("--organs", nargs="*",
                    help="several organs at once, e.g. --organs Bowel Prostate Kidney")
    ap.add_argument("--cancer-only", action="store_true",
                    help="restrict to disease_state Cancer/Tumor")
    ap.add_argument("--visium-only", action="store_true",
                    help="restrict to st_technology == Visium (avoids mixing platforms)")
    ap.add_argument("--species", default=None,
                    help="e.g. 'Homo sapiens' — mixing species is a confound, set this")
    ap.add_argument("--with-thumbnails", action="store_true",
                    help="also fetch thumbnails/ and spatial_plots/ (small, useful for QC)")
    ap.add_argument("--list-organs", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report size, download nothing")
    a = ap.parse_args()

    import pandas as pd
    from huggingface_hub import HfApi, snapshot_download

    os.makedirs(DEST, exist_ok=True)
    csv = check_access()
    meta = pd.read_csv(csv, low_memory=False)
    idcol = "id" if "id" in meta.columns else meta.columns[0]
    print(f"catalogue: {len(meta)} samples, {len(meta.columns)} columns")

    if a.list_organs:
        for col in ("organ", "oncotree_code", "disease_state", "st_technology"):
            if col in meta.columns:
                print(f"\n--- {col} ---")
                print(meta[col].value_counts().to_string())
        return

    sel = meta
    if a.organ:
        sel = sel[sel["organ"].astype(str).str.contains(a.organ, case=False, na=False)]
    if a.organs:
        sel = sel[sel["organ"].astype(str).isin(a.organs)]
    if a.oncotree:
        sel = sel[sel["oncotree_code"].astype(str).str.upper() == a.oncotree.upper()]
    if a.cancer_only:
        sel = sel[sel["disease_state"].astype(str).isin(["Cancer", "Tumor"])]
    if a.visium_only:
        sel = sel[sel["st_technology"].astype(str) == "Visium"]
    if a.species:
        sel = sel[sel["species"].astype(str) == a.species]
    if a.ids:
        sel = sel[sel[idcol].isin(a.ids)]
    if a.ids_file:
        want = [x.strip() for x in open(a.ids_file) if x.strip()]
        sel = sel[sel[idcol].astype(str).isin(want)]
    if sel.empty:
        die("no samples matched that filter (try --list-organs)")

    ids = sel[idcol].astype(str).tolist()
    # skip anything already on disk, so the command is resumable
    have = set(os.listdir(os.path.join(DEST, "st"))) if os.path.isdir(os.path.join(DEST, "st")) else set()
    done = [i for i in ids if f"{i}.h5ad" in have]
    if done:
        print(f"  already downloaded, skipping: {len(done)}")
        ids = [i for i in ids if f"{i}.h5ad" not in have]
        if not ids:
            print("nothing left to fetch"); return
    print(f"\nselected {len(ids)} samples")
    for col in ("organ", "oncotree_code", "st_technology", "species"):
        if col in sel.columns:
            vc = sel[col].value_counts()
            print(f"  {col:16s} " + ", ".join(f"{k}={v}" for k, v in vc.items()))

    # size the request from the repo tree before committing to it
    api = HfApi()
    tree = {t.path: getattr(t, "size", 0) or 0
            for t in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)}
    folders = ["st", "patches", "metadata"] + (
        ["thumbnails", "spatial_plots"] if a.with_thumbnails else [])
    want, total = [], 0
    for sid in ids:
        for fold, ext in [("st", ".h5ad"), ("patches", ".h5"), ("metadata", ".json"),
                          ("thumbnails", ".jpg"), ("spatial_plots", ".png")]:
            if fold not in folders:
                continue
            p = f"{fold}/{sid}{ext}"
            if p in tree:
                want.append(p); total += tree[p]
    print(f"\nfiles to download: {len(want)}   total {total/1024**3:.1f} GB")
    print("  (whole-slide images and transcripts are deliberately excluded — "
          "patches/ already holds the 224px H&E tiles)")

    free = None
    try:
        import shutil
        free = shutil.disk_usage(DEST).free
        print(f"  free space at destination: {free/1024**3:.1f} GB")
        if free < total * 1.15:
            die("not enough free disk space for this subset")
    except Exception:
        pass

    if a.dry_run:
        print("\n--dry-run: nothing downloaded")
        return

    print(f"\ndownloading -> {DEST}")
    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=DEST,
                      allow_patterns=want, max_workers=8)
    print("\nDONE")
    print(f"  expression : {DEST}/st/<id>.h5ad")
    print(f"  H&E patches: {DEST}/patches/<id>.h5")


if __name__ == "__main__":
    main()
