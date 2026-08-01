"""
Run UNI2-h over HEST-1k patches -> one .npz of spot embeddings per sample.

HEST ships 224x224 H&E tiles already cut at the Visium spot coordinates
(`patches/<id>.h5` -> `img` (n,224,224,3) uint8, plus `barcode` and `coords`), so
there is no patch-extraction step: the tiles go straight into the model.

The model construction and preprocessing are IDENTICAL to
03_Embedding_Extraction/Vision/extract_uni2h_local.py (same timm args, same
Resize(256)/CenterCrop(224)/ImageNet normalise, same skip of the 8 register
tokens), so HEST embeddings and our own cohort's embeddings live in the same
space and are directly comparable.

Output: dataset/external/HEST/embeddings/<id>_uni2h.npz
        X        (n_spots, 1536) float32
        barcode  (n_spots,)      str

Run:
    python 05_HEST/hest_embed.py               # every downloaded sample
    python 05_HEST/hest_embed.py NCBI569 TENX116
"""

import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEST = os.path.join(ROOT, "dataset", "external", "HEST")
PATCH_DIR = os.path.join(HEST, "patches")
OUT_DIR = os.path.join(HEST, "embeddings")
BATCH = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class H5Patches(Dataset):
    """Reads img from the HEST patch file. Opened lazily per worker: an h5py
    handle cannot be shared across forked/spawned processes."""

    def __init__(self, path):
        self.path = path
        self._f = None
        with h5py.File(path, "r") as f:
            self.n = f["img"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        a = self._f["img"][i]                       # (224,224,3) uint8
        x = a.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))


def build_model():
    import timm
    m = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=True,
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True)
    return m.eval().cuda().half()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = sys.argv[1:]
    if not ids:
        ids = sorted(f[:-3] for f in os.listdir(PATCH_DIR) if f.endswith(".h5"))
    todo = [i for i in ids
            if not os.path.exists(os.path.join(OUT_DIR, f"{i}_uni2h.npz"))]
    print(f"{len(ids)} samples, {len(todo)} still to embed")
    if not todo:
        return
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    print("GPU:", torch.cuda.get_device_name(0))
    model = build_model()
    print("UNI2-h loaded")

    for k, sid in enumerate(todo, 1):
        p = os.path.join(PATCH_DIR, f"{sid}.h5")
        with h5py.File(p, "r") as f:
            bc = [b[0].decode() if isinstance(b[0], bytes) else str(b[0])
                  for b in f["barcode"][:]]
        dl = DataLoader(H5Patches(p), batch_size=BATCH, num_workers=6,
                        pin_memory=True, shuffle=False)
        embs, t0 = [], time.time()
        for x in dl:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.inference_mode():
                    f_ = model.forward_features(x.cuda(non_blocking=True).half())
                    embs.append(f_[:, 8:, :].mean(1).cpu().float())
        X = torch.cat(embs).numpy()
        np.savez_compressed(os.path.join(OUT_DIR, f"{sid}_uni2h.npz"),
                            X=X, barcode=np.array(bc))
        dt = time.time() - t0
        print(f"  [{k}/{len(todo)}] {sid:10s} {X.shape}  {dt:5.1f}s "
              f"({X.shape[0]/dt:.0f} patch/s)")
    print("DONE ->", OUT_DIR)


if __name__ == "__main__":
    main()
