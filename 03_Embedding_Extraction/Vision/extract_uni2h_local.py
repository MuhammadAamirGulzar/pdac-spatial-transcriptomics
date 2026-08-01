"""
UNI2-h patch embeddings, run locally on the RTX 4090.

A faithful port of `uni2-h-feature-extraction.ipynb` (which ran on Kaggle over the
original 6 samples). Model construction, preprocessing, the reg-token skip and the
output schema are copied verbatim so new samples are interchangeable with the
existing `dataset/Feature Extraction Embeddings/UNI2-h/*_uni2h.pt`.

Differences from the notebook, all I/O only:
  * reads patches from the repo instead of a Kaggle input mount;
  * a DataLoader with worker processes decodes PNGs in parallel, so the GPU is not
    starved by single-threaded JPEG/PNG decode (the notebook opened images one at
    a time in the main process);
  * skips samples whose .pt already has the expected patch count.

Weights come from the local HuggingFace cache; UNI2-h is a gated repo, so if the
cache is cold this needs `huggingface-cli login` with access to MahmoodLab/UNI2-h.

Run:
    python 03_Embedding_Extraction/Vision/extract_uni2h_local.py [SAMPLE ...]
    python 03_Embedding_Extraction/Vision/extract_uni2h_local.py all
"""

import argparse
import glob
import os
import sys
import time

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PNG_ROOT = os.path.join(ROOT, "dataset", ".png patches", ".png patches")
OUT_DIR = os.path.join(ROOT, "dataset", "Feature Extraction Embeddings", "UNI2-h")
BATCH = 64
WORKERS = 8

# identical to the notebook
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class PatchDS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return TRANSFORM(Image.open(p).convert("RGB")), \
            os.path.splitext(os.path.basename(p))[0]


def build_model():
    import timm
    m = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        img_size=224, patch_size=14, depth=24,
        num_heads=24, init_values=1e-5, embed_dim=1536,
        mlp_ratio=2.66667 * 2, num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    return m.eval().cuda().half()


def process(sample, model):
    sdir = os.path.join(PNG_ROOT, sample)
    out = os.path.join(OUT_DIR, f"{sample}_uni2h.pt")
    patches = sorted(glob.glob(os.path.join(sdir, "*.png")))
    if not patches:
        print(f"[SKIP] no PNGs for {sample}")
        return
    if os.path.exists(out):
        saved = torch.load(out, map_location="cpu", weights_only=False)
        if len(saved["patch_names"]) == len(patches):
            print(f"[SKIP] {sample} already done ({len(patches)} patches)")
            return
        print(f"[REDO] {sample}: {len(saved['patch_names'])} saved vs {len(patches)} patches")

    dl = DataLoader(PatchDS(patches), batch_size=BATCH, num_workers=WORKERS,
                    pin_memory=True, shuffle=False)
    embs, names = [], []
    t0 = time.time()
    for x, nm in dl:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
                f = model.forward_features(x.cuda(non_blocking=True).half())
                e = f[:, 8:, :].mean(dim=1)          # skip the 8 register tokens
        embs.append(e.cpu().float())
        names.extend(nm)
    E = torch.cat(embs, 0)
    torch.save({"embeddings": E, "patch_names": names,
                "model": "UNI2-h", "patient": sample}, out)
    dt = time.time() - t0
    print(f"  {sample:14s} {tuple(E.shape)}  {dt:6.1f}s  "
          f"({len(patches)/dt:.0f} patch/s)  -> {os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="*", default=["all"])
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if a.samples == ["all"] or not a.samples:
        todo = sorted(d for d in os.listdir(PNG_ROOT)
                      if os.path.isdir(os.path.join(PNG_ROOT, d)))
    else:
        todo = a.samples
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    print(f"GPU: {torch.cuda.get_device_name(0)} | samples: {todo}")
    model = build_model()
    print("UNI2-h loaded")
    for s in todo:
        process(s, model)
    print("DONE ->", OUT_DIR)


if __name__ == "__main__":
    main()
