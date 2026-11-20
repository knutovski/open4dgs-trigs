"""Rendert exportierte TRiGS-PLYs aus den COLMAP-Kameras und legt sie neben die Ground Truth.
Aufruf: python render_ply_compare.py <data_dir> <ply_dir> <out.png> [--frames 0 5 9] [--cams 0 2 4]
"""
import argparse, math
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization

ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("plys"); ap.add_argument("out")
ap.add_argument("--frames", type=int, nargs="+", default=[0, 5, 9]); ap.add_argument("--cams", type=int, nargs="+", default=[0, 2, 4])
a = ap.parse_args(); dev = "mps" if torch.backends.mps.is_available() else "cpu"

def read_ply(p):
    with open(p, "rb") as fh:
        props, n = [], 0
        while (line := fh.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"): n = int(line.split()[-1])
            elif line.startswith("property"): props.append(line.split()[-1])
        arr = np.frombuffer(fh.read(n * 4 * len(props)), np.float32).reshape(n, len(props))
    return {k: arr[:, i] for i, k in enumerate(props)}

def quat2rot(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], np.float32)
sp = Path(a.data) / "sparse" / "0"
cams = {int(l.split()[0]): l.split() for l in open(sp / "cameras.txt") if l.strip() and not l.startswith("#")}
views = []
for l in open(sp / "images.txt"):
    s = l.split()
    if len(s) >= 10 and not l.startswith("#"):
        R = quat2rot(*map(float, s[1:5])); t = np.array(list(map(float, s[5:8])), np.float32)
        c = cams[int(s[8])]; W, H = int(c[2]), int(c[3]); fx, fy, cx, cy = map(float, c[4:8])
        V = np.eye(4, dtype=np.float32); V[:3, :3] = R; V[:3, 3] = t
        views.append((s[9].split(".")[0], V, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32), W, H))
views.sort(key=lambda v: v[0])
rows, psnrs = [], []
for f in a.frames:
    d = read_ply(Path(a.plys) / f"frame_{f:06d}.ply")
    T = lambda keys: torch.from_numpy(np.stack([d[k] for k in keys], 1).copy()).to(dev)
    means = T(["x", "y", "z"]); quats = torch.nn.functional.normalize(T(["rot_0", "rot_1", "rot_2", "rot_3"]), dim=-1)
    scales = T(["scale_0", "scale_1", "scale_2"]).exp(); opac = torch.sigmoid(T(["opacity"])[:, 0])
    n_rest = len([k for k in d if k.startswith("f_rest_")]); K_sh = n_rest // 3 + 1; deg = int(math.isqrt(K_sh) - 1)
    dc = T(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]
    rest = T([f"f_rest_{i}" for i in range(n_rest)]).reshape(-1, 3, K_sh - 1).transpose(1, 2)
    sh = torch.cat([dc, rest], 1)
    row = []
    for ci in a.cams:
        name, V, K, W, H = views[ci]
        img, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=sh,
                                  viewmats=torch.from_numpy(V)[None].to(dev), Ks=torch.from_numpy(K)[None].to(dev),
                                  width=W, height=H, sh_degree=deg, packed=False,
                                  backgrounds=torch.tensor([[0.05, 0.05, 0.08]], device=dev))
        r = img[0].clamp(0, 1).cpu().numpy()
        gt = np.asarray(Image.open(Path(a.data) / "images" / name / f"{f:06d}.jpg"), np.float32) / 255
        psnrs.append(-10 * math.log10(max(((r - gt) ** 2).mean(), 1e-12)))
        row.append(np.concatenate([gt, r], 1))
    rows.append(np.concatenate(row, 1))
Image.fromarray((np.concatenate(rows, 0) * 255).round().astype(np.uint8)).save(a.out)
print(f"PSNR PLY-Rendering vs GT: mittel {np.mean(psnrs):.2f} dB, min {np.min(psnrs):.2f}, max {np.max(psnrs):.2f} "
      f"({len(psnrs)} Bilder, Frames {a.frames}, Kameras {a.cams}) -> {a.out}")
