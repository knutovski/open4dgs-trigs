"""Rendert exportierte PLYs einer echten Aufnahme aus den kalibrierten Kameras (reference_export.json)
und stellt sie neben die verzeichnungskorrigierten Originalbilder.
Aufruf: python render_real_compare.py <capture> <ply_dir> <out.jpg> [--frames 1 115 230] [--scale 0.5]"""
import argparse, math, json
from pathlib import Path
import numpy as np, torch, cv2
from gsplat import rasterization
ap = argparse.ArgumentParser(); ap.add_argument("cap"); ap.add_argument("plys"); ap.add_argument("out")
ap.add_argument("--frames", type=int, nargs="+", default=[1, 115, 230]); ap.add_argument("--scale", type=float, default=0.5)
a = ap.parse_args(); dev = "mps" if torch.backends.mps.is_available() else "cpu"
def read_ply(p):
    with open(p, "rb") as fh:
        props, n = [], 0
        while (line := fh.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"): n = int(line.split()[-1])
            elif line.startswith("property"): props.append(line.split()[-1])
        arr = np.frombuffer(fh.read(n * 4 * len(props)), np.float32).reshape(n, len(props))
    return {k: arr[:, i] for i, k in enumerate(props)}
cams = sorted(json.load(open(Path(a.cap) / "reference_export.json"))["cameras"], key=lambda c: c["folder"])
rows, ps = [], []
meta = json.load(open(Path(a.plys) / "export_meta.json")); f0 = int(meta.get("start_frame", 0))
for fr in a.frames:
    idx = fr - 1 - f0 if fr - 1 - f0 >= 0 else 0      # Bild 000001 = Export-Frame 0
    d = read_ply(Path(a.plys) / f"frame_{idx:06d}.ply")
    T = lambda ks: torch.from_numpy(np.stack([d[k] for k in ks], 1).copy()).to(dev)
    means = T(["x", "y", "z"]); quats = torch.nn.functional.normalize(T(["rot_0", "rot_1", "rot_2", "rot_3"]), dim=-1)
    scales = T(["scale_0", "scale_1", "scale_2"]).exp(); opac = torch.sigmoid(T(["opacity"])[:, 0])
    nr = len([k for k in d if k.startswith("f_rest_")]); K_sh = nr // 3 + 1
    sh = torch.cat([T(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None], T([f"f_rest_{i}" for i in range(nr)]).reshape(-1, 3, K_sh - 1).transpose(1, 2)], 1)
    row = []
    for c in cams:
        p = c["params"]; W, H = c["width"], c["height"]
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], np.float32); dist = np.array([p[3], p[4], 0, 0], np.float32)
        V = np.eye(4, dtype=np.float32); V[:3, :3] = np.array(c["R"]); V[:3, 3] = np.array(c["t"])
        img, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=sh,
                                  viewmats=torch.from_numpy(V)[None].to(dev), Ks=torch.from_numpy(K)[None].to(dev),
                                  width=W, height=H, sh_degree=int(math.isqrt(K_sh) - 1), packed=False)
        r = img[0].clamp(0, 1).cpu().numpy()
        gt = cv2.cvtColor(cv2.imread(str(Path(a.cap) / "images" / c["folder"] / f"{fr:06d}.jpg")), cv2.COLOR_BGR2RGB)
        gt = cv2.undistort(gt, K, dist).astype(np.float32) / 255
        ps.append(-10 * math.log10(max(((r - gt) ** 2).mean(), 1e-12)))
        pair = np.concatenate([gt, r], 1)
        row.append(cv2.resize(pair, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA))
    rows.append(np.concatenate(row, 0) if False else np.concatenate(row, 1))
out = np.concatenate(rows, 0)
cv2.imwrite(a.out, cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"PSNR (unverzerrtes GT vs PLY-Rendering): mittel {np.mean(ps):.2f} dB, min {np.min(ps):.2f}, max {np.max(ps):.2f}, {len(ps)} Bilder")
