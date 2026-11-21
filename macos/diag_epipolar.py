"""Epipolarfehler von RoMa-v2-Matches gegen reference_export.json (verzeichnungskorrigiert), je Kamerapaar.
Aufruf: python diag_epipolar.py <capture_dir> [frame] [setting: fast|precise]"""
import sys, json, itertools
from pathlib import Path
import numpy as np, cv2, torch
C = Path(sys.argv[1]); F = int(sys.argv[2]) if len(sys.argv) > 2 else 1; S = sys.argv[3] if len(sys.argv) > 3 else "precise"
root = Path(__file__).resolve().parents[2]  # workspace (parent of open4dgs-trigs)
sys.path.insert(0, str(root / "open4dgs-core/prep/init"))
from roma_backends import create_roma_backend
ref = json.load(open(C / "reference_export.json"))["cameras"]
def cam(c):
    p = c["params"]; m = c["camera_model_name"]
    if m == "RADIAL": K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]]); d = np.array([p[3], p[4], 0, 0])
    elif m == "OPENCV": K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]]); d = np.array(p[4:8])
    else: raise SystemExit(m)
    R = np.array(c["R"]) if "R" in c else cv2.Rodrigues(np.array(c["rvec"]))[0]
    return K, d, R, np.array(c["t"])
print("Schluessel:", [k for k in ref[0].keys()])
cams = {c["folder"]: cam(c) for c in ref}
names = sorted(cams)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
roma = create_roma_backend(backend="v2", device=dev, v2_setting=S)
img = lambda n: C / "images" / n / f"{F:06d}.jpg"
rgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
allerr = []
for a, b in itertools.combinations(names, 2):
    ia, ib = rgb(img(a)), rgb(img(b))
    dense = roma.build_dense(img(a), img(b), ia, ib)
    pa, pb, cert = roma.sample_points_from_dense(dense, num=6000)
    keep = cert > 0.5; pa, pb = pa[keep], pb[keep]
    (Ka, da, Ra, ta), (Kb, db, Rb, tb) = cams[a], cams[b]
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2).astype(np.float64), Ka, da).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2).astype(np.float64), Kb, db).reshape(-1, 2)
    R = Rb @ Ra.T; t = tb - R @ ta; E = skew(t) @ R
    ha, hb = np.c_[na, np.ones(len(na))], np.c_[nb, np.ones(len(nb))]
    l = ha @ E.T; e = np.abs((hb * l).sum(1)) / np.hypot(l[:, 0], l[:, 1]) * Kb[0, 0]   # in px (Kamera b)
    allerr.append(e)
    print(f"{a}-{b}: {len(e)} Matches (cert>0.5) | Epipolarfehler px median {np.median(e):.2f}, p75 {np.percentile(e,75):.2f}, p90 {np.percentile(e,90):.2f}, Anteil >2.5px {100*(e>2.5).mean():.0f}%")
e = np.concatenate(allerr)
print(f"GESAMT ({S}, Frame {F}): median {np.median(e):.2f} px, p90 {np.percentile(e,90):.2f}, >2.5px {100*(e>2.5).mean():.0f}%")
