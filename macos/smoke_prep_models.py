"""Phase 5: RoMa v2 und MEMFOF auf mps (bzw. cpu) laden und an data/synth01 pruefen.
RoMa v2: Matches cam00<->cam01 (Frame 0), Epipolarfehler gegen die bekannten Kameras.
MEMFOF: Fluss cam00 Frames 0/1/2, Median-Fluss Hintergrund vs. Gesamt.
"""
import sys, time, math
from pathlib import Path
import numpy as np, torch, cv2
dev = sys.argv[1] if len(sys.argv) > 1 else ("mps" if torch.backends.mps.is_available() else "cpu")
root = Path(__file__).resolve().parents[2]  # workspace (parent of open4dgs-trigs)
sys.path.insert(0, str(root / "open4dgs-core/prep/init"))
from roma_backends import create_roma_backend, MemfofTemporalBackend
D = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "data/synth01"
img = lambda c, f: D / "images" / f"cam{c:02d}" / f"{f:06d}.jpg"
rgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)

def cams():
    K = {int(l.split()[0]): list(map(float, l.split()[4:8])) for l in open(D / "sparse/0/cameras.txt") if l.strip()}
    out = {}
    for l in open(D / "sparse/0/images.txt"):
        s = l.split()
        if len(s) >= 10:
            w, x, y, z = map(float, s[1:5])
            R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
            fx, fy, cx, cy = K[int(s[8])]
            out[s[9].split(".")[0]] = (R, np.array(list(map(float, s[5:8]))), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    return out
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

print(f"Geraet {dev}, torch {torch.__version__}, MPS-Fallback {__import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
t0 = time.time(); roma = create_roma_backend(backend="v2", device=dev); print(f"RoMa v2 geladen ({roma.name}) in {time.time()-t0:.1f} s")
a, b = rgb(img(0, 0)), rgb(img(1, 0))
for rep in range(2):
    t0 = time.time(); dense = roma.build_dense(img(0, 0), img(1, 0), a, b)
    pa, pb, cert = roma.sample_points_from_dense(dense, num=5000)
    if dev == "mps": torch.mps.synchronize()
    print(f"  RoMa v2 Match {a.shape[1]}x{a.shape[0]} Durchlauf {rep+1}: {time.time()-t0:.2f} s, {len(pa)} Punkte, Zertifikat median {np.median(cert):.3f}")
C = cams(); (R0, t0_, K0), (R1, t1, K1) = C["cam00"], C["cam01"]
R = R1 @ R0.T; t = t1 - R @ t0_
F = np.linalg.inv(K1).T @ skew(t) @ R @ np.linalg.inv(K0)
ha, hb = np.c_[pa, np.ones(len(pa))], np.c_[pb, np.ones(len(pb))]
l = ha @ F.T; err = np.abs((hb * l).sum(1)) / np.hypot(l[:, 0], l[:, 1])
good = cert > 0.5
print(f"  Epipolarfehler (px) aller Matches: median {np.median(err):.2f}, 90% {np.percentile(err, 90):.2f}; "
      f"bei Zertifikat>0.5 ({good.sum()}): median {np.median(err[good]) if good.any() else float('nan'):.2f}")
del roma
if dev == "mps": torch.mps.empty_cache()
t0 = time.time(); mf = MemfofTemporalBackend(device=dev); print(f"MEMFOF geladen ({mf.name}) in {time.time()-t0:.1f} s")
for rep in range(2):
    t0 = time.time(); bwd, fwd, sc = mf.triplet_flows(rgb(img(0, 0)), rgb(img(0, 1)), rgb(img(0, 2)))
    print(f"  MEMFOF Durchlauf {rep+1}: {time.time()-t0:.2f} s, Flussfeld {fwd.shape}")
mag = np.hypot(*fwd.astype(np.float32))
print(f"  |Fluss vorwaerts| px: median {np.median(mag):.2f} (Hintergrund ~0 erwartet), 99% {np.percentile(mag, 99):.2f} (bewegte Kugel)")
if dev == "mps": print(f"MPS-Treiberspeicher: {torch.mps.driver_allocated_memory()/2**30:.2f} GiB")
print("SMOKE_OK")
