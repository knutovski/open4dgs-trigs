"""Synthetischer Mini-Datensatz fuer trigs-train-colmap (Phase 4).

Szene: statischer 'Raum' (Boden + Rueckwand mit Schachbrettmuster) und ein bewegtes, rotierendes
Objekt (bunte Kugel aus Splats). Ground-Truth-Bilder werden mit mpsplat gerendert.
Ausgabe (wie FreeTimeParser / roma_combine es erwarten):
  <out>/sparse/0/{cameras,images,points3D}.txt   COLMAP-Textmodell, ein Bild pro Kamera (cam00.jpg ...)
  <out>/images/camXX/000000.jpg ...              Frames je Kamera
  <out>/roma/points3d_frameXXXXXX.npy, colors_frameXXXXXX.npy   verrauschte Punkte je Frame
Aufruf: python make_synth_dataset.py <out> [--cams 6 --frames 10 --width 480 --height 270]
"""
import argparse, math, os
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--cams", type=int, default=6); ap.add_argument("--frames", type=int, default=10)
ap.add_argument("--width", type=int, default=480); ap.add_argument("--height", type=int, default=270)
ap.add_argument("--fps", type=float, default=30.0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); rng = np.random.default_rng(a.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------- Szene ----------
def checker(u, v, s=0.25):
    c = ((np.floor(u / s) + np.floor(v / s)) % 2)[:, None]
    return c * np.array([0.85, 0.80, 0.70]) + (1 - c) * np.array([0.25, 0.30, 0.45])
n_floor, n_wall = 20000, 14000
fx_, fz_ = rng.uniform(-2.5, 2.5, n_floor), rng.uniform(-2.5, 2.5, n_floor)
floor = np.stack([fx_, np.full(n_floor, 1.0), fz_], 1)              # y nach unten -> Boden bei y=+1
wx, wy = rng.uniform(-2.5, 2.5, n_wall), rng.uniform(-1.5, 1.0, n_wall)
wall = np.stack([wx, wy, np.full(n_wall, 2.5)], 1)
static_xyz = np.concatenate([floor, wall]).astype(np.float32)
static_rgb = np.concatenate([checker(fx_, fz_), checker(wx, wy, 0.35)]).astype(np.float32)
n_obj = 4000
d = rng.normal(size=(n_obj, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
obj_local = (d * 0.35 * rng.uniform(0.9, 1.0, (n_obj, 1))).astype(np.float32)
obj_rgb = (0.5 + 0.5 * np.stack([np.sign(d[:, 0]), np.sign(d[:, 1]) * 0.6, np.sign(d[:, 2])], 1)).clip(0.05, 0.95).astype(np.float32)

def obj_pose(f):
    t = f / a.fps
    center = np.array([-0.8 + 4.0 * t, 0.45 - 0.6 * math.sin(math.pi * t / 0.33), 0.6], np.float32)  # Bogen
    ang = 2.0 * math.pi * t * 1.5
    R = np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]], np.float32)
    return R, center

def scene(f):
    R, c = obj_pose(f)
    xyz = np.concatenate([static_xyz, obj_local @ R.T + c])
    rgb = np.concatenate([static_rgb, obj_rgb])
    return xyz.astype(np.float32), rgb.astype(np.float32)

N = len(static_xyz) + n_obj
scales = torch.full((N, 3), math.log(0.025)).exp()
scales[len(static_xyz):] = 0.03
quats = torch.tensor([1.0, 0, 0, 0]).repeat(N, 1)
opac = torch.full((N,), 0.9)

# ---------- Kameras (Halbkreis, Blick auf die Mitte) ----------
def look_at(eye, target, up=np.array([0, -1.0, 0])):
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z]); t = -R @ eye                         # world -> camera (COLMAP: x rechts, y unten, z vor)
    return R, t
def rot2quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z
W, H = a.width, a.height
f_px = 0.9 * W
K = np.array([[f_px, 0, W / 2], [0, f_px, H / 2], [0, 0, 1]], np.float32)
cams = []
for i in range(a.cams):
    phi = math.radians(-60 + 120 * i / max(a.cams - 1, 1))
    eye = np.array([4.2 * math.sin(phi), -0.6, 0.6 - 4.2 * math.cos(phi)])
    cams.append(look_at(eye, np.array([0.0, 0.3, 0.8])))

# ---------- COLMAP-Textmodell ----------
sp = out / "sparse" / "0"; sp.mkdir(parents=True, exist_ok=True)
with open(sp / "cameras.txt", "w") as fh:
    for i in range(a.cams):
        fh.write(f"{i+1} PINHOLE {W} {H} {f_px} {f_px} {W/2} {H/2}\n")
with open(sp / "images.txt", "w") as fh:
    for i, (R, t) in enumerate(cams):
        q = rot2quat(R)
        fh.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i+1} cam{i:02d}.jpg\n\n")
sub = rng.choice(len(static_xyz), 3000, replace=False)
with open(sp / "points3D.txt", "w") as fh:
    for j, k in enumerate(sub):
        p, c = static_xyz[k], (static_rgb[k] * 255).astype(int)
        fh.write(f"{j+1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0.5\n")

# ---------- Rendern + Punkte je Frame ----------
roma = out / "roma"; roma.mkdir(exist_ok=True)
Ks = torch.from_numpy(K)[None].to(dev)
for i in range(a.cams):
    (out / "images" / f"cam{i:02d}").mkdir(parents=True, exist_ok=True)
for f in range(a.frames):
    xyz, rgb = scene(f)
    m = torch.from_numpy(xyz).to(dev)
    for i, (R, t) in enumerate(cams):
        V = np.eye(4, dtype=np.float32); V[:3, :3] = R; V[:3, 3] = t
        img, _, _ = rasterization(means=m, quats=quats.to(dev), scales=scales.to(dev), opacities=opac.to(dev),
                                  colors=torch.from_numpy(rgb).to(dev), viewmats=torch.from_numpy(V)[None].to(dev),
                                  Ks=Ks, width=W, height=H, packed=False,
                                  backgrounds=torch.tensor([[0.05, 0.05, 0.08]], device=dev))
        arr = (img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(out / "images" / f"cam{i:02d}" / f"{f:06d}.jpg", quality=95)
    keep = rng.random(len(xyz)) < 0.5                                # 'triangulierte' Teilmenge, verrauscht
    np.save(roma / f"points3d_frame{f:06d}.npy", (xyz[keep] + rng.normal(0, 0.005, (keep.sum(), 3))).astype(np.float32))
    np.save(roma / f"colors_frame{f:06d}.npy", (rgb[keep] * 255).astype(np.float32))
# ---------- Keyframe-NPZ im Format von open4dgs-core/prep/init/prep-roma-trigs-memory.py ----------
# Bekannte Bewegung -> exakte Geschwindigkeit/Beschleunigung (Welt pro Videoframe), Positionen leicht verrauscht.
ks = 3; kf = list(range(0, a.frames, ks)); f0, f1 = 0, a.frames - 1
def obj_world(f):
    R, c = obj_pose(f); return obj_local @ R.T + c
rows = {k: [] for k in ("positions", "colors", "velocities", "accelerations", "times", "is_static", "source_frames")}
st = rng.choice(len(static_xyz), 12000, replace=False)
rows["positions"].append(static_xyz[st]); rows["colors"].append(static_rgb[st])
rows["velocities"].append(np.zeros((len(st), 3))); rows["accelerations"].append(np.zeros((len(st), 3)))
rows["times"].append(np.zeros(len(st))); rows["is_static"].append(np.ones(len(st), bool)); rows["source_frames"].append(np.zeros(len(st)))
h = 0.5
for f in kf:
    sel = rng.random(n_obj) < 0.5
    xm, x0, xp = obj_world(f - h)[sel], obj_world(f)[sel], obj_world(f + h)[sel]
    rows["positions"].append(x0); rows["colors"].append(obj_rgb[sel])
    rows["velocities"].append((xp - xm) / (2 * h)); rows["accelerations"].append((xp - 2 * x0 + xm) / h ** 2)
    rows["times"].append(np.full(sel.sum(), (f - f0) / (f1 - f0))); rows["is_static"].append(np.zeros(sel.sum(), bool))
    rows["source_frames"].append(np.full(sel.sum(), f))
R_ = {k: np.concatenate(v) for k, v in rows.items()}
n = len(R_["positions"]); dyn = ~R_["is_static"]
R_["positions"] = R_["positions"] + rng.normal(0, 0.004, (n, 3))
np.savez_compressed(out / "keyframes.npz",
    positions=R_["positions"].astype(np.float32), colors=R_["colors"].astype(np.float32),
    velocities=R_["velocities"].astype(np.float32), accelerations=R_["accelerations"].astype(np.float32),
    has_acceleration=dyn, track_flags=np.zeros(n, np.uint8),
    times=R_["times"].reshape(-1, 1).astype(np.float32),
    durations=np.full((n, 1), ks / (f1 - f0), np.float32),
    has_velocity=dyn, is_static=R_["is_static"], dynamic_protect=np.zeros(n, bool), dynamic_allowed=dyn,
    stage_keep=np.ones(n, bool),
    frame_start=np.array(f0, np.int32), frame_end=np.array(f1, np.int32), n_keyframes=np.array(len(kf), np.int32),
    keyframe_step=np.array(ks, np.int32), source_frame_start=np.array(f0, np.int32), source_frame_end=np.array(f1, np.int32),
    source_frames=R_["source_frames"].reshape(-1, 1).astype(np.int32), keyframe_frames=np.array(kf, np.int32),
    fps=np.array(a.fps, np.float32), velocity_units=np.array("world_per_video_frame"),
    time_units=np.array("normalized_0_1_over_source_frame_range"), mode=np.array("trigs_roma_hybrid_static_dynamic"))
print(f"keyframes.npz: {n} Zeilen ({int(dyn.sum())} dynamisch, Keyframes {kf})")
print(f"OK: {a.cams} Kameras x {a.frames} Frames, {W}x{H}, {N} Splats, Geraet {dev} -> {out}")
