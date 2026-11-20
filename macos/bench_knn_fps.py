"""Phase 2: Laufzeit knn_points / sample_farthest_points (trigs.compat.ops) auf dem Mac."""
import time, torch
from trigs.compat.ops import knn_points_torch, sample_farthest_points_torch

def t(fn, dev):
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter(); out = fn()
    if dev == "mps": torch.mps.synchronize()
    return time.perf_counter() - t0, out

torch.manual_seed(0)
for dev in ["mps"]:
    for n in [1_000_000, 4_000_000, 100_000]:
        p = (torch.randn(1, n, 3) * 2.0).to(dev)
        knn_points_torch(p[:, :1000], p[:, :1000], K=11)  # warm-up
        s, _ = t(lambda: knn_points_torch(p, p, K=11), dev)
        print(f"knn  self K=11  N={n:>9,}  {dev}: {s:8.2f} s")
for dev in ["mps"]:
    p = (torch.randn(1, 1_000_000, 3) * 2.0).to(dev)
    for k in [1_000, 10_000]:
        s, _ = t(lambda: sample_farthest_points_torch(p, K=k), dev)
        print(f"fps  N=1,000,000 -> K={k:>6,}  {dev}: {s:8.2f} s  ({1000*s/k:.2f} ms/Punkt)")
if torch.backends.mps.is_available():
    print(f"MPS-Speicher (driver) nach Lauf: {torch.mps.driver_allocated_memory()/2**30:.2f} GiB")
print("BENCH_OK")
