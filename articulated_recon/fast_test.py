"""fast-mode robot fit on the real scene: fewer starts/iterations, report time + accuracy."""
import os
import sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from artirecon.data import load_real
from artirecon.fit_robot import RobotFitter, prepare
from artirecon.robot import URDFRobot
obs = load_real()
gts = obs.meta["gt_states"]
for npts, iters, starts in [(20000, 300, 10), (5000, 60, 2), (5000, 30, 1)]:
    rob = URDFRobot(n_pts_per_m2=npts)
    prep = prepare(obs)
    torch.cuda.synchronize(); t0 = time.time()
    rf = RobotFitter(obs, prep, rob, known_base=np.eye(4), log=lambda s: None)
    r = rf.run(iters=iters, n_starts=starts)
    torch.cuda.synchronize(); dt = time.time() - t0
    rob2 = URDFRobot(n_pts_per_m2=20000)
    with torch.no_grad():
        qg = torch.tensor(np.stack([g["q"] for g in gts]), dtype=torch.float32, device="cuda")
        I = torch.eye(4, device="cuda")[None].expand(len(qg), 4, 4)
        Pg, _ = rob2.surface(qg, I)
        Pp, _ = rob2.surface(torch.tensor(r["q"], dtype=torch.float32, device="cuda"), I)
    err = (Pg - Pp).norm(dim=-1).mean().item() * 1000
    print(f"pts/m2 {npts} iters {iters} starts {starts}: robot fit {dt:.1f}s for {obs.T} frames x {obs.V} views "
          f"({dt / obs.T:.2f}s/frame), surface err {err:.1f} mm", flush=True)
