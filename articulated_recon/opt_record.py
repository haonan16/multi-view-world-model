"""Record the optimisation of one episode for videos: robot frame-0 multi-hypothesis joint fit and the
object stage (shape + pose hypotheses, refinement, per-frame tracking).  -> results/real_v2/<ep>/optrec.pkl
python opt_record.py <episode> [object name]"""
import copy, json, os, pickle, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(ROOT))
import artirecon  # noqa
from artirecon.data import load_real
from artirecon.fit_robot import RobotFitter, prepare, PANDA_READY
from artirecon.objects_v2 import ObjectsV2, Sam3Masks
from artirecon.robot import URDFRobot

EP = sys.argv[1]; OBJ = sys.argv[2] if len(sys.argv) > 2 else None
robot = URDFRobot(n_pts_per_m2=20000)
obs = load_real("v2/" + EP)
R = pickle.load(open(ROOT / "results/real_v2" / EP / "robot.pkl", "rb"))
o = copy.copy(obs); o.c2w = R["c2w"]
o.depth = obs.depth * R["S"][:, :, None, None].astype(np.float32)
# ---- robot: frame-0 joint hypotheses (same as RobotFitter.run, known base), recorded
prep = prepare(o, fg_thr=max(0.008, 3 * R["noise"]))
rf = RobotFitter(o, prep, robot, calib=False, known_base=np.eye(4)); rf.rec = []
g = torch.Generator(device="cpu").manual_seed(0)
ready = torch.tensor(PANDA_READY[:8], device="cuda"); Hn = 24
q8 = ready[None].repeat(Hn, 1) + torch.randn(Hn, 8, generator=g).cuda() * 0.4 * (torch.arange(Hn, device="cuda") > 0)[:, None]
q8 = torch.max(torch.min(q8, rf.hi[:8]), rf.lo[:8])
yaw = torch.zeros(Hn, device="cuda"); xy = torch.zeros(Hn, 2, device="cuda")
rf.optimise(yaw, xy, q8, [0] * Hn, iters=300, opt_base=False)
robrec = rf.rec
# ---- objects
if R["rob_scale"] is not None:
    o.depth = o.depth * R["rob_scale"][:, :, None, None].astype(np.float32)
prep2 = dict(n=R["prep_n"], d=R["prep_d"])
ann = json.load(open(ROOT / "real/v2" / EP / "ann.json"))
for spec in ann["objects"]:
    mp = ROOT / "real/v2" / EP / "sam3d" / (spec["name"] + "_20k.ply")
    if spec.get("shape") in ("hull", "mesh") and mp.exists():
        spec.update(shape="mesh", mesh=str(mp), mesh_json=str(mp).replace("_20k.ply", ".json"))
names = [x["name"] for x in ann["objects"]]
OBJ = OBJ or next(x["name"] for x in ann["objects"] if x.get("role") == "moved")
ann["objects"] = [x for x in ann["objects"] if x["name"] == OBJ]
cf = ROOT / "real/v2" / EP / "color_offset.json"
COFF = {int(k): v for k, v in json.load(open(cf))["offset_px"].items()} if cf.exists() else {}
mf = [ROOT / "real/v2" / EP / "sam3.npz"] + ([ROOT / "real/v2" / EP / "sam3_track.npz"] if (ROOT / "real/v2" / EP / "sam3_track.npz").exists() else [])
sam = Sam3Masks(mf, obs.rgb.shape[2], obs.rgb.shape[3], offsets=COFF)
ob = ObjectsV2(o, prep2, robot, R["rob"], sam, ann, log=print); ob.rec = []
objs = ob.run()
pickle.dump(dict(robot=robrec, objects=ob.rec, obj=OBJ, c2w=o.c2w, K=obs.K, final=objs[0]["poses"], held=objs[0]["held"],
                 gt_q=np.stack([g_["q"] for g_ in obs.meta["gt_states"]])),
            open(ROOT / "results/real_v2" / EP / "optrec.pkl", "wb"))
print("recorded", len(robrec), "robot snapshots,", len(ob.rec), "object snapshots")
