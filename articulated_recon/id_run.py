"""Inverse dynamics from our 3D reconstructions, executed in ManiSkill (StackCube-v1).

Method of Li et al. 2026 (arXiv 2607.01166, github.com/sky-lzy/Structured-4D-Model): subgoal 3D scenes -> actions.
  --mode learned : their released goal-conditioned inverse-dynamics model (diffusion, point clouds of the current
                   and the goal scene from 4 fixed cameras + qpos -> 64 joint-position actions), closed loop.
                   Goal point cloud = OUR reconstruction (fitted robot joints + fitted cube poses) placed in a twin
                   ManiSkill scene and seen by the same 4 cameras (same unprojection / bbox / 4096-pt sampling).
  --mode retarget: their learning-free variant: end-effector SE(3) pose per subgoal (here from the reconstructed
                   robot, forward kinematics; no registration needed) -> motion planner (mplib, screw then RRT);
                   gripper closes / opens when the reconstruction says the red cube is held / released.
Goal source (--goal):
  oracle  : the demo's true sim states at the same frames (upper bound for the subgoal spacing)
  recon   : fit.pkl of our reconstruction (results/wm/<name>/fit.pkl)
The scene is reset with the demo episode's seed, so the initial state is the episode's frame 0 (the input).
python id_run.py --mode learned|retarget --goal oracle|recon --list L [--fit_tag _gt_raw_idgt] [--out results/id/x]
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import gymnasium as gym
import sapien
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
S4D = Path("/n/lab_storage/ydu_lab/Lab/akiruga/ext/Structured-4D-Model")
sys.path.insert(0, str(S4D))
from policy.env_utils.env_wrappers import create_custom_env  # noqa: E402
from policy.common.pointcloud_utils import downsample_pointcloud  # noqa: E402

MSR = Path("/n/lab_storage/ydu_lab/Lab/akiruga/maniskill_render")
ROLL = Path("/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs/rollouts_A7000")
BBOX = [-0.85, -0.6, -0.2, 0.35, 0.6, 1.0]
STEP = 20                                     # rollout / reconstruction frames are every 20 control steps

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="learned"); ap.add_argument("--goal", default="oracle")
ap.add_argument("--list", required=True); ap.add_argument("--fit_tag", default="_gt_raw_idgt")
ap.add_argument("--out", default=None); ap.add_argument("--act_h", type=int, default=20)
ap.add_argument("--reps", type=int, default=1); ap.add_argument("--final_reps", type=int, default=3)
ap.add_argument("--trials", type=int, default=1); ap.add_argument("--video", type=int, default=1)
ap.add_argument("--hard_gripper", type=int, default=1)
ap.add_argument("--snap", type=int, default=0)
ap.add_argument("--stop_on_success", type=int, default=1)
ap.add_argument("--static", type=int, default=0, help="a cube that is not held does not move: it keeps its last pose")
ap.add_argument("--approach", type=int, default=1, help="0: skip the pre-grasp end-effector waypoints (they come from "
                "the robot fit, the least reliable part on generated views) and go straight to the pre-grasp pose")
ap.add_argument("--grasp_prior", type=int, default=0, help="held cube sits between the fingertips: in-hand transform = "
                "cube centre, top-down, fingers closing on the cube faces nearest the reconstructed gripper's")
ap.add_argument("--anchor", default="", help="fit tag of the reconstruction of the CURRENT observation (frame 0, all "
                "cameras), e.g. _gt_raw_idgt: predicted goals are translated so frame 0 matches it (per object; robot "
                "end-effector targets follow the red cube) -- the prediction gives the plan, the observation the where")
a = ap.parse_args()
OUT = Path(a.out or ROOT / "results/id" / f"{a.mode}_{a.goal}{a.fit_tag if a.goal == 'recon' else ''}")
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
torch.set_grad_enabled(False)

create_custom_env("StackCube-v1")
mk = lambda: gym.make("MyDefineEnv", num_extra_cameras=4, obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                      fix_camera=True, render_mode="rgb_array", max_episode_steps=1000)
env, twin = mk(), mk()
U, Ut = env.unwrapped, twin.unwrapped
H5 = h5py.File(MSR / "demos/StackCube-v1/trajectory.h5", "r")
EPI = {e["episode_id"]: e for e in json.load(open(MSR / "demos/StackCube-v1/trajectory.json"))["episodes"]}


# ------------------------------------------------------------------ point clouds exactly as in their eval.py
def pointcloud(obs):
    keys = sorted(k for k in obs["sensor_data"] if "render_camera" in k)
    pts = []
    for k in keys:
        dep = obs["sensor_data"][k]["depth"][0, ..., 0].float().to(dev) / 1000.0
        K = obs["sensor_param"][k]["intrinsic_cv"][0].float().to(dev)
        E = obs["sensor_param"][k]["extrinsic_cv"][0].float().to(dev)
        h, w = dep.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
        X = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(-1, 3) @ torch.linalg.inv(K).T * dep.view(-1, 1)
        Xw = (X - E[:3, 3]) @ torch.linalg.inv(E[:3, :3]).T
        m = (Xw[:, 0] >= BBOX[0]) & (Xw[:, 0] <= BBOX[3]) & (Xw[:, 1] >= BBOX[1]) & (Xw[:, 1] <= BBOX[4]) & \
            (Xw[:, 2] >= BBOX[2]) & (Xw[:, 2] <= BBOX[5])
        pts.append(Xw[m])
    return downsample_pointcloud(torch.cat(pts), num_points=4096)


def set_scene(E, q, poses):
    """put the robot at joints q (9) and the cubes at poses {actor: 4x4} in env E, return its observation."""
    E.agent.robot.set_qpos(torch.as_tensor(np.asarray(q, np.float32))[None])
    for name, T in poses.items():
        actor = {"cubeA": E.cubeA, "cubeB": E.cubeB}[name]
        qw = Rot.from_matrix(T[:3, :3]).as_quat()
        actor.set_pose(sapien.Pose(p=T[:3, 3].astype(np.float32), q=np.r_[qw[3], qw[:3]].astype(np.float32)))
    E.scene.update_render()
    return E.get_obs()


def demo_goals(eid, steps):
    g = H5[f"traj_{eid}"]["env_states"]
    art = g["articulations"]; rk = [k for k in art if k.startswith("panda")][0]
    A = art[rk][:]; n = len(A); st = np.clip(steps, 0, n - 1)
    out = []
    for s in st:
        P = {}
        for nm in ("cubeA", "cubeB"):
            x = g["actors"][nm][s]
            T = np.eye(4); T[:3, :3] = Rot.from_quat([x[4], x[5], x[6], x[3]]).as_matrix(); T[:3, 3] = x[:3]
            P[nm] = T
        out.append(dict(q=A[s, 13:22], poses=P, held=bool(A[s, 20] < 0.03)))
    if a.static:
        static(out)
    return out


def recon_goals(name):
    F = pickle.load(open(ROOT / "results/wm" / (name + a.fit_tag) / "fit.pkl", "rb"))
    q = F["rob"]["q"]; T = F["T"]
    ob = {o["name"]: o for o in F["objects"]}
    out = []
    for t in range(T):
        P = {}
        for nm, act in (("red_cube", "cubeA"), ("green_cube", "cubeB")):
            if nm not in ob:
                continue
            Tp = ob[nm]["poses"][t].copy()
            c = ob[nm]["mesh"][0].mean(0)
            Tp[:3, 3] = Tp[:3, :3] @ c + Tp[:3, 3]
            P[act] = Tp
        held = bool(ob["red_cube"]["held"][t]) if "red_cube" in ob else None
        out.append(dict(q=q[t], poses=P, held=held))
    if a.anchor:
        anchor(out, name)
    if a.snap:
        snap(out)
    if a.static:
        static(out)
    return out


def static(out):
    """a cube that is not held does not move: green keeps its frame-0 pose, red until it is first held/lifted."""
    z = np.array([g["poses"]["cubeA"][2, 3] for g in out]); rest = np.minimum.accumulate(z)
    held = [bool(g["held"]) or (z[k] - rest[k] > 0.03) for k, g in enumerate(out)]
    kg = held.index(True) if True in held else len(out)
    for k in range(1, len(out)):
        out[k]["poses"]["cubeB"] = out[0]["poses"]["cubeB"]
        if k < kg:
            out[k]["poses"]["cubeA"] = out[0]["poses"]["cubeA"]


def anchor(goals, name):
    ep = name.split("_s")[0]
    src = [x for x in open(ROOT / "results/id/gt_list.txt").read().split() if x.split("_s")[0] == ep][0]
    obs0 = recon_goals_raw(src + a.anchor)[0]
    d = {nm: obs0["poses"][nm][:3, 3] - goals[0]["poses"][nm][:3, 3] for nm in ("cubeA", "cubeB")}
    for k, g in enumerate(goals):
        for nm in ("cubeA", "cubeB"):
            g["poses"][nm] = g["poses"][nm].copy(); g["poses"][nm][:3, 3] += d[nm]
        if k == 0:
            g["q"] = obs0["q"].copy(); g["poses"] = {nm: obs0["poses"][nm].copy() for nm in ("cubeA", "cubeB")}; continue
        Tt = T_of(tcp_of(g["q"])); Tt[:3, 3] += d["cubeA"]
        qn, ok = ik(Tt, g["q"]); qn[7:] = g["q"][7:]
        g["q"] = qn
    goals[0]["anchor_shift_mm"] = {k: float(np.linalg.norm(v) * 1000) for k, v in d.items()}


def recon_goals_raw(fname):
    F = pickle.load(open(ROOT / "results/wm" / fname / "fit.pkl", "rb"))
    ob = {o["name"]: o for o in F["objects"]}
    out = []
    for t in range(F["T"]):
        P = {}
        for nm, act in (("red_cube", "cubeA"), ("green_cube", "cubeB")):
            Tp = ob[nm]["poses"][t].copy(); c = ob[nm]["mesh"][0].mean(0)
            Tp[:3, 3] = Tp[:3, :3] @ c + Tp[:3, 3]; P[act] = Tp
        out.append(dict(q=F["rob"]["q"][t], poses=P, held=bool(ob["red_cube"]["held"][t])))
    return out


def snap(goals, h=0.02):
    """contact rule on the reconstruction: a cube that is not held rests on the table (z = 0 plane, known) or on
    the other cube if it is above it; resting cubes are levelled (yaw kept). h: cube half size."""
    z0 = goals[0]["poses"]["cubeA"][2, 3]
    for k, g in enumerate(goals):
        lifted = g["poses"]["cubeA"][2, 3] - min(x["poses"]["cubeA"][2, 3] for x in goals[:k + 1]) > 0.03
        for nm in ("cubeB", "cubeA"):
            if nm == "cubeA" and (g["held"] or lifted):
                continue
            T = g["poses"][nm].copy()
            yaw = np.arctan2(T[1, 0], T[0, 0]); T[:3, :3] = Rot.from_euler("z", yaw).as_matrix()
            o = g["poses"]["cubeB"]
            on = nm == "cubeA" and np.linalg.norm(T[:2, 3] - o[:2, 3]) < 0.03 and T[2, 3] > o[2, 3] + 0.025
            T[2, 3] = o[2, 3] + 2 * h if on else h
            g["poses"][nm] = T


def tcp_of(q):
    """end-effector SE(3) of joints q: forward kinematics in the twin scene."""
    Ut.agent.robot.set_qpos(torch.as_tensor(np.asarray(q, np.float32))[None])
    p = Ut.agent.tcp.pose
    return sapien.Pose(p=p.p[0].cpu().numpy(), q=p.q[0].cpu().numpy())


def T_of(P):
    T = np.eye(4); T[:3, :3] = Rot.from_quat(np.r_[P.q[1:], P.q[0]]).as_matrix(); T[:3, 3] = P.p
    return T


def P_of(T):
    x = Rot.from_matrix(T[:3, :3]).as_quat()
    return sapien.Pose(p=T[:3, 3].astype(np.float32), q=np.r_[x[3], x[:3]].astype(np.float32))


def make_plan(goals):
    """subgoal frames -> grasp / carry / release plan, all from the (reconstructed) frames.
    held_k = reconstruction's held flag OR red cube lifted > 3 cm above where it rested.
    The frames stop before the release (steps 0..100, release at ~105-130), so the plan completes the last
    held frame: the red cube is centred on and lowered onto the green cube's top and the gripper opens; the end effector
    follows through the in-hand transform of the last held frame. The grasp end-effector pose is the in-hand
    transform of the first held frame applied to the cube's pose in the frame before it (where it rested)."""
    z = np.array([g["poses"]["cubeA"][2, 3] for g in goals])
    rest = np.minimum.accumulate(z)
    held = np.array([bool(g["held"]) or (z[k] - rest[k] > 0.03) for k, g in enumerate(goals)])
    held[0] = False
    tcp = [T_of(tcp_of(g["q"])) for g in goals]
    if not held.any():
        return dict(held=held, tcp=tcp, ok=False)
    kg = int(np.argmax(held)); L = int(np.nonzero(held)[0].max())
    if a.grasp_prior:
        # grasp on the resting cube (frame before the first held frame): top-down, centre, fingers on the face pair
        # needing the least wrist rotation from the current gripper; the cube does not turn in the hand, so carry
        # targets keep that orientation and follow the predicted cube positions.
        Cg = goals[kg - 1]["poses"]["cubeA"]
        ap_ = np.array([0, 0, -1.0])
        y0 = T_of(tcp_of(goals[0]["q"]))[:3, 1]; y0[2] = 0; y0 /= np.linalg.norm(y0) + 1e-9
        yaw = np.arctan2(Cg[1, 0], Cg[0, 0])
        cl = max([np.array([np.cos(yaw + j * np.pi / 2), np.sin(yaw + j * np.pi / 2), 0]) for j in range(4)], key=lambda v: v @ y0)
        Rg = np.stack([np.cross(cl, ap_), cl, ap_], 1)
        grasp = np.eye(4); grasp[:3, :3] = Rg; grasp[:3, 3] = Cg[:3, 3]
        inhand = np.linalg.inv(Cg) @ grasp
        for k in np.nonzero(held)[0]:
            tcp[k] = np.eye(4); tcp[k][:3, :3] = Rg; tcp[k][:3, 3] = goals[k]["poses"]["cubeA"][:3, 3]
        L = int(np.nonzero(held)[0].max()); Tg = goals[L]["poses"]["cubeB"]
        place = Cg.copy(); place[:2, 3] = Tg[:2, 3]; place[2, 3] = Tg[2, 3] + 0.042
        return dict(held=held, tcp=tcp, ok=True, kg=kg, L=L, grasp=grasp, release=place @ inhand, place=place)
    inhand_g = np.linalg.inv(goals[kg]["poses"]["cubeA"]) @ tcp[kg]
    grasp = goals[kg - 1]["poses"]["cubeA"] @ inhand_g
    Tc = goals[L]["poses"]["cubeA"].copy(); Tg = goals[L]["poses"]["cubeB"]
    place = Tc.copy(); place[:2, 3] = Tg[:2, 3]; place[2, 3] = Tg[2, 3] + 0.02 + 0.02 + 0.002   # centred on green, half sizes + 2 mm
    release = place @ np.linalg.inv(Tc) @ tcp[L]
    return dict(held=held, tcp=tcp, ok=True, kg=kg, L=L, grasp=grasp, release=release, place=place)


def ik(Tw, q0):
    """joint angles for an end-effector world pose (pinocchio, robot root frame)."""
    rob = Ut.agent.robot
    pm = rob._objs[0].create_pinocchio_model()
    links = [l.name for l in rob._objs[0].get_links()]
    Tb = T_of(sapien.Pose(p=rob.pose.p[0].cpu().numpy(), q=rob.pose.q[0].cpu().numpy()))
    q, ok, err = pm.compute_inverse_kinematics([i for i, l in enumerate(links) if l.endswith("panda_hand_tcp")][0], P_of(np.linalg.inv(Tb) @ Tw),
                                               initial_qpos=np.asarray(q0, np.float64), active_qmask=np.r_[np.ones(7), 0, 0].astype(int),
                                               max_iterations=500)
    return np.asarray(q), bool(ok)


def success(info):
    s = info.get("success", False)
    return bool(s.item() if hasattr(s, "item") else s)


# ------------------------------------------------------------------ learned inverse dynamics (their checkpoint)
policy = None
if a.mode == "learned":
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from policy.workspace.robotworkspace import RobotWorkspace
    from policy.dataset.hdf5_dataset import DATA_STAT
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(S4D / "policy/config"), version_base=None):
        cfg = compose(config_name="inverse_dynamics")
    ws = RobotWorkspace(cfg)
    ws.load_checkpoint(S4D / ".ckpt/structured-4d-model/inverse_dynamics/inverse_dynamics.ckpt")
    policy = ws.model.cuda().eval()
    DS = {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in DATA_STAT.items()}

    def predict(cur, goal, qpos):
        st = (qpos.to(dev) - DS["state_min"]) / (DS["state_max"] - DS["state_min"]) * 2 - 1
        o = {"pointcloud": torch.stack([cur, goal])[None], "agent_pos": st[:, :8][None]}
        act = policy.predict_action(o)["action"][0]
        act = ((act + 1) / 2 * (DS["action_max"] - DS["action_min"]) + DS["action_min"]).cpu().numpy()
        if a.hard_gripper:
            act[:, 7] = np.where(act[:, 7] > 0, 1.0, -1.0)
        return act


def run_learned(goals, frames):
    obs = U.get_obs()
    info = {}; ever = False
    plan = make_plan(goals)
    scenes = [(g["q"], g["poses"]) for g in goals]
    if plan["ok"] and a.grasp_prior:
        # goal scenes follow the object-centric plan: pre-grasp (open, 5 cm above), grasp (closed), carry frames
        # (hand on the cube), release (open, cube on green); robot joints by IK of the plan's end-effector poses
        kg, L = plan["kg"], plan["L"]
        upT = lambda T, h: np.r_[np.c_[T[:3, :3], T[:3, 3] + [0, 0, h]], [[0, 0, 0, 1]]]
        rest = dict(goals[kg - 1]["poses"])
        q_pre, _ = ik(upT(plan["grasp"], 0.05), goals[0]["q"]); q_pre[7:] = 0.04
        q_gr, _ = ik(plan["grasp"], q_pre); q_gr[7:] = 0.04
        q_cl = q_gr.copy(); q_cl[7:] = 0.02
        sc = [(goals[0]["q"], goals[0]["poses"])] + ([(g["q"], g["poses"]) for g in goals[1:kg]] if a.approach else [])
        sc += [(q_pre, rest), (q_gr, rest), (q_cl, rest)]
        qk = q_cl
        for k in range(kg, L + 1):
            qk, _ = ik(plan["tcp"][k], qk); qk = qk.copy(); qk[7:] = 0.02
            sc.append((qk, goals[k]["poses"]))
        P = dict(goals[L]["poses"], cubeA=plan["place"])
        q_rel, _ = ik(plan["release"], qk); q_rel[7:] = 0.02
        q_open = q_rel.copy(); q_open[7:] = 0.04
        sc += [(q_rel, P), (q_open, P)]
        scenes = sc
    elif plan["ok"]:
        scenes = scenes[:plan["L"] + 1]
        qr, okk = ik(plan["release"], goals[plan["L"]]["q"])
        qr[7:] = 0.04                                                        # gripper open
        scenes.append((qr, dict(goals[plan["L"]]["poses"], cubeA=plan["place"])))
    seq = [k for k in range(1, len(scenes)) for _ in range(a.reps)] + [len(scenes) - 1] * a.final_reps
    gpcs = [pointcloud(set_scene(Ut, q, P)) for q, P in scenes]
    for k in seq:
        act = predict(pointcloud(obs), gpcs[k], obs["agent"]["qpos"])
        for x in act[:a.act_h]:
            obs, _, term, trunc, info = env.step(x)
            frames.append(env.render()[0].cpu().numpy())
            ever |= success(info)
            if ever and a.stop_on_success:              # as in their eval.py: the episode ends at success
                return True, True
    return success(info), ever


def run_retarget(goals, frames):
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    pl = PandaArmMotionPlanningSolver(env, debug=False, vis=False, base_pose=U.agent.robot.pose,
                                      visualize_target_grasp_pose=False, print_env_info=False)
    orig_step = env.step

    def step(x):
        r = orig_step(x)
        frames.append(env.render()[0].cpu().numpy())
        return r
    env.step = step
    info = {}; ever = False; fails = 0
    plan = make_plan(goals)

    def go(T):
        nonlocal fails, info, ever
        r = pl.move_to_pose_with_screw(P_of(T))
        if r == -1:
            r = pl.move_to_pose_with_RRTConnect(P_of(T))
        if r == -1:
            fails += 1
        elif isinstance(r, tuple):
            info = r[-1]; ever |= success(info)
    up = lambda T, h: np.r_[np.c_[T[:3, :3], T[:3, 3] + [0, 0, h]], [[0, 0, 0, 1]]]
    try:
        if not plan["ok"]:
            return False, False, -1
        for k in (range(1, plan["kg"]) if a.approach else []):   # approach frames (gripper open)
            go(plan["tcp"][k])
        go(up(plan["grasp"], 0.05)); go(plan["grasp"]); pl.close_gripper()
        for k in range(plan["kg"], plan["L"] + 1):            # carry frames
            go(plan["tcp"][k])
        go(up(plan["release"], 0.05)); go(plan["release"]); pl.open_gripper(); go(up(plan["release"], 0.06))
        for _ in range(10):                                  # settle
            r = env.step(np.r_[U.agent.robot.get_qpos()[0, :7].cpu().numpy(), 0.04])
            info = r[-1]; ever |= success(info)
    finally:
        env.step = orig_step
        pl.close()
    return success(info), ever, int(fails)


# ------------------------------------------------------------------ main loop
import imageio.v3 as iio
names = open(a.list).read().split()
res = []
for n in names:
    m = json.load(open(ROLL / n / "meta.json"))
    eid = int(m["episode"].split("episode-")[1])
    seed = EPI[eid]["episode_seed"]
    T = 6
    frames_steps = np.arange(T) * STEP
    if a.goal == "oracle":
        goals = demo_goals(eid, frames_steps)
    else:
        if not (ROOT / "results/wm" / (n + a.fit_tag) / "fit.pkl").exists():
            print("no fit", n, flush=True); continue
        goals = recon_goals(n)
    for tr in range(a.trials):
        torch.manual_seed(tr); np.random.seed(tr)
        obs, _ = env.reset(seed=seed)
        twin.reset(seed=seed)
        if tr == 0:                 # diagnostics only (true sim poses are never used for control)
            pl_ = make_plan(goals)
            ra, gb = U.cubeA.pose.p[0].cpu().numpy(), U.cubeB.pose.p[0].cpu().numpy()
            diag = dict(grasp_err_mm=round(float(np.linalg.norm(pl_["grasp"][:3, 3] - ra) * 1000), 1) if pl_["ok"] else None,
                        place_xy_err_mm=round(float(np.linalg.norm(pl_["place"][:2, 3] - gb[:2]) * 1000), 1) if pl_["ok"] else None,
                        inhand_mm=round(float(np.linalg.norm((np.linalg.inv(goals[pl_["kg"]]["poses"]["cubeA"]) @ pl_["tcp"][pl_["kg"]])[:3, 3]) * 1000), 1) if pl_["ok"] else None,
                        kg=pl_.get("kg"), L=pl_.get("L"), shift=goals[0].get("anchor_shift_mm"))
        x0 = H5[f"traj_{eid}"]["env_states"]["actors"]["cubeA"][0, :3]
        d0 = float(np.linalg.norm(U.cubeA.pose.p[0].cpu().numpy() - x0) * 1000)
        frames = [env.render()[0].cpu().numpy()]
        if a.mode == "learned":
            ok, ever = run_learned(goals, frames); fails = 0
        else:
            ok, ever, fails = run_retarget(goals, frames)
        r = dict(name=n, episode=m["episode"], trial=tr, success=ok, ever=ever, steps=len(frames) - 1, plan_fail=fails,
                 init_cube_mm=round(d0, 2), **diag)
        print(json.dumps(r), flush=True)
        res.append(r)
        if a.video and tr == 0:
            iio.imwrite(OUT / f"{n}.mp4", np.stack(frames[::2]), fps=15, codec="libx264")
summ = dict(n=len(res), success=sum(r["success"] for r in res), ever=sum(r["ever"] for r in res))
print("SUMMARY", json.dumps(summ), flush=True)
json.dump(dict(args=vars(a), runs=res, summary=summ), open(OUT / "results.json", "w"), indent=1)
