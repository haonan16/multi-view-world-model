"""Evaluator: compare a fit (fit.pkl) with simulator ground truth. Uses evaluator_only data."""
import numpy as np
import torch
import trimesh

from . import gt as G
from .data import PKG
from .objects import knn_d
from .render import render, scene_parts

OBJ_NAMES = {"standard": {"peg": "peg_0", "box_with_hole": "box_with_hole_0"},
             "drawer": {"drawer": "drawer", "cabinet_frame": "cabinet_frame"}}


def gt_meshes(scene):
    sg = G.scene_graph(scene)
    out = {}
    for nd in sg["nodes"]:
        if nd.get("local_visual_mesh"):
            try:
                out[nd["name"]] = trimesh.load(PKG / scene / "evaluator_only" / nd["local_visual_mesh"], force="mesh")
            except Exception:
                pass
    return out, sg


def sample_mesh(m, n=4000):
    p, _ = trimesh.sample.sample_surface(m, n, seed=0)
    return np.asarray(p)


def chamfer_mm(A, B):
    return float((knn_d(A, B).mean() + knn_d(B, A).mean()) / 2 * 1000)


def gt_states(res):
    if res["scene"] == "real":
        from .data import load_real
        g = load_real().meta["gt_states"]
        return g[:len(res["robot"]["q"])]
    if res["source"] == "gt512" or res["scene"] == "drawer":
        return G.gt_states_keyframes(res["scene"])[:len(res["robot"]["q"])]
    return G.gt_states_steps(list(res["steps"]))


def evaluate(res, robot, obs_full=None):
    scene = res["scene"]
    gts = gt_states(res)
    T = len(res["robot"]["q"])
    out = {"frames": T}
    # ---- robot ----
    with torch.no_grad():
        qg = torch.tensor(np.stack([g["q"] for g in gts]), dtype=torch.float32, device="cuda")
        bg = torch.tensor(np.stack([g["robot_base"] for g in gts]), dtype=torch.float32, device="cuda")
        Pg, _ = robot.surface(qg, bg)
        qp = torch.tensor(res["robot"]["q"], dtype=torch.float32, device="cuda")
        bp = torch.tensor(res["robot"]["base"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4)
        Pp, _ = robot.surface(qp, bp)
        e = (Pg - Pp).norm(dim=-1)
        hand = robot.link_of >= robot.mesh_links.index("panda_hand")
        out["robot_surface_mm"] = (e.mean(1) * 1000).cpu().numpy().tolist()
        out["gripper_surface_mm"] = (e[:, hand].mean(1) * 1000).cpu().numpy().tolist()
    qgt = np.stack([g["q"] for g in gts])
    out["joint_err_deg"] = np.degrees(np.abs(res["robot"]["q"][:, :7] - qgt[:, :7])).mean(1).tolist()
    out["finger_err_mm"] = (np.abs(res["robot"]["q"][:, 7] - qgt[:, 7]) * 1000).tolist()
    out["base_err_mm"] = float(np.linalg.norm(res["robot"]["base"][:3, 3] - gts[0]["robot_base"][:3, 3]) * 1000)
    # ---- objects: match each GT manipulated/target object to the nearest predicted part ----
    if scene == "real":
        return out
    meshes, sg = gt_meshes(scene)
    names = OBJ_NAMES[scene]
    objs = {}
    wp = None
    if res["source"] == "gt512" or scene == "drawer":
        z = np.load(PKG / scene / "observations_gt.npz")
        segn = [str(x) for x in z["segment_names"]]
        wp = {n: [G.pose7_to_T(z["world_poses_wxyz"][t, i]) for t in range(T)] for i, n in enumerate(segn)}
    for key, segname in names.items():
        actor = [k for k in gts[0]["actors"] if k.startswith(key)] if key in ("peg", "box_with_hole") else []
        if wp is not None and segname in wp:
            Tg = wp[segname]
        elif actor:
            Tg = [g["actors"][actor[0]] for g in gts]
        else:
            continue
        m = meshes.get(segname)
        if m is None:
            continue
        Ploc = sample_mesh(m)
        Pw = [(Tg[t][:3, :3] @ Ploc.T).T + Tg[t][:3, 3] for t in range(T)]
        objs[key] = Pw
    pred_parts = []
    for ob in res["objects"]:
        from .objects import box_surface
        if ob["type"] == "articulated":
            from .objects import part_boxes
            from .render import box_mesh
            import trimesh as _tm

            def samp(T_, P_):
                return sample_mesh(_tm.util.concatenate([box_mesh(h, Tb) for Tb, h in part_boxes(T_, P_["half"], P_.get("slabs"))]), 4000)
            pred_parts.append((ob["name"] + "/child", [samp(ob["child"]["poses"][t], ob["child"]) for t in range(T)]))
            pred_parts.append((ob["name"] + "/parent", [samp(ob["parent"]["pose"], ob["parent"])] * T))
        elif ob["type"] == "static" and ob.get("hole"):
            from .objects import static_boxes
            from .render import box_mesh
            import trimesh as _tm
            mm = _tm.util.concatenate([box_mesh(h, Tb) for Tb, h in static_boxes(ob)])
            Pw = sample_mesh(mm, 4000)
            pred_parts.append((ob["name"], [Pw] * T))
        else:
            Pl, _ = box_surface(ob["half"], 3000)
            pred_parts.append((ob["name"], [(ob["poses"][t][:3, :3] @ Pl.T).T + ob["poses"][t][:3, 3] for t in range(T)]))
    for key, Pw in objs.items():
        if not pred_parts:
            out[f"{key}_chamfer_mm"] = None
            continue
        c0 = Pw[0].mean(0)
        best = min(pred_parts, key=lambda p: np.linalg.norm(p[1][0].mean(0) - c0))
        out[f"{key}_match"] = best[0]
        out[f"{key}_chamfer_mm"] = [chamfer_mm(Pw[t], best[1][t]) for t in range(T)]
        out[f"{key}_centre_mm"] = [float(np.linalg.norm(Pw[t].mean(0) - best[1][t].mean(0)) * 1000) for t in range(T)]
    # articulation (drawer): GT travel = drawer displacement from frame 0 along the GT joint axis (world -x)
    if scene == "drawer" and wp is not None:
        gtrav = np.array([-(wp["drawer"][t][0, 3] - wp["drawer"][0][0, 3]) for t in range(T)])
        out["gt_travel_m"] = gtrav.tolist()
        arts = [o for o in res["objects"] if o["type"] == "articulated"]
        frees = [o for o in res["objects"] if o["type"] == "free"]
        if arts:
            a = arts[0]
            ax = np.asarray(a["joint"]["axis"]) / np.linalg.norm(a["joint"]["axis"])
            out["joint_type"] = a["joint"]["type"]
            out["joint_axis_err_deg"] = float(np.degrees(np.arccos(np.clip(ax @ np.array([-1.0, 0, 0]), -1, 1))))
            out["travel_err_mm"] = (np.abs(np.asarray(a["joint"]["values"]) - gtrav) * 1000).tolist()
        elif frees:   # infer from the tracked motion of the best-matching free part
            c0 = wp["drawer"][0][:3, 3]
            f = min(frees, key=lambda o: np.linalg.norm(o["poses"][0][:3, 3] - c0))
            p = f["poses"][:, :3, 3] - f["poses"][0, :3, 3]
            u, s_, vt = np.linalg.svd(p, full_matrices=False)
            ax = vt[0] if p[-1] @ vt[0] > 0 else -vt[0]
            out["joint_type"] = "free(tracked)"
            out["joint_axis_err_deg"] = float(np.degrees(np.arccos(np.clip(ax @ np.array([-1.0, 0, 0]), -1, 1))))
            out["travel_err_mm"] = (np.abs(p @ ax - gtrav) * 1000).tolist()
    return out


def evaluate_drawer(res):
    """articulation-specific GT: drawer joint axis/travel from states."""
    gts = G.gt_states_keyframes("drawer")
    out = {}
    arts = gts[0]["articulations"]
    print(list(arts))
    return out


def render_agreement(res, robot, obs, gt_obs=None, views=None):
    """render the reconstruction into every camera of `obs` (incl. views not used for fitting):
    depth agreement with the observation (fraction of rendered pixels within 1 cm) and, when GT
    segmentation is available, IoU of the rendered dynamic parts with the GT dynamic-part masks."""
    T, V, H, W = obs.depth.shape if obs.depth is not None else obs.rgb.shape[:4]
    out = {"depth_agree_1cm": np.zeros((T, V)), "depth_agree_3cm": np.zeros((T, V)), "iou": np.full((T, V), np.nan)}
    dyn_ids = None
    if gt_obs is not None and gt_obs.seg is not None:
        names = gt_obs.meta["segment_names"]
        ids = gt_obs.meta["segment_ids"]
        keep = [i for i, n in zip(ids, names) if n.startswith("panda") or n.startswith("peg") or n.startswith("box_with")
                or n in ("drawer", "cabinet_frame", "drawer_red_block", "drawer_yellow_block")]
        dyn_ids = np.array(keep)
    for t in range(T):
        parts = scene_parts(robot, res, t)
        pr = [(m, c) for m, c, _ in parts]
        for v in range(V):
            w2c = np.linalg.inv(obs.c2w[v])
            img, alpha, dep, pid = render(pr, obs.K[v], w2c, H, W)
            if obs.depth is not None:
                D = obs.depth[t, v]
                m = alpha & (D > 0)
                out["depth_agree_1cm"][t, v] = float((np.abs(dep[m] - D[m]) < 0.01).mean()) if m.any() else np.nan
                out["depth_agree_3cm"][t, v] = float((np.abs(dep[m] - D[m]) < 0.03).mean()) if m.any() else np.nan
            if dyn_ids is not None:
                g = np.isin(gt_obs.seg[t, v], dyn_ids)
                out["iou"][t, v] = float((g & alpha).sum() / max(1, (g | alpha).sum()))
    return out


def box_sdf(P, T, half):
    """signed distance of world points P [n,3] to an oriented box (negative inside)"""
    q = np.abs((P - T[:3, 3]) @ T[:3, :3]) - half
    out = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return out + inside


def plausibility(res, robot):
    """GT-free physical checks of a reconstruction (usable on world-model rollouts):
    robot-object interpenetration, object-table penetration, finger-object gap while grasped."""
    import torch
    T = len(res["robot"]["q"])
    with torch.no_grad():
        qt = torch.as_tensor(res["robot"]["q"], dtype=torch.float32, device="cuda")
        bt = torch.as_tensor(res["robot"]["base"], dtype=torch.float32, device="cuda")[None].expand(T, 4, 4)
        P, _ = robot.surface(qt, bt)
    P = P.cpu().numpy()
    fing = (robot.link_of >= robot.mesh_links.index("panda_leftfinger")).cpu().numpy()
    n, d = res.get("plane", (np.array([0, 0, 1.0]), 0.0))
    out = {"robot_obj_penetration_mm": [], "obj_table_penetration_mm": [], "grasp_gap_mm": []}
    for t in range(T):
        pen, tab, gap = 0.0, 0.0, np.nan
        for ob in res["objects"]:
            if ob["type"] in ("attachment",):
                continue
            if ob["type"] == "articulated":
                from .objects import part_boxes
                boxes = part_boxes(ob["child"]["poses"][t], ob["child"]["half"], ob["child"].get("slabs"))
            else:
                boxes = [(ob["poses"][t], np.asarray(ob["half"]))]
            for Tb, h in boxes:
                s = box_sdf(P[t], Tb, h)
                pen = max(pen, float(-s.min()) * 1000 if s.min() < 0 else 0.0)
                corners = (np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * h) @ Tb[:3, :3].T + Tb[:3, 3]
                tab = max(tab, float(-(corners @ n + d).min()) * 1000)
                lifted = float((corners @ n + d).min()) > 0.005
                if ob["type"] == "free" and lifted:            # held in the air: fingers must touch it
                    sf = box_sdf(P[t][fing], Tb, h)
                    gap = float(np.abs(sf).min() * 1000)
        out["robot_obj_penetration_mm"].append(pen)
        out["obj_table_penetration_mm"].append(max(tab, 0.0))
        out["grasp_gap_mm"].append(gap)
    return out
