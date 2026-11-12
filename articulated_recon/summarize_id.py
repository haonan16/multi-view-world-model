"""results/id/SUMMARY.md: inverse dynamics from 3D executed in ManiSkill (StackCube-v1)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results/id"


def row(d):
    f = R / d / "results.json"
    if not f.exists():
        return None
    x = json.load(open(f))["runs"]
    n = len(x)
    return n, sum(r["ever"] for r in x), sum(r["success"] for r in x)


L = ["# Inverse dynamics from our 3D reconstruction, executed in ManiSkill (StackCube-v1)", "",
     "Method: Li et al. 2026 (arXiv 2607.01166). Subgoal 3D scenes -> actions, two variants:",
     "- **learned**: their released inverse-dynamics model (diffusion; current + goal point cloud from 4 fixed cameras, "
     "qpos -> 64 joint actions, 32 executed, closed loop). Goal point cloud = our reconstruction placed in a twin scene.",
     "- **retarget**: their learning-free variant: end-effector SE(3) per subgoal -> motion planner (mplib) + gripper open/close.",
     "", "Scene reset with the demo episode's seed = the episode's frame 0 (the world model's input). Subgoals = the 6 "
     "reconstructed frames (steps 0..100, every 20); the demos release at step 105-130, so the plan completes the last "
     "held frame by lowering the red cube onto the green cube and opening.", "",
     "Success = ManiSkill StackCube success; as in their eval.py the episode ends at success "
     "(stacked-at-end in brackets where different).", "",
     "| goal source | plan | learned | retarget |", "|---|---|---|---|"]

def cell(d):
    r = row(d)
    if r is None:
        return "-"
    n, ever, end = r
    return f"{ever}/{n}" + (f" ({end} at end)" if end != ever else "")

L.append(f"| oracle (demo sim states) | raw frames | {cell('learned_oracle')} | {cell('retarget_oracle')} |")
L.append(f"| ours, GT frames (sim RGB + depth) | raw frames | {cell('learned_recon_gt_raw_idgt')} | {cell('retarget_recon_gt_raw_idgt')} |")
L.append(f"| ours, world-model predictions | raw frames | {cell('learned_recon_da3_final')} | {cell('retarget_recon_da3_final')} |")
L.append(f"| oracle (demo sim states) | object-centric | {cell('v3_learned_oracle')} | {cell('v3_retarget_oracle')} |")
L.append(f"| ours, GT frames | object-centric | {cell('v3_learned_reconGT')} | {cell('v2_retarget_reconGT')} |")
L.append(f"| ours, world-model predictions (+ anchored) | object-centric | {cell('v3_learned_reconWM')} | {cell('v2_retarget_reconWM')} |")
L += ["", "Object-centric plan (same rules for every goal source): contact snap (unheld cubes rest on the table / the "
      "other cube), unheld cubes do not move, grasp prior (held cube between the fingertips, top-down, face pair "
      "with least wrist rotation), no end-effector waypoints from the robot fit before the grasp. World-model goals "
      "are anchored to the current observation: per-cube translation so frame 0 matches our reconstruction of the "
      "sim's frame 0 (4 cameras, the same input their inverse dynamics gets).", "",
      "Why: the world-model reconstruction is off by median 30 mm (red), 50 mm (green), 48 mm (robot) at frame 0 "
      "(generated views misregistered with their cameras); its held frames put the gripper 43-87 mm from the cube.",
      "From the prediction the plan keeps: which cube, when it is grasped, the carry path, where it is placed."]
(R / "SUMMARY.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
