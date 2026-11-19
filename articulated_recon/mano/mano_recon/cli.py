"""Fit a prepared calibrated clip from an explicit initialization."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .fitting import SequenceFitter
from .model import MANOHand, translation_base
from .observations import Observations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("observations", type=Path)
    p.add_argument("initialization", type=Path, help="NPZ q[T,3+P], translation[T,3] metres, optional betas[1,10]")
    p.add_argument("--side", choices=["left", "right"], required=True)
    p.add_argument("--models", default="assets/mano")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--holdout", type=int, default=None)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    obs = Observations.load(args.observations)
    if args.holdout is not None:
        if not 0 <= args.holdout < obs.depth.shape[1] or obs.depth.shape[1] < 2:
            p.error("holdout must identify an existing camera; at least one camera must remain")
        obs = obs.select_views(v for v in range(obs.depth.shape[1]) if v != args.holdout)
    with np.load(args.initialization, allow_pickle=False) as z:
        q, translation = z["q"], z["translation"]
        betas = z["betas"] if "betas" in z else None
    hand = MANOHand(args.models, args.side, components=q.shape[1]-3, device=args.device)
    result = SequenceFitter(hand, obs).fit(q, translation, betas, iterations=args.iterations)
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        vertices, joints = hand(result["q"], result["betas"], translation_base(result["translation"]))
    np.savez_compressed(args.out / "fit.npz", **{k: v.cpu().numpy() for k, v in result.items() if k != "history"},
                        vertices=vertices.cpu().numpy(), joints21=joints.cpu().numpy(), faces=hand.faces.cpu().numpy(), timestamps=obs.timestamps)
    (args.out / "fit.json").write_text(json.dumps({
        "observations": str(args.observations.resolve()), "side": args.side,
        "shared_shape": True, "units": "metres", "heldout_view": args.holdout,
        "history": result["history"], "stage": "hand only; no object/contact or physics fitting",
    }, indent=2))


if __name__ == "__main__":
    main()
