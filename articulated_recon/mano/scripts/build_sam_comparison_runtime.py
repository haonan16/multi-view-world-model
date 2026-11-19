"""Build an isolated, auditable Amani fitter for the SAM ablation.

Existing baseline code and results are never modified. The legacy human grasp
adapter is kept for the matched comparisons. An additional upstream-grasp mode
uses Amani's original state machine with MANO thumb/index aperture and frame.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def guard_missing_centres(s):
    """Tracked 2-D masks may have no usable depth centroid; not release evidence."""
    helper='''\n\ndef stable_3d_centres(chosen, centre, minimum=2):
    centres = [x["c"] for x in chosen.values() if x["c"] is not None]
    return len(centres) >= minimum and all(np.linalg.norm(c - centre) < 0.03 for c in centres)
'''
    s=s.replace('all(np.linalg.norm(x["c"] - cen[tp]) < 0.03 for x in chosen.values())',
                'stable_3d_centres(chosen, cen[tp])')
    s=s.replace('if np.max([np.linalg.norm(a - c2) for a in cs]) < 0.03:',
                'if stable_3d_centres(ch2, c2):')
    if 'def stable_3d_centres(' not in s:s+=helper
    return s


def guard_unknown_human_width(s):
    # Invisible hands use a sentinel width for safety. It is not an observed
    # open-hand aperture and must not calibrate the grasp/release threshold.
    return s.replace('w_open = float(np.max(w))',
        'hand_visible = np.asarray(self.rob.get("visible", np.ones(len(w), bool)))\n'
        '            w_open = float(np.max(w[hand_visible])) if hand_visible.any() else 0.0')


def replace(s, old, new):
    if old not in s:
        raise RuntimeError('Upstream changed; review patch: ' + old)
    return s.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    dest = args.run/'runtime'/'amani_comparison'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'__init__.py').write_text('')
    patches=[]; hashes={}
    for name in ['geom.py','objects_v2.py','knn.py']:
        path=ROOT/'mano_recon/amani_objects'/name
        old=path.read_text(); s=old
        if name in ['geom.py','objects_v2.py']:
            s=replace(s,'import torch','import torch\nimport os\nDEVICE = os.environ.get("AMANI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")')
            s=s.replace('"cpu"','DEVICE')
            # Restore the literal fallback in the DEVICE declaration itself.
            s=s.replace('else DEVICE)','else "cpu")')
        if name=='objects_v2.py':
            s=replace(s,'from ..model import translation_base\n','')
            for a,b in [('mesh_samples(mesh, 500)','mesh_samples(mesh, 1500)'),('iters=60','iters=150'),('iters=100','iters=250'),('[:1000]','[:3000]'),('[:150]','[:300]')]:
                s=replace(s,a,b)
            # Preserve both grasp interpretations to avoid confounding the
            # mask/asset comparison with a new human grasp detector.
            upstream=(args.run/'upstream/artirecon/objects_v2.py').read_text()
            st=upstream.index('        w = self.width\n',upstream.index('def track('))
            en=upstream.index('        return sel, np.array(cen), held',st)
            original_grasp=upstream[st:en].replace('spec.get("finger_reach", 0.15)','spec.get("finger_reach", 0.0)')
            st=s.index('        static = spec.get("role") == "static"',s.index('def track('))
            en=s.index('        return sel, np.array(cen), held',st)
            legacy=s[st:en]
            indent=lambda block: ''.join('    '+line if line.strip() else line for line in block.splitlines(True))
            s=s[:st]+'        if os.environ.get("AMANI_GRASP", "legacy") == "upstream":\n'+indent(original_grasp)+'        else:\n'+indent(legacy)+s[en:]
            s=replace(s,'            best = None\n            for sv in variants:', '            if os.environ.get("AMANI_FORCE_MESH") == "1" and spec.get("shape") == "mesh":\n                variants = [spec]\n            candidate_losses = []\n            best = None\n            for sv in variants:')
            s=replace(s,'                if best is None or lp < best[0]:','                candidate_losses.append(dict(shape=sv.get("shape", "box"), loss=float(lp)))\n                if best is None or lp < best[0]:')
            s=replace(s,'out.append(dict(name=spec["name"], type=', 'out.append(dict(selected_shape=kind_used, candidate_losses=candidate_losses, name=spec["name"], type=')
            s=guard_missing_centres(s)
            s=guard_unknown_human_width(s)
        (dest/name).write_text(s)
        compile(s,str(dest/name),'exec')
        hashes[name]=dict(baseline_sha256=hashlib.sha256(old.encode()).hexdigest(),runtime_sha256=hashlib.sha256(s.encode()).hexdigest())
        patches.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='baseline/'+name,tofile='comparison/'+name))
    (dest/'changes.diff').write_text(''.join(patches))
    (dest/'provenance.json').write_text(json.dumps(hashes,indent=2))
    print(dest)


if __name__=='__main__':
    main()
