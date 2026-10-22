"""Patch Lyra 2.0 so run_lyra2_sample() accepts seed views separately from the trajectory.

Applied to lyra_2/_src/inference/lyra2_ar_inference.py (nv-tlabs/lyra, Lyra-2/). Run from the Lyra-2 directory:
    python /path/to/patch_lyra_multiview_seeds.py
"""
p = "lyra_2/_src/inference/lyra2_ar_inference.py"; s = open(p).read()
old = """    multiview_data = None
    if getattr(args, "multiview_ids", None):
        multiview_data = {"""
new = """    multiview_data = None
    if getattr(args, "multiview_ids", None) and "mv_video" in data_batch:
        # seed views supplied separately from the trajectory (light-stage driver)
        multiview_data = {
            "video": data_batch["mv_video"],
            "depth": data_batch["mv_depth"],
            "camera_w2c": data_batch["mv_w2c"],
            "intrinsics": data_batch["mv_K"],
        }
    elif getattr(args, "multiview_ids", None):
        multiview_data = {"""
if "\"mv_video\" in data_batch" in s: print("already patched")
else:
    assert old in s, "unexpected Lyra version"; open(p, "w").write(s.replace(old, new)); print("patched", p)
