# Offline replay of frozen execution

`render_registered_execution.py` makes a 20 fps video from a **completed** native
execution. It does not run ManiSkill. Every video frame uses one saved robot
joint vector and one saved object pose; no reconstructed or target trajectory
is substituted. The independent reference's table tilt and offset are retained.

From the uploaded `articulated_recon` code snapshot, using the existing CPU
runtime after execution finishes:

```bash
python benchmarks/render_registered_execution.py \
  --submission /path/run/execution-preparation/variant/submission.json \
  --evaluation-receipt /path/run/execution-preparation/variant/evaluation/evaluation-attachment.json \
  --execution /path/run/execution/variant \
  --output /path/run/media/variant-execution.mp4
```

The default URDF is the existing cluster Panda asset; `--urdf` can supply its
verified copy. `--view contact` selects a closer fixed camera and explicitly
labels the translucent robot. The default overview keeps it opaque. Neither
view changes recorded state or retimes playback.

Before rendering, the adapter verifies the submission, evaluator attachment,
exclusive execution claim, complete log/action hashes, robot URDF and 20 Hz
timestamps. Offline forward kinematics must agree with the logged TCP within
2 mm. It rechecks the records and assets after rendering. The output includes
first, maximum-height and final stills plus a provenance JSON. These are
collision-mesh visualizations of recorded physics in a reconstructed reference
proxy, not photorealistic simulator frames or exact ground-truth geometry.

The packaged CPU rasterizer may compile its small C++ helper on first use;
the cluster runtime needs `clang++`, `imageio-ffmpeg`, OpenCV and trimesh. No
model weights, licensed MANO data or GPU are required.
