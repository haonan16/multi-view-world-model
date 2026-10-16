# Multiview video world model: views as tokens vs views tiled (RH20T)

Ablation of two ways to feed V calibrated camera views to one video diffusion model
(Wan2.1-T2V-1.3B, flow matching), trained on RH20T robot episodes (4 front-facing views,
13 frames at 3 fps, 160×320 per view):

- **tokens** — each view is VAE-encoded separately; all views' tokens share one sequence
  (same RoPE grid per view, so views are told apart only by their camera).
- **tiled** — the 4 views are downscaled into one 2×2 canvas of the same size (4× fewer tokens).

Both get each view's camera as a Plücker raymap concatenated to the latent channels
(zero-initialised), condition on frame 0 of every view, and predict frames 1–12.

| file | |
|---|---|
| `algorithms/wan/wan_multiview.py` | training / sampling / validation (`algorithm.design=tokens\|tiled`) |
| `algorithms/wan/modules/model_multiview.py` | multiview Wan backbone (raymap channels, per-view RoPE) |
| `datasets/rh20t.py`, `datasets/_multiview.py` | synced multiview RH20T loader, tiling helpers |
| `configurations/{algorithm/wan_multiview,dataset/rh20t_multiview}.yaml` | configs |
| `scripts/rh20t/` | data prep, training, inference, point clouds, viewers |

Paths in the configs and `.sbatch` files point at the Harvard FASRC cluster; change them for another machine.

### Data (once)
```bash
sbatch scripts/rh20t/lowres_download.sbatch           # official 320×180 RGB + calibration
sbatch scripts/rh20t/make_null_prompt_embed.sbatch    # UMT5 embedding of "" (no captions used)
sbatch scripts/rh20t/check_dataset_wandb.sbatch       # optional: sanity-check samples in wandb
```

### Train (4 GPUs, DDP)
```bash
DESIGN=tokens NAME=mv_tokens STEPS=30000 sbatch scripts/rh20t/train_multiview.sbatch algorithm.gradient_checkpointing_rate=1
DESIGN=tiled  NAME=mv_tiled  STEPS=30000 sbatch scripts/rh20t/train_multiview.sbatch
```
`RUN_DIR=<run dir>` resumes from its `checkpoints/last.ckpt`, `WANDB_RUN_ID=<id>` keeps logging
into the same wandb run, `CKPT_EVERY` sets the save interval, and `RACE=1` lets the same job be
queued in several partitions (first to start cancels the others). Tokens needs full gradient
checkpointing to fit in 80 GB. `scripts/rh20t/compare_runs.py <tokens_id> <tiled_id>` prints both
runs' validation curves.

### Inference and evaluation
```bash
OUT=<dir>
python scripts/rh20t/infer_multiview.py --out $OUT --n 15 \
    --tokens_ckpt <tokens.ckpt> --tiled_ckpt <tiled.ckpt>       # held-out scenes: GT / tokens / tiled + metrics.json
python scripts/rh20t/make_compare_videos.py $OUT <viewer>/vid   # GT-vs-prediction videos per scene
python scripts/rh20t/vggt_dynamic_clouds.py --root $OUT --out $OUT/clouds_native --native   # VGGT-Omega dynamic point clouds
python scripts/rh20t/render_cloud_videos.py --root $OUT --clouds clouds_native --out $OUT/videos   # point-cloud videos
python scripts/rh20t/bake_cloud_viewer.py --root $OUT --out <cloud_viewer>   # data for scripts/rh20t/viewer/index.html
```
`infer_multiview.sbatch` (with `VGGT=1`) runs the first and third steps on a GPU node. The
interactive viewer is a static page: copy `scripts/rh20t/viewer/index.html` plus three.js r128
(`three.min.js`, `OrbitControls.js`) next to the baked data and serve it with `python -m http.server`.
VGGT-Omega is loaded from a local clone (path set at the top of `vggt_dynamic_clouds.py`).

### Result (30k steps, held-out validation)
| | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| tokens | 22.0–22.8 | 0.72–0.73 | 0.080–0.090 |
| tiled | 20.6–21.2 | 0.56–0.57 | 0.246–0.257 |
| copy frame 0 | 25.9 | — | 0.055 |

Tokens wins at every checkpoint and at equal compute; neither model yet beats copying frame 0.

---

# Clean training code — Model F (5B) and the 1.3B R1 ablations

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.

The template's original project instructions (cluster setup, checkpoint download, etc.) are
preserved verbatim in **`README_TEMPLATE.md`**.

---

Branch `clean-training-code-aug-4`, cut **2026-08-04** from `origin/original-boyuan-code`
(`58d279d`) and reduced to **only the code needed to train**:

* **Model F** — 5B Wan2.2 MoT-concat geometry model, at its best checkpoint (cumulative
  step 2500, "eff-2500")
* **the 1.3B R1 ablations** — M_full / M_noNVS / M_noDepth / M_noPose

Nothing here is for evaluation, reporting, dataset conversion, or the other model variants.

### Where the file lists live

| file | what it is |
|---|---|
| **`metafiles/PYTHON_CLOSURE.txt`** | the **50 Python files** a training run actually loads (flat list) |
| `metafiles/APPLIED_FILES.txt` | all 100 files layered onto the base commit — Python + YAML + asset + scripts |
| `TRAINING_CODE_TRACE.md` | the derivation, the per-model breakdown, and the provenance audit |

`metafiles/PYTHON_CLOSURE.txt` is generated, not hand-maintained — regenerate or re-check after any edit:

```bash
python scripts/list_training_closure.py .            # the closure (should stay at 50)
python scripts/list_training_closure.py . --unused   # .py present but never imported
```

`--unused` should report exactly two: `algorithms/common/base_algo.py` (kept from the
template, unused by these runs) and the tracer itself.

## What's in it

| | count |
|---|---|
| Python files reachable from `main.py` | **50** (was 108 before pruning) |
| Hydra YAML | 43 |
| launch scripts | 6 |
| binary asset | 1 (`datasets/geometry_verification/aria/ase_undistort_maps_portrait.npz`) |

Verified after pruning: the package chain imports, and both configs compose (`exit 0`).

```
algorithms/
  common/            base_pytorch_algo.py            (+ base_algo.py, unused by these runs)
  wan/
    wan_t2v_ray_depth_mot_concat_5b.py               <- trained Model F
    wan_t2v_ray_depth_mot_concat_rebuttal.py         <- trained the 1.3B ablations
    modules/  model_ray_depth_mot_concat_5b.py  (5B DiT)
              model_ray_depth_mot_concat.py     (1.3B DiT)
              model.py attention.py t5.py tokenizers.py vae.py vae_v2.py
    utils/    fm_solvers.py fm_solvers_unipc.py
datasets/    mixture.py video_base.py _scale_norm.py _crop_utils.py
             _geometry_constants.py _caption_cache.py aria_new.py
             + 13 subset loaders (14 subsets: bedlam.py serves bedlam and bedlam2)
experiments/ exp_base.py exp_video.py
utils/       7 helpers
configurations/  algorithm/ cluster/ dataset/ experiment/
scripts/     6 sbatch launchers
```

### Pruning applied to five eager registries

These imported their whole package regardless of the run, which is why the boot set was
108 files. Each was trimmed to the two model classes / 14 subsets actually used:

* `algorithms/wan/__init__.py` — 17 wrappers → 2
* `experiments/exp_video.py` — the `algorithms.wan` + `datasets.*` import blocks and both
  `compatible_algorithms` / `compatible_datasets` registries
* `datasets/__init__.py` — emptied (it was only a convenience re-export)
* `datasets/mixture.py` — 22 loaders → 14
* `experiments/__init__.py` — dropped the `process_data` task, which pulled in the robot
  datasets, `algorithms/cogvideo/**`, and three `utils` modules

## Reproducing Model F (eff-2500)

`scripts/launch_5b_F_eff2500_16node.sbatch`

Rebuilt from the **authoritative record** — the resolved overrides of job `33283037`
(`…/vwm_train_runs/s5bF_Aresume14_64g_33283037/.hydra/overrides.yaml`) — *not* from the
launch script that was on disk in August, which had drifted to `batch_size=4` /
`gradient_checkpointing_rate=0.50` for a later resume.

| | eff-2500 (correct) | the drifted script |
|---|---|---|
| `batch_size` | **16** → global batch **1024** | 4 → global batch 256 |
| `gradient_checkpointing_rate` | **0.925** | 0.50 |

Also note `configurations/experiment/exp_video.yaml` defaults
`accumulate_grad_batches: 4`; the run overrides it to **1**. Omit that override and you
silently get global batch 4096.

**F is a warm-start chain, not a single run.** Each hop is weights-only, so each resets the
AdamW state and re-runs the 1000-step LR warmup — you must replay all three:

```
A-best  s5bA_perclip_32g_30557574/checkpoints/epoch=0-step=2000.ckpt
  -> F 32852495   cum step 1000
  -> F 33023613   cum step 2000
  -> F 33283037   cum step 2500   <- F-best
```

```bash
export HF_HOME=/path/to/hf-cache
export ASE_DATA_ROOT=/path/to/ase/scenes
TUNED_CKPT=/path/to/previous/hop/last.ckpt \
  sbatch scripts/launch_5b_F_eff2500_16node.sbatch
```

16 nodes × 4 H200. Scale normalisation is **per-clip** (`scale_preserving=false`);
`dataset=mixture_parallax14` supplies only the 14-subset roster — scheme-C parallax is *not*
active. Task split: NVS (`[2,3]`) for `re10k` + `dl3dv_colmap`; the 12 synthetic subsets are
depth+pose only (`[1]`).

## Reproducing the 1.3B R1 ablations

```
scripts/launch_13b_tta_re10k_F_1node.sbatch        M_full
scripts/launch_13b_ablation_noNVS_1node.sbatch     M_noNVS
scripts/launch_13b_ablation_noDepth_1node.sbatch   M_noDepth
scripts/launch_13b_ablation_noPose_1node.sbatch    M_noPose
scripts/launch_13b_tartanair_F_1node.sbatch        TartanAir-only 1.3B reference
```

1 node × 2 GPUs, bs4, `gradient_checkpointing_rate=1.0` (0.50 OOMs on 80 GB H100),
`gradient_clip_val` must stay **null** — stock Lightning FSDP rejects norm clipping and only
the 5B class carries the custom override. 240×320, raymap 30×40, 50 frames.

Each variant removes its modality as **both target and input** — zeroing the loss weight alone
is not enough, because the noised ground truth still reaches the other tasks through shared
attention:

| variant | tartanair | re10k | env flag |
|---|---|---|---|
| M_full | `[0,1,2,3,4]` | `[0,1,2,3]` | — |
| M_noNVS | `[1]` | `[1]` | `REBUTTAL_RGB_CONTEXT_ONLY=1` |
| M_noDepth | `[0,1,2,3]` | `[0,1,2,3]` | `REBUTTAL_DROP_DEPTH=1` |
| M_noPose | `[0,1,4]` | `[0]` | `REBUTTAL_DROP_RAYMAP=1` |

### Fixed here: the `weight_type` footgun

`configurations/dataset/mixture_tartanair_re10k.yaml` shipped `weight_type: relative` in the
working repo. `relative` multiplies each weight by dataset size — TartanAir 362 records vs
RE10K 65,642 → **~0.5% / 99.5%**, which starves depth (only TartanAir has it) and makes the
ablations uninterpretable. The training runs passed `absolute` on the CLI, so they were fine;
**the YAML default is corrected to `absolute` on this branch.**

### Provenance caveat

The evaluated ablation checkpoints were trained from a *different* working copy —
`/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model_optim_efficiency_latest`,
under `kempner_rcai_lab` (jobs 35013525 / 35014222 / 35014234). The files here are byte-identical
to that checkout **except** `wan_t2v_ray_depth_mot_concat_rebuttal.py`, which carries ~90 lines
added afterwards for checkpoint loading and subclass plumbing (`classes_to_shard` became a
classmethod using `cls.attention_block_cls`, which is `WanAttentionBlock` here, so FSDP wraps
identically). Noise scheduling, `DEFAULT_P`, `subset_settings`, `_masked_mean`, the loss terms
and the `REBUTTAL_*` gates are unchanged. That is a code-reading conclusion — nothing has been
re-trained to confirm it empirically.

## Environment

`requirements.txt` is the template's original list; `requirements_cur.txt` is a freeze of the
`test2` conda env these models actually trained under (it contains absolute paths — treat it as
a record, not something to `pip install -r` verbatim).

## Verifying a change didn't break composition

```bash
export HF_HOME=/path/to/hf-cache ASE_DATA_ROOT=/path/to/ase/scenes
python -m main --cfg job --resolve experiment=exp_video \
  algorithm=wan_t2v_ray_depth_mot_concat_5b dataset=mixture_parallax14 \
  dataset.scale_preserving=false cluster=seas_h200 +name=probe
python -m main --cfg job --resolve experiment=exp_video \
  algorithm=wan_t2v_ray_depth_mot_concat_rebuttal dataset=mixture_tartanair_re10k \
  cluster=fas_high +name=probe
```

Both must exit 0. Data roots live in `configurations/dataset/*.yaml` and point at FASRC
netscratch paths — repoint them for your site.
