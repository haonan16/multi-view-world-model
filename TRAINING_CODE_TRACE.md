# Training-code trace — Model F (best, eff-2500) and the 1.3B R1 ablations

Purpose: enumerate every repo file involved in producing these two model families, so a clean
branch can be cut that contains **training code only**.

**How this was derived** (not from memory — regenerate any time):
1. **Python closure** — AST walk of `import` / `from ... import` starting at `main.py`,
   resolving only in-repo modules, and including every parent-package `__init__.py`.
   Result: **108 Python files** load at startup.
2. **Config closure** — recursive resolution of Hydra `defaults:` lists from
   `configurations/config.yaml` plus the CLI group overrides each run used.
3. **Provenance** — each run's own `.hydra/overrides.yaml` on netscratch, which records what
   was *actually* submitted, rather than the current state of the launch script.

**Read this first — the import graph over-reports.** Four files are eager registries that
import their whole package regardless of what the run uses:

| registry | pulls in |
|---|---|
| `algorithms/wan/__init__.py` | all 17 `wan_*.py` wrappers + all 13 `modules/model_*.py` |
| `experiments/exp_video.py` | all algorithm classes + ~30 dataset classes |
| `datasets/__init__.py` | 8 loaders |
| `datasets/mixture.py` | 21 loaders |

So the 108-file boot set is what Python *loads*; the per-run tables below are what each run
*exercises*. Both are listed, because a clean branch must either keep the boot set or prune
those four files (see Part 3).

---

# PART 1 — Model F, best checkpoint (eff-2500)

## Provenance

| | |
|---|---|
| checkpoint | `/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/s5bF_eff2500_frozen.ckpt` (29.6 GB) |
| origin | job **33283037**, `s5bF_Aresume14_64g_33283037/checkpoints/epoch=0-step=500.ckpt` |
| cumulative F step | 2500 (`eval_pi3/hf/F_clips.tsv`: `33283037  s5bF_2500  2500  2560000  gb1024`) |
| warm-start chain | A-best `s5bA_perclip_32g_30557574/…step=2000.ckpt` → F `32852495` → `33023613` → **`33283037`** |
| launch script | `scripts/launch_5b_F_Aresume_14ds_dpose_16node.sbatch` |
| hardware | 16 nodes × 4 H200 = 64 GPUs, `kempner_h200_priority`, qos `kemp_gpu64_id36` |

### ⚠ The launch script on disk has drifted from what produced eff-2500

The current script is a **later revision** (used for the `33987485` resume). Differences that
matter, from `33283037/.hydra/overrides.yaml`:

| | eff-2500 (job 33283037) | script on disk today |
|---|---|---|
| `experiment.training.batch_size` | **16** → global batch **1024** | 4 → global batch 256 |
| `algorithm.gradient_checkpointing_rate` | **0.925** | 0.50 |
| `algorithm.model.tuned_ckpt_path` | `…33023613/checkpoints/last.ckpt` | `…33987485/checkpoints/last.ckpt` |

Everything else is identical. **To reproduce eff-2500, use the overrides file, not the script.**

Full recorded override list: `…/vwm_train_runs/s5bF_Aresume14_64g_33283037/.hydra/overrides.yaml`
(43 entries; key ones: `algorithm=wan_t2v_ray_depth_mot_concat_5b`, `dataset=mixture_parallax14`,
`dataset.scale_preserving=false`, `288×384×50`, `lr=4e-5`, `gradient_clip_val=1.0`,
`diffusion_forcing.mode=ray_depth_prediction_firstlast`, `train_rgb=true`,
`subset_settings.{tartanair,spring,bedlam,bedlam2,vkitti2,mid_air,mvs_synth}=[1]`,
`num_nodes=16`, `+fsdp_shard_within_node=true`, `cluster=seas_h200`).

Note `dataset=mixture_parallax14` supplies only the 14-subset **roster** —
`scale_preserving=false` means the per-clip (A/F) encoding is active and parallax/scheme-C is **not**.

## Entry point and framework (shared with Part 2)

```
main.py
configurations/config.yaml
experiments/__init__.py
experiments/exp_base.py
experiments/exp_video.py                  # registry — see Part 3
utils/__init__.py                         # empty package marker
utils/print_utils.py
utils/distributed_utils.py
utils/ckpt_utils.py
utils/cluster_utils.py
utils/wandb_utils.py
utils/video_utils.py
algorithms/__init__.py                    # empty package marker
algorithms/common/__init__.py             # empty package marker
algorithms/common/base_pytorch_algo.py
```
`algorithms/common/base_algo.py` is **not** imported by either run (only the yaml
`configurations/algorithm/base_algo.yaml` participates, via the defaults chain).

## Algorithm — the 5B MoT-concat model

```
algorithms/wan/__init__.py                            # registry — see Part 3
algorithms/wan/wan_t2v_ray_depth_mot_concat_5b.py     # ← the class that trained F
algorithms/wan/modules/__init__.py
algorithms/wan/modules/model_ray_depth_mot_concat_5b.py   # ← the DiT definition
algorithms/wan/modules/model.py                       # base WanModel / WanAttentionBlock
algorithms/wan/modules/attention.py
algorithms/wan/modules/t5.py                          # umt5 text encoder
algorithms/wan/modules/tokenizers.py
algorithms/wan/modules/vae.py
algorithms/wan/modules/vae_v2.py                      # Wan2.2 48-ch VAE (the 5B one)
algorithms/wan/utils/__init__.py
algorithms/wan/utils/fm_solvers.py
algorithms/wan/utils/fm_solvers_unipc.py
```

## Datasets — the 14 subsets actually sampled

Shared dataset infrastructure:
```
datasets/__init__.py                # registry — see Part 3
datasets/mixture.py                 # the weighted sampler; registry — see Part 3
datasets/video_base.py              # temporal cropping, trim_mode=random_cut
datasets/_scale_norm.py             # per-clip / parallax / global-metric encodings
datasets/_crop_utils.py
datasets/_geometry_constants.py
datasets/_caption_cache.py          # imported by video_base.py
datasets/aria_new.py                # provides camera_to_raymap()
```

Per-subset loaders (13 modules serve 14 subsets — `bedlam.py` serves both `bedlam` and `bedlam2`):
```
datasets/tartanair.py         # also the BASE CLASS for spring/bedlam/matrixcity/mvs_synth/mid_air
datasets/tartanair_v2.py
datasets/spring.py            # subclasses tartanair
datasets/bedlam.py            # subclasses tartanair; serves bedlam + bedlam2
datasets/vkitti2.py
datasets/ase.py
datasets/omniworld_game.py
datasets/re10k.py
datasets/dl3dv_colmap.py
datasets/scenenet.py
datasets/matrixcity.py        # subclasses tartanair
datasets/mvs_synth.py         # subclasses tartanair
datasets/mid_air.py           # subclasses tartanair
```

## Non-Python asset — do not lose this

```
datasets/geometry_verification/aria/ase_undistort_maps_portrait.npz   # 2.5 MB
```
Hardcoded at `datasets/ase.py:59` as `Path(__file__).parent/"geometry_verification"/"aria"/…`.
ASE fisheye→pinhole undistortion fails without it, and ASE is ~8% of F's training mix.

Also referenced (as an **append-mode output**, not an input) by `re10k.py:639` and
`aria_new.py:532`: `ignore/failed_videos.txt` — the `ignore/` directory must exist.

## Config closure — 40 YAML files

```
configurations/config.yaml
configurations/experiment/base_experiment.yaml
configurations/experiment/base_pytorch.yaml
configurations/experiment/exp_video.yaml
configurations/algorithm/base_algo.yaml
configurations/algorithm/base_pytorch_algo.yaml
configurations/algorithm/wan_i2v.yaml
configurations/algorithm/wan_t2v.yaml
configurations/algorithm/wan_t2v_ray_depth_mot.yaml
configurations/algorithm/wan_t2v_ray_depth_mot_5b.yaml
configurations/algorithm/wan_t2v_ray_depth_mot_concat_5b.yaml
configurations/cluster/base_slurm.yaml
configurations/cluster/seas_h200.yaml
configurations/dataset/video_base.yaml
configurations/dataset/mixture.yaml
configurations/dataset/mixture_all.yaml
configurations/dataset/mixture_parallax14.yaml
configurations/dataset/tartanair.yaml
configurations/dataset/tartanair_v2.yaml
configurations/dataset/spring.yaml
configurations/dataset/bedlam.yaml
configurations/dataset/bedlam2.yaml
configurations/dataset/vkitti2.yaml
configurations/dataset/ase.yaml
configurations/dataset/omniworld_game.yaml
configurations/dataset/re10k.yaml
configurations/dataset/dl3dv_colmap.yaml
configurations/dataset/scenenet.yaml
configurations/dataset/matrixcity.yaml
configurations/dataset/mvs_synth.yaml
configurations/dataset/mid_air.yaml
configurations/dataset/agibot_world.yaml      # pulled by mixture.yaml defaults, not sampled
configurations/dataset/bridge.yaml            #   "
configurations/dataset/droid.yaml             #   "
configurations/dataset/ego4d.yaml             #   "
configurations/dataset/epic_kitchen.yaml      #   "
configurations/dataset/language_table.yaml    #   "
configurations/dataset/openx_base.yaml        #   "
configurations/dataset/pandas.yaml            #   "
configurations/dataset/something_something.yaml  #   "
```
The last nine come from `configurations/dataset/mixture.yaml`'s own defaults list. They are
required for Hydra composition to succeed even though no such subset is sampled.

---

# PART 2 — 1.3B R1 ablations (M_full / M_noNVS / M_noDepth / M_noPose)

## ⚠ Provenance: these were **not** trained from this checkout

The evaluated checkpoints came from a **different working copy**, on holylfs05, run under the
`kempner_rcai_lab` account (collaborator's runs), recorded in each run's `hydra.runtime.cwd`:

```
/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model_optim_efficiency_latest
```

Checkpoints preserved at:
```
/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/video_world_model_optim_efficiency_latest/
    preserved_checkpoints/r1_6way_with_tartanair_and_single_transformer_35097889/milestones/
        full_tartanair_only/step=7500.ckpt      full_tta_re10k/step=7500.ckpt
        noNVS/step=7500.ckpt   noDepth/step=7500.ckpt   noPose/step=7500.ckpt
```

Recorded training runs (`.hydra/overrides.yaml` present for three of the five):

| variant | job | run dir |
|---|---|---|
| noNVS | 35013525 | `runs/s13B_ablNoNVS_2g_35013525` |
| noDepth | 35014222 | `runs/s13B_ablNoDepth_2g_35014222` |
| noPose | 35014234 | `runs/s13B_ablNoPose_2g_35014234` |
| full_tta_re10k, full_tartanair_only | — | no preserved run dir; milestones only |

Common config (identical across the three, from their overrides files):
`algorithm=wan_t2v_ray_depth_mot_concat_rebuttal`, `dataset=mixture_tartanair_re10k`,
**`dataset.training.weight_type=absolute`**, `dataset.scale_preserving=false`,
`240×320`, `raymap 30×40`, `n_frames=50`, `batch_size=4`, `num_nodes=1`, 2 GPUs → **global batch 8**,
`gradient_checkpointing_rate=1.0`, `accumulate_grad_batches=1`, `cluster=fas_high`,
`account=kempner_rcai_lab`, env `psamp`, `wandb.project=ei_world_model`.

Per-variant differences — `subset_settings` plus an **environment variable** that does not
appear in the Hydra overrides:

| variant | tartanair | re10k | env flag |
|---|---|---|---|
| M_full | `[0,1,2,3,4]` | `[0,1,2,3]` | — |
| M_noNVS | `[1]` | `[1]` | `REBUTTAL_RGB_CONTEXT_ONLY=1` |
| M_noDepth | `[0,1,2,3]` | `[0,1,2,3]` | `REBUTTAL_DROP_DEPTH=1` |
| M_noPose | `[0,1,4]` | `[0]` | `REBUTTAL_DROP_RAYMAP=1` |

## ⚠ TRAP: `weight_type` in this repo's YAML is wrong for reproduction

| | |
|---|---|
| rcai checkout (what trained the ablations) | `training: {weight_type: absolute, …}` |
| **this repo** `configurations/dataset/mixture_tartanair_re10k.yaml:22` | `training: {weight_type: relative, …}` |

`relative` multiplies each weight by dataset size → TartanAir 362 records vs RE10K 65,642 gives
**~0.5% / ~99.5%**, which starves depth (only TartanAir has it) and makes the ablations
uninterpretable. `absolute` gives the intended 50/50.

The four launch scripts in this repo **do** pass `dataset.training.weight_type=absolute` on the
CLI (verified in all four), so they are safe as-is — but the YAML default is a live footgun for
anyone composing this dataset without the override. Fix the YAML on the clean branch.

## ⚠ Code drift since training

`algorithms/wan/wan_t2v_ray_depth_mot_concat_rebuttal.py` in this repo is **2.3 KB larger**
(90 diff lines) than the version that trained the ablations. What changed:

- `classes_to_shard()`: `staticmethod → classmethod`, `{WanAttentionBlock,…}` →
  `{cls.attention_block_cls,…}`. **Behaviourally identical** here — the class sets
  `attention_block_cls = WanAttentionBlock` (line 250), so FSDP wraps the same modules.
- added `_is_native_model_checkpoint`, `_is_incompatible_model_checkpoint`,
  `_validate_native_checkpoint_load`, and a `configure_model` checkpoint-loading refactor.

I diffed for changes to the training math specifically — noise scheduling (`t[...]`),
`DEFAULT_P`, `subset_settings` handling, `_masked_mean`, the loss terms, and the `REBUTTAL_*`
gates are **unchanged**. The drift is confined to checkpoint loading and subclass plumbing
(added to support the single-transformer eval). Training behaviour should be identical, but
this is a code-reading conclusion, not an empirical one — nothing has been re-trained to confirm it.

`experiments/exp_video.py` also differs: this repo adds the
`wan_t2v_ray_depth_single_transformer` registry entry (eval-only addition).

## Launch scripts in this repo

These were authored here and match the recorded overrides above; they were prepared for the
collaborator's submission rather than submitted from this checkout:
```
scripts/launch_13b_tta_re10k_F_1node.sbatch          # M_full
scripts/launch_13b_ablation_noNVS_1node.sbatch       # M_noNVS
scripts/launch_13b_ablation_noDepth_1node.sbatch     # M_noDepth
scripts/launch_13b_ablation_noPose_1node.sbatch      # M_noPose
scripts/launch_13b_tartanair_F_1node.sbatch          # TartanAir-only 1.3B (s13B_tta_F, job 34980104)
```

## Algorithm — the 1.3B MoT-concat rebuttal class

Everything in Part 1's *framework* and *shared wan modules* lists applies unchanged. Only these
two files differ from F:
```
algorithms/wan/wan_t2v_ray_depth_mot_concat_rebuttal.py   # ← the class that trained the ablations
algorithms/wan/modules/model_ray_depth_mot_concat.py      # ← the 1.3B DiT definition
```
(`wan_t2v_ray_depth_mot_concat_rebuttal.py` is a deliberate duplicate of
`wan_t2v_ray_depth_mot_concat.py` so the `REBUTTAL_*` gates could never affect production runs.)

## Datasets — only two subsets

```
datasets/mixture.py
datasets/video_base.py
datasets/_scale_norm.py
datasets/_crop_utils.py
datasets/_geometry_constants.py
datasets/_caption_cache.py
datasets/aria_new.py
datasets/tartanair.py
datasets/re10k.py
```
No ASE → the `ase_undistort_maps_portrait.npz` asset is **not** needed for this family.

## Config closure — 26 YAML files

```
configurations/config.yaml
configurations/experiment/{base_experiment,base_pytorch,exp_video}.yaml
configurations/algorithm/{base_algo,base_pytorch_algo,wan_i2v,wan_t2v,wan_t2v_ray_depth_mot}.yaml
configurations/algorithm/wan_t2v_ray_depth_mot_concat_rebuttal.yaml
configurations/cluster/{base_slurm,fas_high}.yaml
configurations/dataset/{video_base,mixture,mixture_tartanair_re10k,tartanair,re10k}.yaml
configurations/dataset/{agibot_world,bridge,droid,ego4d,epic_kitchen,language_table,
                        openx_base,pandas,something_something}.yaml   # mixture.yaml defaults
```

---

# PART 3 — Cutting the clean branch

## Union keep-list

Everything in Parts 1 and 2, i.e.:

- **1** entry point + **4** experiment files + **7** utils + **3** algorithms-common
- **13** wan files (2 wrappers + 2 model defs + 9 shared modules/utils) + 3 `__init__.py`
- **8** dataset infra + **13** subset loaders (F) of which 2 also serve the 1.3B runs
- **1** binary asset (`ase_undistort_maps_portrait.npz`)
- **43** unique YAML files (40 for F ∪ 26 for 1.3B, 23 shared)
- **6** launch scripts

Exact counts: **50 Python files** (4 framework + 7 utils + 3 algorithms-common + 15 wan
+ 8 dataset infra + 13 subset loaders), **43 YAML**, **1** binary asset, **6** launch scripts.
All 57 spot-checked paths verified present on 2026-08-04.

## The pruning that actually shrinks it

The union above is ~90 files, but **108 Python files load at startup** because of the four
registries. To drop the difference you must edit them — deleting the files alone breaks the import:

1. `algorithms/wan/__init__.py` — trim to `WanTextToVideoRayDepthMoTConcat5B` and
   `WanTextToVideoRayDepthMoTConcatRebuttal`.
2. `experiments/exp_video.py` — trim the `from algorithms.wan import (...)` block, the
   `from datasets.* import ...` block, and the matching `compatible_algorithms` /
   dataset-registry entries.
3. `datasets/__init__.py` — currently imports 8 loaders eagerly; empty it or trim.
4. `datasets/mixture.py` — imports 21 loaders; trim to the 13 F subsets (+ the 2 for 1.3B,
   which are a subset of those 13).

Then these become deletable — they are reachable *only* through those registries:

```
algorithms/cogvideo/**                       (7 files, incl. its own text_encoder package)
algorithms/wan/wan_i2v.py                    algorithms/wan/wan_t2v.py
algorithms/wan/wan_t2v_v2_14b.py             algorithms/wan/wan_t2v_concat_rgb_depth.py
algorithms/wan/wan_t2v_depth*.py             (5 files)
algorithms/wan/wan_t2v_ray_mot*.py           (2 files)
algorithms/wan/wan_t2v_ray_depth_mot.py      algorithms/wan/wan_t2v_ray_depth_mot_concat.py
algorithms/wan/wan_t2v_ray_depth_mot_split_ray_patch.py
algorithms/wan/wan_t2v_ray_depth_single_transformer.py     # keep if you want the ST ablation
algorithms/wan/modules/model_depth*.py       (5)   algorithms/wan/modules/model_ray_mot*.py (2)
algorithms/wan/modules/model_ray_depth_mot.py
algorithms/wan/modules/model_ray_depth_mot_split_ray_patch.py
algorithms/wan/modules/model_ray_depth_single_transformer.py
algorithms/wan/modules/clip.py               algorithms/wan/modules/xlm_roberta.py
datasets/{droid,ego4d,epic_kitchen,openx_base,pandas,something_something}.py
datasets/{spatialvid,spatialvid_depth,synthetic_datasets_depth,tanksandtemples}.py
datasets/{dl3dv_imagefolder,dl3dv_test,dummy}.py
datasets/{agibot_world,aria,dl3dv,point_odyssey,scannetpp,scenenet_depth,spatialvid_nvs}.py
utils/{detector_utils,gemini_utils,tf_utils}.py      # only experiments/process_data.py + openx_base
experiments/process_data.py                          # only needed for the process_data task
```

Keep `datasets/aria_new.py` — despite the name it is a live dependency (`camera_to_raymap`) of
eight subset loaders.

## Verification

After pruning, confirm nothing broke without touching a GPU:

```bash
# 1. import + config composition for F
mamba run -n test2 python3 -m main --cfg job --resolve \
  experiment=exp_video algorithm=wan_t2v_ray_depth_mot_concat_5b \
  dataset=mixture_parallax14 dataset.scale_preserving=false cluster=seas_h200 +name=probe >/dev/null

# 2. same for the 1.3B ablations
mamba run -n test2 python3 -m main --cfg job --resolve \
  experiment=exp_video algorithm=wan_t2v_ray_depth_mot_concat_rebuttal \
  dataset=mixture_tartanair_re10k dataset.training.weight_type=absolute \
  cluster=fas_high +name=probe >/dev/null
```

Both must exit 0. Then re-run the tracer to confirm the boot set shrank to the keep-list.

## Repo-path warning

There are **three** working copies of this project and they are not the same filesystem:

| path | state |
|---|---|
| `/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new` | **current / authoritative** |
| `/net/holy-isilon/ifs/rc_labs/ydu_lab/…/video_world_model_new` | stale, and its filesystem is **100% full** |
| `/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model_optim_efficiency_latest` | the copy that trained the 1.3B ablations |

Several committed launch scripts still hardcode the isilon path for `cd` and for `#SBATCH -o`.
Cut the branch from lab_storage and fix those paths, or jobs will run stale code and fail to
write logs.
