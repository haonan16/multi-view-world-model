# Downloading RH20T (full resolution, all views, RGB + depth + low-dim + calib)

How to rebuild a complete local copy of **RH20T**
([paper](https://arxiv.org/abs/2307.00595) · [project page](https://rh20t.github.io/) · [API](https://github.com/rh20t/rh20t_api))
at the full 640×360 resolution: all 7 rig configs, all 8–10 cameras per episode,
RGB and depth video, low-dim robot signals, calibration, and the paired human demos.

The authors' Google Drive links are almost always over their download quota, and
the low-dim and calibration archives were only ever on Baidu. Everything below
comes from HuggingFace instead.

| | |
|---|---|
| Download size | ~1.54 TB of archives (29 logical archives, some split into parts) |
| Extracted size | **~1.76 TiB** (1,758.7 GiB) |
| Episodes | ~24.9k folders: ~13.3k robot + ~11.5k human demos |
| Time | ~1 day on a good link (resumable) + several hours to extract |

> **Disk:** downloading everything and then extracting needs about 3.3 TB at the
> peak. If space is tight, use `DELETE_AFTER=1` in step 3 so each archive is deleted
> once it has extracted (peak ≈ 1.8 TB + the largest archive).

## Sources

| HF dataset repo | Files used | Contents | Size |
|---|---|---|---|
| [`Malak-Mansour/RH20T`](https://huggingface.co/datasets/Malak-Mansour/RH20T) | all | Full-res **RGB**, cfg1–7 (`RH20T_cfgN.part.aa…`) | 0.60 TB |
| [`HaojunChen/RH20T`](https://huggingface.co/datasets/HaojunChen/RH20T) | `depth/*` | Full-res **depth**, cfg1–7 (`RH20T_cfgN_depth.tar.gz.part-000…`) | 0.70 TB |
| [`tonyfanggalaxies/RH20T`](https://huggingface.co/datasets/tonyfanggalaxies/RH20T) (official authors' upload) | `RH20T_lowdim/*`, `RH20T_calib/*` | Low-dim robot data + calibration, cfg1–7 | ~0.24 TB |
| [`ldn060904/rh20t`](https://huggingface.co/datasets/ldn060904/rh20t) | `patch.tar.gz` only | Official calib/joint-angle **patch** for cfg1 + cfg2 | 3.7 GB |

The color and depth mirrors hold the original authors' archives, byte for byte. All
29 passed a full gzip CRC check when we did this (Sept 2026).

## 1. Download

```bash
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export STAGING=/path/to/rh20t_staging     # ~1.54 TB
# huggingface-cli login                   # only if a repo returns 401/403
```

`download.py`:

```python
import sys
from huggingface_hub import snapshot_download

staging = sys.argv[1]
JOBS = [
    ("Malak-Mansour/RH20T", None),                                    # RGB
    ("HaojunChen/RH20T", ["depth/*"]),                                # depth
    ("ldn060904/rh20t", ["patch.tar.gz"]),                            # cfg1/2 patch
    ("tonyfanggalaxies/RH20T", ["RH20T_lowdim/*", "RH20T_calib/*"]),  # lowdim + calib
]
for repo, patterns in JOBS:
    print(f"=== {repo} {patterns or 'ALL'}", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns,
                      local_dir=f"{staging}/{repo.replace('/', '__')}", max_workers=8)
print("all downloads complete")
```

```bash
python3 download.py $STAGING
```

You can resume it: files that already finished are skipped, so just rerun it if it
dies or times out. It is I/O-bound, so 1–2 CPUs are plenty.

## 2. (Optional) Validate

Extraction in step 3 already catches corruption, because tar/gzip exit non-zero and
the script stops. To check every archive *before* extracting:

```bash
cat $STAGING/HaojunChen__RH20T/depth/RH20T_cfg1_depth.tar.gz.part-* | tar tzf - > /dev/null && echo OK
```

The color cfg1 archive is not gzipped (see quirks), so use `tar tf -` for it.

## 3. Extract

Split parts are never joined on disk. They are streamed: `cat parts | tar x`.

**Mirror quirks the script handles:**
1. `Malak-Mansour` **color cfg1** was repacked as an **uncompressed** tar wrapped in an
   extra `ali.abouzeid/` top folder. The script detects gzip vs. plain tar and strips the prefix.
2. **Color cfg2** has a stray symlink member
   `RH20T_cfg2/task_0078_user_0014_scene_0004_cfg_0002/transformed`. It collides
   with the real `transformed/` folder from the low-dim archive, so it is excluded
   (harmless; the real data comes from low-dim).
3. Order matters: calib → patch → low-dim → depth → color.

`extract.sh`:

```bash
#!/bin/bash
# usage: STAGING=... DEST=... [DELETE_AFTER=1] bash extract.sh
set -euo pipefail
: "${STAGING:?}" "${DEST:?}"; DELETE_AFTER=${DELETE_AFTER:-0}
M=$STAGING/Malak-Mansour__RH20T
H=$STAGING/HaojunChen__RH20T/depth
T=$STAGING/tonyfanggalaxies__RH20T
L=$STAGING/ldn060904__rh20t
mkdir -p "$DEST/data" "$DEST/calib" "$DEST/.done"

extract() {  # name outdir part-file...
  local name=$1 out=$2; shift 2
  local parts=("$@")
  [ -e "$DEST/.done/$name" ] && { echo "skip $name"; return; }
  [ ${#parts[@]} -gt 0 ] && [ -e "${parts[0]}" ] || { echo "MISSING $name"; exit 1; }
  local magic; magic=$(head -c2 "${parts[0]}" | od -An -tx1 | tr -d ' ')
  local z=""; [ "$magic" = "1f8b" ] && z="z"
  # strip the uploader prefix dir if the archive isn't rooted at RH20T_cfg*/calib/patch
  local first; first=$(cat "${parts[@]}" | tar t${z}f - 2>/dev/null | head -1 || true)
  local strip=0; case "${first%%/*}" in RH20T_cfg*|calib|patch) ;; *) strip=1 ;; esac
  echo "=== $name -> $out (${#parts[@]} parts, gzip=${z:-no}, strip=$strip)"
  cat "${parts[@]}" | tar x${z}f - --strip-components=$strip \
      --exclude='*/task_0078_user_0014_scene_0004_cfg_0002/transformed' -C "$out"
  touch "$DEST/.done/$name"
  [ "$DELETE_AFTER" = 1 ] && rm -f "${parts[@]}"
  return 0
}

extract patch "$DEST/calib" "$L/patch.tar.gz"
for i in 1 2 3 4 5 6 7; do extract calib_cfg$i  "$DEST/calib" "$T/RH20T_calib/RH20T_cfg$i.tar.gz";  done
for i in 1 2 3 4 5 6 7; do extract lowdim_cfg$i "$DEST/data"  "$T/RH20T_lowdim/RH20T_cfg$i.tar.gz"; done
for i in 1 2 3 4 5 6 7; do extract depth_cfg$i  "$DEST/data"  "$H"/RH20T_cfg${i}_depth.tar.gz.part-*; done
for i in 1 2 3 4 5 6 7; do extract color_cfg$i  "$DEST/data"  "$M"/RH20T_cfg$i.part.*; done
echo "ALL DONE"; du -sh "$DEST"
```

You can rerun it: finished archives are marked in `$DEST/.done/`. One CPU each for
gzip and tar is enough.

**Sanity check:** you should get roughly 24,874 episode folders and ~1.76 TiB.
```bash
ls -d $DEST/data/RH20T_cfg*/task_* | wc -l
ls -d $DEST/data/RH20T_cfg*/task_*_human | wc -l     # ~11,535
```

## Resulting layout

```
$DEST/
├── data/RH20T_cfg{1..7}/
│   ├── task_TTTT_user_UUUU_scene_SSSS_cfg_CCCC/     # robot episode
│   │   ├── cam_<serial>/                            # 8-10 synchronized cameras
│   │   │   ├── color.mp4                            # RGB 640x360 @10Hz
│   │   │   ├── depth.mp4                            # video-encoded depth (see caveats)
│   │   │   └── timestamps.npy                       # per-frame capture times (ms)
│   │   ├── transformed/                             # tcp, tcp_base, joint, gripper,
│   │   │                                            #   force_torque(_base), high_freq_data .npy
│   │   ├── audio_mixed/  robot_command/  pedal_command/
│   │   └── metadata.json                            # rating, calib timestamp, calib_quality
│   └── task_..._cfg_CCCC_human/                     # paired human demo (cams, audio, metadata)
└── calib/
    ├── RH20T_cfg{1..7}/calib/<ts>/                  # intrinsics + extrinsics per calib timestamp
    └── patch/RH20T_cfg{1,2}/                        # official fixes for cfg1/2 (see caveats)
```

| cfg | robot + gripper | episodes | human demos | cams |
|---|---|--:|--:|:-:|
| 1 | Flexiv Rizon 4s + Dahuan AG-95 | 8,809 | 4,188 | 9 |
| 2 | Flexiv Rizon 4s + Robotiq 2F-85 | 3,765 | 1,793 | 8 |
| 3 | UR5 + WSG-50 | 1,575 | 775 | 8 |
| 4 | UR5 + Robotiq 2F-85 | 4,338 | 2,124 | 8 |
| 5 | Franka Emika + Franka gripper | 2,515 | 1,191 | 8–9 |
| 6 | Kuka LBR iiwa + Robotiq 2F-85 | 2,955 | 1,464 | 10 |
| 7 | Kuka LBR iiwa + Robotiq 2F-85 (+ tactile) | 917 | 0 | 10 |

## Usage notes and caveats

- **Read the data with [rh20t_api](https://github.com/rh20t/rh20t_api).** It decodes depth (don't use plain ffmpeg),
  looks up calibration, and aligns timing. Cameras are calibrated to the robot
  base frame; use each camera's `timestamps.npy`, because frame times differ slightly across cameras.
- `metadata.json → rating`: 1 = failure, 2–9 = completion quality. `calib` is the
  timestamp key into `calib/RH20T_cfgN/calib/<ts>/`. `calib_quality`: 0 = some
  cameras uncalibrated, 1–5 = accuracy (lower is better).
- **Apply the cfg1/cfg2 patch** (the authors recommend it):
  `rsync -a $DEST/calib/patch/RH20T_cfg1/ $DEST/data/RH20T_cfg1/` (same for cfg2).
  The script above deliberately leaves it unmerged.
- **Depth is noisy:** `depth.mp4` is lossy video-encoded. The authors' fixed depth exists
  only at **320×180**, in `tonyfanggalaxies/RH20T` (`RH20T_depth/`, not downloaded here).
  That repo also has a full official 320×180 release (`RH20T_color/`) if low-res is enough.
- cfg7 has no human demos and is the only config with tactile data.
- **License differs by scene:** scenes 0001–0005 are CC BY-SA 4.0; scenes 0006–0010 are
  **CC BY-NC 4.0** (non-commercial). The scene id is in the folder name.
- Questions about the dataset itself: RH20T authors (fhaoshu@gmail.com).
