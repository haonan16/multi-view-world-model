"""Run cached SAM3 on isolated recorded RGB; retain every prompted candidate.

No depth, cameras, GT masks, click annotations, or simulator inputs are accepted.
Candidates are independent per image: row IDs are not temporal object identities.
Prompt categories are task-supplied; this is not automatic relevance discovery.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rgb_frontend import load_source_bundle, sha256
from artirecon.input_protocol import validate_manifest


def pack_candidates(masks, scores, boxes, height, width):
    masks = np.asarray(masks)
    scores, boxes = np.asarray(scores, dtype=np.float32), np.asarray(boxes, dtype=np.float32)
    # Installed SAM3 skips spatial resizing when thresholding leaves zero
    # instances, so an empty result retains its internal decoder resolution.
    if masks.ndim == 3 and masks.shape[0] == len(scores) == 0 and boxes.shape == (0, 4):
        return np.empty((0, (height * width + 7) // 8), np.uint8), scores, boxes
    if masks.shape != (len(scores), height, width) or boxes.shape != (len(scores), 4):
        raise ValueError(f'SAM3 candidate arrays do not match the native image grid: masks={masks.shape}, '
                         f'scores={scores.shape}, boxes={boxes.shape}, image={(height, width)}')
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('SAM3 returned invalid candidate scores/boxes')
    return np.packbits(masks.astype(bool).reshape(len(scores), height * width), axis=1), scores, boxes


def prepare_request(request):
    if not isinstance(request, dict) or set(request) != {'bundle', 'out', 'prompts'}:
        raise ValueError('Each request must contain only bundle, out, prompts')
    prompts = request['prompts']
    if (not isinstance(prompts, list) or not prompts or len(set(prompts)) != len(prompts)
            or any(not isinstance(p, str) or not p.strip() for p in prompts)):
        raise ValueError('Prompts must be unique nonempty task descriptions')
    bundle, out = Path(request['bundle']).resolve(), Path(request['out']).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite masks: {out}')
    manifest, source, rgb, timestamps = load_source_bundle(bundle)
    return dict(bundle=bundle, out=out, prompts=prompts, manifest=manifest,
                source=source, rgb=rgb, timestamps=timestamps)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def infer_bundle(job, model, processor, *, checkpoint, device, threshold=.3,
                 mask_threshold=.5, model_load_seconds=None):
    import torch
    out, prompts, rgb = job['out'], job['prompts'], job['rgb']
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    t_ids, p_ids, packed, scores, boxes, timings, missing = [], [], [], [], [], [], []
    height, width = rgb.shape[1:3]
    receipt = dict(status='running', frames=len(rgb), prompts=prompts,
                   checkpoint=str(checkpoint), checkpoint_revision=checkpoint.name,
                   model_load_seconds_shared_job=model_load_seconds,
                   threshold=threshold, mask_threshold=mask_threshold,
                   camera_calibration_used=False, measured_depth_used=False, GT_masks_used=False,
                   temporal_object_identity_estimated=False,
                   relevance='Task-supplied text categories; all candidates retained',
                   input_manifest_sha256=sha256(job['bundle'] / 'input-manifest.json'))
    try:
        with torch.inference_mode():
            for t, frame in enumerate(rgb):
                begin = time.monotonic()
                counts = []
                for p, prompt in enumerate(prompts):
                    inputs = processor(images=[frame], text=[prompt], return_tensors='pt').to(device)
                    prediction = model(**inputs)
                    result = processor.post_process_instance_segmentation(
                        prediction, threshold=threshold, mask_threshold=mask_threshold,
                        target_sizes=[(height, width)])[0]
                    m, s, b = pack_candidates(result['masks'].detach().cpu().numpy(),
                                             result['scores'].detach().float().cpu().numpy(),
                                             result['boxes'].detach().float().cpu().numpy(), height, width)
                    n = len(s)
                    t_ids.extend([t] * n); p_ids.extend([p] * n)
                    packed.append(m); scores.append(s); boxes.append(b); counts.append(n)
                    if n == 0:
                        missing.append({'frame_index': t, 'prompt_index': p})
                timings.append(time.monotonic() - begin)
                write_json(out / 'progress.json', dict(status='running', processed_frames=t + 1,
                    frames=len(rgb), elapsed_seconds=time.monotonic() - started, last_candidate_counts=counts))
                print(f'SAM3 {out.name}: {t + 1}/{len(rgb)} frames, candidates={counts}', flush=True)
        # Source bytes must remain identical throughout inference.
        source = job['source']
        source_path = (job['bundle'] / source['path']).resolve()
        if sha256(source_path) != source['sha256']:
            raise ValueError('Source RGB changed during segmentation')
        np.savez_compressed(out / 'masks.npz', mask=np.concatenate(packed, axis=0),
            score=np.concatenate(scores), box=np.concatenate(boxes, axis=0),
            t=np.asarray(t_ids, dtype=np.int64), p=np.asarray(p_ids, dtype=np.int64),
            prompts=np.asarray(prompts), shape=np.asarray([height, width], dtype=np.int64),
            timestamps=job['timestamps'], frame_inference_seconds=np.asarray(timings),
            candidate_row_id=np.arange(len(t_ids), dtype=np.int64))
        manifest = job['manifest']
        source['path'] = str(source_path)
        manifest['artifacts'].extend([
            {'id': 'sam3_weights', 'kind': 'model_prior', 'origin': 'prior', 'prior_type': 'learned_model_weights',
             'scope': 'global', 'parent_ids': [], 'model_name': 'facebook/sam3',
             'revision': checkpoint.name, 'config_sha256': sha256(checkpoint / 'config.json')},
            {'id': 'sam3_masks', 'kind': 'masks', 'origin': 'model_prediction',
             'parent_ids': ['source_rgb', 'sam3_weights'], 'path': 'masks.npz',
             'sha256': sha256(out / 'masks.npz'), 'prompts': prompts,
             'selection': 'All threshold-surviving instances; no temporal identity or cross-class exclusivity imposed'}])
        manifest['method_inputs'].append('sam3_masks')
        validation = validate_manifest(manifest, out, verify_files=True)
        write_json(out / 'input-manifest.json', manifest)
        receipt.update(status='complete', processed_frames=len(rgb), candidates=len(t_ids),
                       missing_detections=missing, frame_inference_seconds=timings,
                       validation=validation,
                       completion_meaning='All requested images processed, not proof of mask correctness or tracking success')
    except Exception as error:
        receipt.update(status='failed', error=repr(error), processed_frames=len(timings))
        raise
    finally:
        receipt['seconds'] = time.monotonic() - started
        write_json(out / 'execution.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True, help='JSON list of {bundle,out,prompts}')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Existing local SAM3 snapshot directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preflight', action='store_true', help='Check isolated inputs and checkpoint presence without inference')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    for name in ['config.json', 'model.safetensors']:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f'Local SAM3 checkpoint file missing/empty: {name}')
    requests = json.loads(args.requests.read_text())
    if not isinstance(requests, list) or not requests:
        raise ValueError('At least one segmentation request is required')
    jobs = [prepare_request(request) for request in requests]
    if len({job['out'] for job in jobs}) != len(jobs):
        raise ValueError('Each request needs a distinct output directory')
    if args.preflight:
        print(json.dumps({'status': 'inputs_ready', 'checkpoint': str(checkpoint),
              'jobs': [{'out': str(job['out']), 'shape': list(job['rgb'].shape), 'prompts': job['prompts']} for job in jobs]}, indent=2))
        return
    import torch
    from transformers import Sam3Model, Sam3Processor
    started = time.monotonic()
    processor = Sam3Processor.from_pretrained(str(checkpoint), local_files_only=True)
    model = Sam3Model.from_pretrained(str(checkpoint), dtype=torch.float32, local_files_only=True).to(args.device).eval()
    load_seconds = time.monotonic() - started
    for job in jobs:
        infer_bundle(job, model, processor, checkpoint=checkpoint, device=args.device, model_load_seconds=load_seconds)


if __name__ == '__main__':
    main()
