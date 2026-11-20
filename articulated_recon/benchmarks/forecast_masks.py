"""Framewise SAM3 candidates on all four strict-forecast RGB streams.

Only frame0/view0 is observed. Generated future/source/helper provenance is
preserved; candidate rows are not tracks or cross-view object identities.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, validate_manifest
from benchmarks.da3_helper_views import read_inputs, generated_parent_ids
from benchmarks.sam3_rgb_masks import pack_candidates


DEFAULT_PROMPTS = ['wooden block on patterned mat', 'hand', 'tabletop']


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def load_forecast(helper, condition, prompts=None):
    prompts = DEFAULT_PROMPTS.copy() if prompts is None else list(prompts)
    if (not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts)
            or len(set(prompts)) != len(prompts)):
        raise ValueError('Use unique nonempty task-supplied prompts')
    # Includes validate_forecast(condition), source equality, temporal indices,
    # the one-image observed mask, and camera/scene ancestry checks before GPU.
    job = read_inputs(helper, condition)
    if job['mode'] != 'forecast':
        raise ValueError('This adapter accepts only strict first-image forecasts')
    job['prompts'] = prompts
    job['generated_parent_ids'] = generated_parent_ids(job['manifest'], job['active_ids'],
                                                      source_only=False, mode='forecast')
    files = dict(observations=job['helper']/'observations.npz',
                 helper_manifest=job['helper']/'input-manifest.json',
                 condition=job['condition']/'condition.npz',
                 condition_manifest=job['condition']/'input-manifest.json',
                 preparation=job['condition']/'preparation.json')
    job['input_files'] = {name: dict(path=str(path), sha256=file_sha256(path)) for name, path in files.items()}
    return job


def _unchanged(job):
    for name, record in job['input_files'].items():
        if file_sha256(record['path']) != record['sha256']:
            raise ValueError('Forecast input changed during segmentation: ' + name)
    # Also verify all active first-image/camera/scene ancestor file identities.
    load_forecast(job['helper'], job['condition'], job['prompts'])


def write_product(job, checkpoint, output, arrays, *, source_only=False):
    """Write all-view or independently computed source-view candidate evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'masks.npz', **arrays)
    manifest = deepcopy(job['manifest'])
    weights_id, masks_id = 'forecast_sam3_weights', 'forecast_sam3_masks'
    if any(a['id'] in {weights_id, masks_id} for a in manifest['artifacts']):
        raise ValueError('Forecast mask artifact IDs already exist in source lineage')
    parents = generated_parent_ids(manifest, job['active_ids'], source_only=source_only, mode='forecast')
    views = [0] if source_only else [0, 1, 2, 3]
    manifest['artifacts'] += [
        dict(id=weights_id, kind='model_prior', origin='prior', prior_type='learned_model_weights',
             scope='global', parent_ids=[], model_name='facebook/sam3', revision=checkpoint.name,
             config_sha256=file_sha256(checkpoint/'config.json')),
        dict(id=masks_id, kind='masks', origin='model_prediction',
             parent_ids=parents+[weights_id], path='masks.npz', sha256=file_sha256(output/'masks.npz'),
             prompts=job['prompts'], frame_indices=list(range(len(job['timestamps']))), view_indices=views,
             selection='Independent framewise predictions; all prompted candidates for declared frame/view pairs',
             temporal_tracking=False, cross_view_identity_estimated=False,
             source_subset_of_independent_inference=source_only)]
    manifest['method_inputs'] = [masks_id]
    validation = validate_manifest(manifest, output, verify_files=True)
    write_json(output/'input-manifest.json', manifest)
    return dict(mask_artifact_id=masks_id, masks_sha256=file_sha256(output/'masks.npz'),
                validation=validation, generated_parent_ids=parents, selected_view_indices=views)


def run(helper, condition, output, checkpoint, predict, *, prompts=None, threshold=.3,
        mask_threshold=.5, model_load_seconds=None):
    """Consume each exact forecast image; predict(image,prompt) returns m,s,b.

    The callback contract lets CPU tests verify frame/view routing and lineage
    without pretending to run SAM3. The CLI supplies the real cached SAM3 model.
    """
    for value in (threshold, mask_threshold):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError('SAM3 thresholds must be finite and between zero and one')
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    if output.exists(): raise FileExistsError(output)
    job = load_forecast(helper, condition, prompts)
    rgb = job['rgb']; frames, views, height, width, _ = rgb.shape
    output.mkdir(parents=True)
    started = time.monotonic()
    receipt = dict(status='running', mode='strict_forecast_framewise_sam3', frames=frames, views=views,
        images=frames*views, prompts=job['prompts'], threshold=threshold, mask_threshold=mask_threshold,
        checkpoint=str(checkpoint), checkpoint_revision=checkpoint.name, model_load_seconds=model_load_seconds,
        input_files=job['input_files'], source_observed_until_s=job['manifest']['observed_until_s'],
        source_rgb_sha256=job['observed']['sha256'], generated_parent_ids=job['generated_parent_ids'],
        producer_sha256=file_sha256(__file__), image_order='t-major, then v=0..3; every prompt per image',
        observed_images=int(job['observed_mask'].sum()), generated_images=int(rgb.shape[0]*rgb.shape[1]-job['observed_mask'].sum()),
        depth_or_camera_arrays_passed_to_model=False, simulator_used=False, GT_masks_used=False,
        temporal_tracking=False, cross_view_identity_estimated=False,
        task_relevance='Prompt categories supplied by caller; all surviving candidates retained')
    write_json(output/'execution.json', receipt)
    chunks, scores, boxes, tids, vids, pids = [], [], [], [], [], []
    seconds = np.zeros((frames, views), dtype=float)
    counts = np.zeros((frames, views, len(job['prompts'])), dtype=np.int64)
    processed = 0
    try:
        for t in range(frames):
            for v in range(views):
                begin = time.monotonic()
                for p, prompt in enumerate(job['prompts']):
                    # A copy prevents a third-party processor from changing the
                    # authenticated in-memory packet used by later prompts.
                    masks, score, box = predict(rgb[t, v].copy(), prompt)
                    packed, score, box = pack_candidates(masks, score, box, height, width)
                    n = len(score)
                    chunks.append(packed); scores.append(score); boxes.append(box)
                    tids.extend([t]*n); vids.extend([v]*n); pids.extend([p]*n)
                    counts[t, v, p] = n
                seconds[t, v] = time.monotonic() - begin
                processed += 1
                write_json(output/'progress.json', dict(status='running', processed_images=processed,
                    total_images=frames*views, frame_index=t, view_index=v,
                    last_candidate_counts=counts[t, v].tolist(), seconds=time.monotonic()-started))
                print(f'SAM3 forecast {processed}/{frames*views}: t={t} v={v}, candidates={counts[t,v].tolist()}', flush=True)
        _unchanged(job)
        arrays = dict(mask=np.concatenate(chunks, axis=0),
            score=np.concatenate(scores), box=np.concatenate(boxes, axis=0),
            t=np.asarray(tids, dtype=np.int64), v=np.asarray(vids, dtype=np.int64), p=np.asarray(pids, dtype=np.int64),
            prompts=np.asarray(job['prompts']), shape=np.asarray([height, width], dtype=np.int64),
            timestamps=job['timestamps'], observed_mask=job['observed_mask'], source_frame_indices=job['ids'],
            frame_inference_seconds=seconds, candidate_counts=counts,
            image_rgb_sha256=np.asarray([[hashlib.sha256(rgb[t,v].tobytes()).hexdigest()
                                         for v in range(views)] for t in range(frames)]),
            candidate_row_id=np.arange(len(tids), dtype=np.int64))
        product = write_product(job, checkpoint, output, arrays)
        # A source-only control must not merely filter masks while retaining
        # helper dependence. Each model call was independent and stateless:
        # export its own selected candidates and source-forecast-only ancestry.
        subset = {name: value.copy() for name, value in arrays.items()}
        keep = arrays['v'] == 0
        for name in ['mask', 'score', 'box', 't', 'v', 'p', 'candidate_row_id']:
            subset[name] = arrays[name][keep]
        for name in ['observed_mask', 'frame_inference_seconds', 'candidate_counts', 'image_rgb_sha256']:
            subset[name] = arrays[name][:, :1]
        source_product = write_product(job, checkpoint, output/'source-only', subset, source_only=True)
        write_json(output/'source-only/execution.json', dict(status='complete', mode='source_subset_of_independent_framewise_sam3',
            **source_product, producer_sha256=file_sha256(__file__), input_files=job['input_files'],
            image_order='Only t=0..T-1,v=0 results from independent image/prompt calls',
            helper_pixels_used_for_selected_predictions=False, temporal_tracking=False,
            source_observed_until_s=job['manifest']['observed_until_s'],
            observed_images=1, generated_images=frames-1, frames=frames, views=1,
            processed_images=frames, candidates=int(keep.sum()), prompts=job['prompts'],
            threshold=threshold, mask_threshold=mask_threshold,
            runtime_note='Source calls executed as part of the all-view run; per-image times retained, shared model load not counted again'))
        receipt.update(status='complete', processed_images=processed, candidates=len(tids), **product,
            source_only_product=source_product,
            completion_meaning='All forecast images segmented; no claim of mask correctness, temporal tracking or object identity')
        write_json(output/'progress.json', dict(status='complete', processed_images=processed,
            total_images=frames*views, candidates=len(tids)))
    except Exception as error:
        receipt.update(status='failed', error=repr(error), processed_images=processed)
        raise
    finally:
        receipt['seconds'] = time.monotonic() - started
        write_json(output/'execution.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['helper', 'condition', 'output', 'checkpoint']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prompts', nargs='+', default=DEFAULT_PROMPTS)
    parser.add_argument('--threshold', type=float, default=.3)
    parser.add_argument('--mask-threshold', type=float, default=.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    job = load_forecast(args.helper, args.condition, args.prompts)
    checkpoint = args.checkpoint.resolve()
    for name in ['config.json', 'model.safetensors']:
        if not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size:
            raise ValueError('Cached SAM3 file missing or empty: ' + name)
    if args.output.exists(): raise FileExistsError(args.output)
    if args.preflight:
        print(json.dumps(dict(status='inputs_ready', shape=list(job['rgb'].shape), prompts=job['prompts'],
            observed_images=int(job['observed_mask'].sum()), generated_parent_ids=job['generated_parent_ids'],
            input_files=job['input_files']), indent=2))
        return
    import torch
    from transformers import Sam3Model, Sam3Processor
    start = time.monotonic()
    processor = Sam3Processor.from_pretrained(str(checkpoint), local_files_only=True)
    model = Sam3Model.from_pretrained(str(checkpoint), dtype=torch.float32, local_files_only=True).to(args.device).eval()
    load_seconds = time.monotonic() - start
    def predict(image, prompt):
        inputs = processor(images=[image], text=[prompt], return_tensors='pt').to(args.device)
        output = model(**inputs)
        result = processor.post_process_instance_segmentation(output, threshold=args.threshold,
            mask_threshold=args.mask_threshold, target_sizes=[image.shape[:2]])[0]
        return (result['masks'].detach().cpu().numpy(), result['scores'].detach().float().cpu().numpy(),
                result['boxes'].detach().float().cpu().numpy())
    with torch.inference_mode():
        run(args.helper, args.condition, args.output, checkpoint, predict, prompts=args.prompts,
            threshold=args.threshold, mask_threshold=args.mask_threshold, model_load_seconds=load_seconds)


if __name__ == '__main__': main()
