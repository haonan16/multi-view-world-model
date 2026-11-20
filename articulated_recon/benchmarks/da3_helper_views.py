"""Matched DA3 ablation on recorded reconstruction or strict forecast pixels.

Each timestamp is inferred independently: this does not estimate a rigid temporal
scene. DA3 Giant's native scale is arbitrary. The optional requested-camera run
uses model-designed poses as an uncertain conditioning hypothesis, never GT.
Forecast mode consumes the observed initial image and generated future pixels;
it never restores a recorded future source frame.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import load_manifest, validate_manifest, file_sha256


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def absolute_manifest(path):
    result = json.loads(path.read_text())
    for a in result['artifacts']:
        if 'path' in a:
            a['path'] = str((path.parent / a['path']).resolve())
    return result


def verify_matched_pixels(helper, source, times, source_times, ids, crop):
    """Recreate every observed pixel directly from the isolated original RGB."""
    import cv2
    if helper.ndim != 5 or helper.shape[1] != 4 or helper.dtype != np.uint8:
        raise ValueError('Expected uint8 [T,4,H,W,3] helper observations')
    if ids.shape != (len(helper),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Invalid source frame indices')
    if np.any(ids < 0) or np.any(ids >= len(source)) or np.any(np.diff(ids) <= 0):
        raise ValueError('Source indices must be increasing and inside recorded sequence')
    if times.shape != (len(helper),) or not np.array_equal(times, source_times[ids]):
        raise ValueError('Generated/helper timing differs from recorded source')
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > source.shape[2] or y+h > source.shape[1]:
        raise ValueError('Crop lies outside recorded RGB')
    size = (helper.shape[3], helper.shape[2])
    matched = np.stack([cv2.resize(im[y:y+h, x:x+w], size, interpolation=cv2.INTER_AREA)
                        for im in source[ids]])
    if not np.array_equal(helper[:, 0], matched):
        raise ValueError('Helper view0 is not the exact resized/cropped recorded RGB')
    return matched


def verify_forecast_pixels(helper, source, times, source_times, ids, known, crop):
    """Verify only the observed anchor; preserve all generated future pixels."""
    if (helper.ndim != 5 or helper.shape[1] != 4 or helper.shape[-1] != 3
            or helper.dtype != np.uint8 or len(helper) < 2):
        raise ValueError('Expected uint8 forecast [T>=2,4,H,W,3]')
    if (source.ndim != 4 or len(source) != 1 or source.dtype != np.uint8
            or np.asarray(source_times).shape != (1,)):
        raise ValueError('Strict forecast ancestry must contain exactly one observed image')
    if (times.shape != (len(helper),) or not np.isfinite(times).all()
            or times[0] != source_times[0] or not (np.diff(times) > 0).all()):
        raise ValueError('Forecast timing must start at the observed anchor and increase')
    expected_ids = np.full(len(helper), -1, dtype=np.int64)
    expected_ids[0] = 0
    if ids.dtype.kind not in 'iu' or not np.array_equal(ids, expected_ids):
        raise ValueError('Future forecast frames have no recorded source index; use -1')
    expected_known = np.zeros(helper.shape[:2], bool)
    expected_known[0, 0] = True
    if known.dtype.kind != 'b' or not np.array_equal(known, expected_known):
        raise ValueError('Only the initial source image may be marked observed in a forecast')
    verify_matched_pixels(helper[:1], source, times[:1], source_times,
                          np.array([0], dtype=np.int64), crop)
    # Deliberately copy the generated source-camera future, not source[ids].
    return helper[:, 0].copy()


def generated_parent_ids(manifest, active_ids, *, source_only, mode):
    """Bind the pixels actually sent to DA3, including predicted future motion."""
    wanted = {'source_forecast'} if source_only else {'source_forecast', 'helper'}
    if mode == 'reconstruction':
        wanted = {'helper'}
    records = [a for a in manifest['artifacts'] if a['id'] in active_ids
               and a['kind'] == 'rgb' and a['origin'] == 'model_prediction'
               and a.get('view_role') in wanted]
    if {a.get('view_role') for a in records} != wanted:
        raise ValueError('Missing generated RGB lineage for the selected DA3 streams')
    return [a['id'] for a in records]


def read_inputs(helper, condition):
    helper, condition = Path(helper).resolve(), Path(condition).resolve()
    report = load_manifest(helper/'input-manifest.json', [('rgb', helper/'observations.npz')])
    condition_report = load_manifest(condition/'input-manifest.json', [('cameras', condition/'condition.npz')])
    mode = report['task_mode']
    if (mode not in {'reconstruction', 'forecast'} or condition_report['task_mode'] != mode
            or any(r['condition'] != 'rgb_only' for r in [report, condition_report])):
        raise ValueError('Require matching RGB-only reconstruction or forecast inputs')
    if mode == 'forecast':
        # A forecast label alone does not exclude future camera/scene arrays.
        # Reuse the generator's first-image ancestry and array-content checks.
        from benchmarks.strict_forecast import validate_forecast
        _, strict_report = validate_forecast(condition)
        if strict_report['manifest_sha256'] != condition_report['manifest_sha256']:
            raise ValueError('Forecast preparation manifest changed during validation')
    manifest = absolute_manifest(helper/'input-manifest.json')
    active = set(report['method_artifact_ids'])
    observed = [a for a in manifest['artifacts'] if a['id'] in active and a['kind']=='rgb'
                and a['origin']=='observed_rgb']
    if len(observed) != 1:
        raise ValueError('Exactly one observed RGB camera is required')
    condition_manifest = absolute_manifest(condition/'input-manifest.json')
    condition_sources = [a for a in condition_manifest['artifacts']
                         if a['id'] in condition_report['method_artifact_ids'] and a['origin'] == 'observed_rgb']
    if len(condition_sources) != 1 or condition_sources[0]['sha256'] != observed[0]['sha256']:
        raise ValueError('Camera preparation and generated pixels must have the identical observed RGB root')
    with np.load(observed[0]['path'], allow_pickle=False) as z:
        if set(z.files) != {'rgb', 'timestamps'}:
            raise ValueError('Observed ancestry must be an isolated RGB/timestamps bundle')
        source, source_times = z['rgb'].copy(), z['timestamps'].copy()
    with np.load(helper/'observations.npz', allow_pickle=False) as z:
        rgb, times, ids, known, k, c = [z[n].copy() for n in
            ['rgb','timestamps','source_frame_indices','observed_mask','K','c2w']]
    with np.load(condition/'condition.npz', allow_pickle=False) as z:
        conditioned_rgb = rgb[:, 0] if mode == 'reconstruction' else rgb[:1, 0]
        if not all(np.array_equal(z[n], a) for n, a in
                   [('rgb',conditioned_rgb),('K',k),('c2w',c),('timestamps',times),('source_frame_indices',ids)]):
            raise ValueError('Requested camera/input condition differs from generated bundle')
    prep = json.loads((condition/'preparation.json').read_text())
    if mode == 'forecast':
        if prep.get('mode') != 'forecast' or manifest.get('observed_until_s') != source_times[0]:
            raise ValueError('Forecast preparation/cutoff differs from initial observed image')
        matched = verify_forecast_pixels(rgb, source, times, source_times, ids, known, prep['source_crop_xywh'])
        generated_parent_ids(manifest, active, source_only=False, mode=mode)
    else:
        if (known.shape != rgb.shape[:2] or not np.all(known[:,0]) or np.any(known[:,1:])):
            raise ValueError('Only original source view0 may be marked observed')
        matched = verify_matched_pixels(rgb, source, times, source_times, ids, prep['source_crop_xywh'])
    if (k.shape != (4,3,3) or c.shape != (4,4,4) or not np.isfinite(k).all()
            or not np.isfinite(c).all() or not np.allclose(c[0],np.eye(4))):
        raise ValueError('Malformed model-designed camera requests')
    camera_ids = [a['id'] for a in manifest['artifacts'] if a['id'] in active
                  and a['kind'] == 'cameras' and a.get('path') == str(condition/'condition.npz')]
    if not camera_ids:
        raise ValueError('Generated bundle lacks authenticated requested-camera ancestry')
    return dict(helper=helper, condition=condition, manifest=manifest, rgb=rgb, source=matched,
                observed=observed[0], timestamps=times, ids=ids, K=k, c2w=c,
                mode=mode, observed_mask=known, active_ids=active, camera_request_ids=camera_ids,
                helper_sha256=file_sha256(helper/'observations.npz'))


def prediction_arrays(prediction, original_hw):
    """Return depth and K on the input grid; DA3 resize mode performs no crop."""
    import cv2
    d = np.asarray(prediction.depth, np.float32)
    k = np.asarray(prediction.intrinsics, np.float32).copy()
    e = np.asarray(prediction.extrinsics, np.float32)
    if d.ndim != 3 or k.shape != (len(d),3,3) or e.shape not in {(len(d),3,4),(len(d),4,4)}:
        raise ValueError('DA3 did not return complete depth and camera predictions')
    if not np.isfinite(d).all() or np.any(d <= 0) or not np.isfinite(k).all() or not np.isfinite(e).all():
        raise ValueError('DA3 returned nonfinite or nonpositive geometry')
    if e.shape[-2] == 3:
        full = np.broadcast_to(np.eye(4,dtype=np.float32),(len(d),4,4)).copy()
        full[:,:3] = e
        e = full
    raw_c = np.linalg.inv(e)
    # Gauge each independent frame to its source camera. This does NOT estimate
    # cross-time camera motion; fixed source camera remains a capture assumption.
    c = np.linalg.inv(raw_c[0])[None] @ raw_c
    h, w = original_hw
    k[:,0] *= w/d.shape[2]; k[:,1] *= h/d.shape[1]
    depth = np.stack([cv2.resize(v,(w,h),interpolation=cv2.INTER_LINEAR) for v in d])
    if prediction.conf is None:
        confidence = np.full(depth.shape,np.nan,np.float32)
    else:
        conf = np.asarray(prediction.conf,np.float32)
        if conf.shape != d.shape or not np.isfinite(conf).all():
            raise ValueError('Invalid DA3 confidence grid')
        confidence = np.stack([cv2.resize(v,(w,h),interpolation=cv2.INTER_LINEAR) for v in conf])
    return dict(depth=depth,confidence=confidence,K=k,c2w=c,raw_c2w=raw_c,
                native_hw=np.array(d.shape[-2:],np.int64),is_metric=bool(prediction.is_metric))


def run_condition(job, model, checkpoint, output, name, process_res, load_seconds):
    out = output/name
    out.mkdir(parents=True, exist_ok=False)
    source_only = name == 'source_unposed'
    requested = name == 'helper_requested_cameras'
    mode = job.get('mode', 'reconstruction')
    frames = job['source'][:,None] if source_only else job['rgb']
    start = time.monotonic()
    receipt = dict(status='running',condition=name,frame_count=len(frames),views=frames.shape[1],
        model_load_seconds_shared_job=load_seconds,process_res=process_res,
        temporal_mode='Independent synchronized view groups per timestamp, no cross-time rigid fitting',
        camera_mode='Uncertain designed virtual camera hypothesis' if requested else 'Estimated from RGB by DA3',
        supplied_dataset_calibration=False,sensor_depth_used=False,
        confidence_interpretation='DA3 score, not calibrated probability',
        scale_interpretation='Inherited model-designed camera baseline; no new metric-scale measurement' if requested else 'Arbitrary DA3 Giant scale per timestamp; not metric',
        task_mode=mode, source_pixels_verified=mode == 'reconstruction',
        initial_observed_pixels_verified=True,
        source_stream='observed_initial_plus_generated_future' if mode == 'forecast' else 'recorded_source_sequence',
        generated_views_used=not source_only or mode == 'forecast', generated_helper_views_used=not source_only,
        code_sha256=file_sha256(__file__),checkpoint_revision=checkpoint.name)
    arrays, seconds = [], []
    try:
        for index, views in enumerate(frames):
            before = time.monotonic()
            kwargs = dict(process_res=process_res,process_res_method='upper_bound_resize',
                          ref_view_strategy='first',align_to_input_ext_scale=requested)
            if requested:
                kwargs.update(extrinsics=np.linalg.inv(job['c2w']),intrinsics=job['K'].copy())
            else:
                kwargs.update(extrinsics=None,intrinsics=None)
            prediction = model.inference(list(views),**kwargs)
            arrays.append(prediction_arrays(prediction,views.shape[1:3]))
            seconds.append(time.monotonic()-before)
            write_json(out/'progress.json',dict(status='running',processed_frames=index+1,
                frames=len(frames),elapsed_seconds=time.monotonic()-start))
            print(f'DA3 {name} {index+1}/{len(frames)}',flush=True)
        if file_sha256(job['helper']/'observations.npz') != job['helper_sha256']:
            raise ValueError('Helper RGB changed during inference')
        if file_sha256(job['observed']['path']) != job['observed']['sha256']:
            raise ValueError('Recorded RGB changed during inference')
        evidence_flags = {}
        if 'observed_mask' in job:
            evidence_flags['observed_mask'] = job['observed_mask'][:, :1] if source_only else job['observed_mask']
        np.savez_compressed(out/'predictions.npz',rgb=frames,timestamps=job['timestamps'],
            source_frame_indices=job['ids'],frame_inference_seconds=np.asarray(seconds),
            **evidence_flags,
            **{k:np.stack([a[k] for a in arrays]) for k in arrays[0]})
        if source_only and mode == 'reconstruction':
            np.savez_compressed(out/'matched-source.npz',rgb=job['source'],timestamps=job['timestamps'])
            source_record=copy.deepcopy(job['observed'])
            manifest=dict(schema_version=1,condition='rgb_only',task_mode='reconstruction',artifacts=[source_record],method_inputs=[])
            # The protocol's observed_rgb category is root-only. Bind prediction
            # lineage to original recorded RGB and audit this deterministic view
            # separately; do not relabel a crop as generated imagery.
            receipt['matched_source']=dict(path='matched-source.npz',sha256=file_sha256(out/'matched-source.npz'),
                source_artifact_id=source_record['id'],
                derivation='Deterministic crop/resize of original recorded pixels, equality verified with helper view0')
            parents=[source_record['id']]
        else:
            manifest=copy.deepcopy(job['manifest'])
            parents=generated_parent_ids(manifest, job['active_ids'], source_only=source_only, mode=mode)
            if requested: parents.extend(job['camera_request_ids'])
            if source_only:
                receipt['source_forecast']=dict(
                    derivation='Generated source-camera future preserved exactly; only frame0 is observed',
                    parent_artifact_ids=parents)
        manifest['artifacts'].append(dict(id='da3_weights',kind='model_prior',origin='prior',
            prior_type='learned_model_weights',scope='global',parent_ids=[],
            model_name='depth-anything/DA3-GIANT-1.1',revision=checkpoint.name,
            config_sha256=file_sha256(checkpoint/'config.json')))
        for kind in ['depth','cameras']:
            manifest['artifacts'].append(dict(id='da3_'+kind,kind=kind,origin='model_prediction',
                parent_ids=parents+['da3_weights'],path='predictions.npz',sha256=file_sha256(out/'predictions.npz')))
        manifest['method_inputs']=['da3_depth','da3_cameras']
        inherited_scale = job.get('manifest', {}).get('metric_scale', {}).get('status', 'unresolved')
        metric_status = 'estimated' if requested and inherited_scale in {'known', 'estimated'} else 'unresolved'
        manifest['metric_scale']=dict(status=metric_status,
            artifact_ids=job.get('camera_request_ids', []) if metric_status == 'estimated' else [],description=receipt['scale_interpretation'])
        manifest['robot_registration']=dict(status='unresolved',artifact_ids=[],description='No robot-to-recorded-scene registration')
        receipt['validation']=validate_manifest(manifest,out,verify_files=True)
        write_json(out/'input-manifest.json',manifest)
        receipt.update(status='complete',frame_inference_seconds=seconds,
            output_sha256=file_sha256(out/'predictions.npz'),
            completion_meaning='Predictions computed; no claim of geometry accuracy or execution success')
        write_json(out/'progress.json',dict(status='complete',processed_frames=len(frames),frames=len(frames)))
    except Exception as error:
        receipt.update(status='failed',error=repr(error),processed_frames=len(arrays))
        raise
    finally:
        receipt['seconds']=time.monotonic()-start
        write_json(out/'execution.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--helper',type=Path,required=True)
    p.add_argument('--condition',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--process-res',type=int,default=320)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    job=read_inputs(args.helper,args.condition)
    checkpoint=args.checkpoint.resolve()
    for n in ['config.json','model.safetensors']:
        if not (checkpoint/n).is_file() or not (checkpoint/n).stat().st_size:
            raise ValueError('Missing cached DA3 checkpoint file: '+n)
    if args.output.exists(): raise FileExistsError(args.output)
    if args.preflight:
        print(json.dumps(dict(status='ready',frames=len(job['rgb']),shape=list(job['rgb'].shape),checkpoint=str(checkpoint)),indent=2))
        return
    from depth_anything_3.api import DepthAnything3
    start=time.monotonic()
    model=DepthAnything3.from_pretrained(str(checkpoint),local_files_only=True).to('cuda').eval()
    load_seconds=time.monotonic()-start
    args.output.mkdir(parents=True)
    for name in ['source_unposed','helper_unposed','helper_requested_cameras']:
        run_condition(job,model,checkpoint,args.output,name,args.process_res,load_seconds)


if __name__ == '__main__':
    main()
