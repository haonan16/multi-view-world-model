import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def frontend():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/rgb_frontend.py'
    spec = importlib.util.spec_from_file_location('rgb_frontend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rgb_extraction_drops_privileged_arrays_and_forecast_future(tmp_path):
    module = frontend()
    source = tmp_path / 'mixed.npz'
    rgb = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)
    np.savez(source, rgb=rgb, timestamps=[0., 1., 2.], K=np.eye(3), depth_m=np.ones((3, 4, 5)))
    out = tmp_path / 'forecast'
    manifest = module.prepare(source, out, task_mode='forecast')
    with np.load(out / 'rgb.npz', allow_pickle=False) as data:
        assert set(data.files) == {'rgb', 'timestamps'}
        np.testing.assert_array_equal(data['rgb'], rgb[:1])
    assert manifest['observed_until_s'] == 0
    assert manifest['robot_registration']['status'] == 'unresolved'
    assert json.loads((out / 'extraction-audit.json').read_text())['copied_camera_calibration'] is False
    with pytest.raises(FileExistsError):
        module.prepare(source, out)


def test_inverse_uncertainty_rank_preserves_ties_and_invalid_pixels():
    module = frontend()
    uncertainty = np.array([[.1, .5, .5, 2., np.nan]])
    ranks = module.rank_confidence(uncertainty, np.ones_like(uncertainty, bool))
    assert ranks[0, 0] > ranks[0, 1] == ranks[0, 2] > ranks[0, 3] > ranks[0, 4]


def test_multiview_container_requires_explicit_source_view(tmp_path):
    module = frontend()
    source = tmp_path / 'views.npz'
    np.savez(source, rgb=np.zeros((2, 2, 4, 5, 3), np.uint8), timestamps=[0., 1.])
    with pytest.raises(ValueError, match='exactly one'):
        module.prepare(source, tmp_path / 'bad')
    with pytest.raises(ValueError, match='outside'):
        module.prepare(source, tmp_path / 'negative', camera_index=-1)


@pytest.mark.parametrize('forecast', [False, True])
def test_video_extraction_preserves_pixels_fps_and_forecast_cutoff(tmp_path, monkeypatch, forecast):
    module = frontend()
    source = tmp_path / 'recording.mp4'
    source.write_bytes(b'Video decoding API fixture; not a real movie')
    bgr_frames = [np.full((6, 10, 3), [i, 20, 180], dtype=np.uint8) for i in range(3)]

    class Capture:
        def __init__(self):
            self.read_calls = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, _):
            return 10.

        def read(self):
            index = self.read_calls
            self.read_calls += 1
            return (True, bgr_frames[index]) if index < 3 else (False, None)

        def release(self):
            self.released = True

    capture = Capture()
    cv2 = ModuleType('cv2')
    cv2.VideoCapture = lambda _: capture
    cv2.CAP_PROP_FPS = 5
    cv2.COLOR_BGR2RGB = 4
    cv2.cvtColor = lambda frame, _: frame[:, :, ::-1].copy()
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    out = tmp_path / 'video-bundle'
    module.prepare(source, out, task_mode='forecast' if forecast else 'reconstruction')
    with np.load(out / 'rgb.npz') as data:
        n = 1 if forecast else 3
        np.testing.assert_array_equal(data['rgb'], np.stack(bgr_frames[:n])[..., ::-1])
        np.testing.assert_allclose(data['timestamps'], np.arange(n) / 10.)
    assert capture.released is True
    assert capture.read_calls == (1 if forecast else 4)
    audit = json.loads((out / 'extraction-audit.json').read_text())
    assert audit['exported_resolution_hw'] == [6, 10]
    assert audit['container_sha256'] == module.sha256(source)


def test_single_image_forecast_preserves_native_resolution(tmp_path):
    from PIL import Image
    module = frontend()
    source = tmp_path / 'frame.png'
    pixels = np.full((48, 64, 3), [5, 100, 230], dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    out = tmp_path / 'image-bundle'
    manifest = module.prepare(source, out, task_mode='forecast')
    with np.load(out / 'rgb.npz') as data:
        np.testing.assert_array_equal(data['rgb'], pixels[None])
        np.testing.assert_array_equal(data['timestamps'], [0.])
    assert manifest['observed_until_s'] == 0


def make_bundle(module, tmp_path):
    source = tmp_path / 'source.npz'
    np.savez(source, rgb=np.zeros((3, 4, 5, 3), np.uint8), timestamps=[0., 1., 2.])
    bundle = tmp_path / 'bundle'
    module.prepare(source, bundle)
    return bundle


def test_forecast_cannot_hide_future_frames_behind_a_false_time_range(tmp_path):
    module = frontend()
    bundle = make_bundle(module, tmp_path)
    path = bundle / 'input-manifest.json'
    manifest = json.loads(path.read_text())
    manifest.update(task_mode='forecast', observed_until_s=0)
    manifest['artifacts'][0]['time_range_s'] = [0., 0.]
    path.write_text(json.dumps(manifest))
    # Graph validation alone trusts the declared time range; the loader also
    # checks timestamps in the actual NPZ that the model will consume.
    with pytest.raises(ValueError, match='actual RGB timestamps'):
        module.load_source_bundle(bundle)


def test_source_rgb_must_be_the_active_input(tmp_path):
    module = frontend()
    bundle = make_bundle(module, tmp_path)
    path = bundle / 'input-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['artifacts'].append({**manifest['artifacts'][0], 'id': 'other_rgb'})
    manifest['method_inputs'] = ['other_rgb']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='isolated active source_rgb'):
        module.load_source_bundle(bundle)


def install_mutating_unidepth(monkeypatch, invalid_depth=False):
    """Lightweight API double, not a model accuracy/inference test.

    Upstream UniDepth infer -> BatchCamera.from_camera -> crop/resize preserves
    references and mutates the caller's K. Its returned intrinsics remain an
    independent camera-head prediction, even when supplied rays drive depth.
    """
    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getitem__(self, key):
            return Tensor(self.data[key])

        def detach(self):
            return self

        def float(self):
            return Tensor(self.data.astype(np.float32, copy=False))

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def to(self, device):
            assert device == 'cpu'
            return self

        def permute(self, *axes):
            return Tensor(self.data.transpose(axes))

    class Pinhole:
        def __init__(self, K):
            self.K = K

    first_k = np.array([[100., 0., 2.], [0., 120., 2.], [0., 0., 1.]], np.float32)

    class Model:
        def __init__(self):
            self.input_cameras = []

        def to(self, device):
            assert device == 'cpu'
            return self

        def eval(self):
            return self

        def infer(self, image, camera=None):
            self.input_cameras.append(None if camera is None else camera.K.data.copy())
            if camera is not None:
                camera.K.data[:, :2] *= 2.0
                camera.K.data[:, :2, 2] += 3.0
            predicted_k = first_k.copy()
            predicted_k[0, 0] += 10 * (len(self.input_cameras) - 1)
            h, w = image.data.shape[-2:]
            return {
                'depth': Tensor(np.full((1, 1, h, w), np.nan if invalid_depth else 2.)),
                'confidence': Tensor(np.ones((1, 1, h, w))),
                'intrinsics': Tensor(predicted_k[None]),
            }

    model = Model()

    def from_pretrained(path, *, local_files_only):
        assert local_files_only is True and Path(path).is_dir()
        return model

    torch = ModuleType('torch')
    torch.from_numpy = Tensor
    torch.inference_mode = nullcontext
    models = ModuleType('unidepth.models')
    models.UniDepthV2 = SimpleNamespace(from_pretrained=from_pretrained)
    cameras = ModuleType('unidepth.utils.camera')
    cameras.Pinhole = Pinhole
    for name, value in {'torch': torch, 'unidepth': ModuleType('unidepth'),
                        'unidepth.models': models, 'unidepth.utils': ModuleType('unidepth.utils'),
                        'unidepth.utils.camera': cameras}.items():
        monkeypatch.setitem(sys.modules, name, value)
    return model, first_k


def test_fixed_estimated_camera_does_not_accumulate_upstream_resizes(tmp_path, monkeypatch):
    module = frontend()
    bundle = make_bundle(module, tmp_path)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    model, first_k = install_mutating_unidepth(monkeypatch)
    out = tmp_path / 'prediction'
    module.infer_unidepth(bundle, out, checkpoint, 'test-snapshot', device='cpu')
    assert model.input_cameras[0] is None
    for camera in model.input_cameras[1:]:
        np.testing.assert_array_equal(camera, first_k[None])
    with np.load(out / 'geometry.npz') as data:
        np.testing.assert_array_equal(data['K'], first_k[None])
        # Per-frame camera-head estimates are diagnostics, not the lifting K.
        assert data['per_frame_output_intrinsics'][2, 0, 0] != data['K'][0, 0, 0]
        np.testing.assert_array_equal(data['c2w'], np.eye(4)[None])
    report = json.loads((out / 'inference.json').read_text())
    assert report['known_intrinsics_supplied'] is False
    assert report['fixed_camera_assumption'] is True
    assert report['validation']['execution_ready'] is False


def test_invalid_model_output_cannot_be_reported_as_complete(tmp_path, monkeypatch):
    module = frontend()
    bundle = make_bundle(module, tmp_path)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    install_mutating_unidepth(monkeypatch, invalid_depth=True)
    out = tmp_path / 'prediction'
    with pytest.raises(ValueError, match='no usable depth'):
        module.infer_unidepth(bundle, out, checkpoint, 'test-snapshot', device='cpu')
    assert not (out / 'inference.json').exists()


def test_inference_requires_a_local_checkpoint_before_loading_models(tmp_path):
    module = frontend()
    bundle = make_bundle(module, tmp_path)
    with pytest.raises(ValueError, match='existing local checkpoint'):
        module.infer_unidepth(bundle, tmp_path / 'prediction', 'remote/repo-id', 'revision')
    assert not (tmp_path / 'prediction').exists()
