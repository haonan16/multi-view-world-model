import pytest

from mano_recon.run_integrity import output_records, verify_cached_outputs


def test_cached_fit_and_actual_exported_hand_must_both_match(tmp_path):
    paths = {'fit': tmp_path / 'fit.npz', 'frozen_hand': tmp_path / 'frozen-hand.npz'}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    audit = {'output_artifacts': output_records(paths)}
    assert verify_cached_outputs(audit, paths)
    paths['frozen_hand'].write_bytes(b'unrelated old hand')
    with pytest.raises(ValueError, match='changed: frozen_hand'):
        verify_cached_outputs(audit, paths)


def test_verified_cache_requires_original_hashes_and_present_outputs(tmp_path):
    path = tmp_path / 'fit.npz'
    path.write_bytes(b'fit')
    with pytest.raises(ValueError, match='lacks a saved hash'):
        verify_cached_outputs({}, {'fit': path})
    audit = {'output_artifacts': output_records({'fit': path})}
    path.unlink()
    with pytest.raises(ValueError, match='missing: fit'):
        verify_cached_outputs(audit, {'fit': path})
