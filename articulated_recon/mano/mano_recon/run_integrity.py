"""File identity checks for derived outputs reused by a verified run."""
import hashlib
from pathlib import Path


def artifact_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def output_records(paths):
    return {name: artifact_record(path) for name, path in paths.items()}


def verify_cached_outputs(audit, paths):
    """A matching input manifest alone cannot authenticate derived cache bytes."""
    records = audit.get('output_artifacts', {})
    for name, path in paths.items():
        previous = records.get(name)
        if not isinstance(previous, dict):
            raise ValueError(f'Verified cached output lacks a saved hash: {name}; use a fresh run directory')
        if not Path(path).is_file():
            raise ValueError(f'Verified cached output is missing: {name}')
        if previous != artifact_record(path):
            raise ValueError(f'Verified cached output changed: {name}; refusing reuse')
    return True
