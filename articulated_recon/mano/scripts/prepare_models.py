"""Read the user's official MANO archive; keep licensed assets local.

Chumpy is needed only here. Its legacy imports require compatibility aliases on
Python 3.12 / NumPy 1.26. The fitter loads numeric NPZ arrays without pickle.
"""
import argparse
import hashlib
import inspect
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("assets/mano"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Compatibility only inside this short-lived conversion process.
    inspect.getargspec = inspect.getfullargspec
    for name, value in dict(bool=bool, int=int, float=float, complex=complex,
                            object=object, str=str, unicode=str).items():
        if name not in np.__dict__:
            setattr(np, name, value)
    import chumpy  # noqa: F401

    hashes = {}
    keys = ["v_template", "shapedirs", "posedirs", "J_regressor", "weights",
            "kintree_table", "f", "hands_components", "hands_mean"]
    with zipfile.ZipFile(args.archive) as archive:
        for side in ("LEFT", "RIGHT"):
            name = f"mano_v1_2/models/MANO_{side}.pkl"
            raw = archive.read(name)
            # Trusted user-supplied official model archive, never arbitrary data.
            model = pickle.loads(raw, encoding="latin1")
            arrays = {}
            for key in keys:
                value = model[key]
                if hasattr(value, "toarray"):
                    value = value.toarray()
                arrays[key] = np.asarray(value)
                if arrays[key].dtype.hasobject:
                    raise ValueError(f"Unexpected object array: {key}")
            np.savez_compressed(args.out / f"MANO_{side}.npz", **arrays)
            hashes[name] = hashlib.sha256(raw).hexdigest()
            print(side, "vertices", arrays["v_template"].shape,
                  "extent (native units)", np.ptp(arrays["v_template"], axis=0))
        (args.out / "LICENSE.txt").write_bytes(archive.read("mano_v1_2/LICENSE.txt"))
    (args.out / "provenance.json").write_text(json.dumps({
        "source_archive": str(args.archive.resolve()),
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "source_models_sha256": hashes,
        "conversion": "Numeric arrays only; no shape or pose changes",
    }, indent=2))


if __name__ == "__main__":
    main()
