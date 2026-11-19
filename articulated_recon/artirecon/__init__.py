from importlib.util import find_spec

# Metadata/protocol tools can run without the fitting environment. When torch
# is installed, preserve the patch ordering required by the GPU fitters.
if find_spec("torch") is not None:
    from . import p3d_compat  # noqa: F401
