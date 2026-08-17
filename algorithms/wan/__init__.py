# Pruned for the clean training branch: only the two model classes that trained
# Model F (5B) and the 1.3B R1 ablations. See metafiles/APPLIED_FILES.txt / TRAINING_CODE_TRACE.md.
from .wan_t2v_ray_depth_mot_concat_5b import WanTextToVideoRayDepthMoTConcat5B
from .wan_t2v_ray_depth_mot_concat_rebuttal import WanTextToVideoRayDepthMoTConcatRebuttal

__all__ = [
    "WanTextToVideoRayDepthMoTConcat5B",
    "WanTextToVideoRayDepthMoTConcatRebuttal",
]
