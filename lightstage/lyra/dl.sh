#!/bin/bash
cd /tmp/akiruga_lyra/src/Lyra-2
export PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0
/tmp/akiruga_envs/test2/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Lyra-2.0', allow_patterns=['checkpoints/*'], local_dir='.', max_workers=16)
print('DONE')"
