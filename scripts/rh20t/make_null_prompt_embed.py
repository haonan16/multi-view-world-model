"""Save the UMT5-XXL embedding of the empty prompt "" for Wan2.1 (RH20T has no captions).
Same encoder + tokenizer path as WanTextToVideoRayDepthMoTConcat5B.encode_text.
Usage: HF_HOME=... python scripts/rh20t/make_null_prompt_embed.py <out.pt>"""
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.wan.modules.t5 import umt5_xxl  # noqa: E402
from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer  # noqa: E402

enc = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16,
               device=torch.device("cpu")).eval().requires_grad_(False)
enc.load_state_dict(torch.load(hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
                               map_location="cpu", weights_only=True))
tok = HuggingfaceTokenizer(name="google/umt5-xxl", seq_len=512, clean="whitespace")
ids, mask = tok([""], return_mask=True, add_special_tokens=True)
with torch.no_grad():
    ctx = enc(ids, mask)[0][: int(mask.gt(0).sum())]
torch.save(ctx.clone(), sys.argv[1])
print("saved", tuple(ctx.shape), ctx.dtype, "->", sys.argv[1])
