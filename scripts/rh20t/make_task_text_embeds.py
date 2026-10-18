"""Cache UMT5-XXL embeddings (Wan2.1 text encoder) of every RH20T task description so
training never runs the text encoder. Same encoder + tokenizer as make_null_prompt_embed.py.

Input:  {"task_0001": "Press the button from top to bottom", ...}  (rh20t task_description.json,
        English field, flattened)
Output: torch file {"task_0001": {"text": str, "embed": bf16 [L_i, 4096]}, ..., "": null prompt}
Usage: HF_HOME=... python scripts/rh20t/make_task_text_embeds.py <descriptions.json> <out.pt>
"""
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.wan.modules.t5 import umt5_xxl  # noqa: E402
from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer  # noqa: E402

desc = json.loads(Path(sys.argv[1]).read_text())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16,
               device=dev).eval().requires_grad_(False)
enc.load_state_dict(torch.load(hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
                               map_location="cpu", weights_only=True))
tok = HuggingfaceTokenizer(name="google/umt5-xxl", seq_len=512, clean="whitespace")
out = {}
for key, text in list(desc.items()) + [("", "")]:
    ids, mask = tok([text], return_mask=True, add_special_tokens=True)
    with torch.no_grad():
        ctx = enc(ids.to(dev), mask.to(dev))[0][: int(mask.gt(0).sum())]
    out[key] = {"text": text, "embed": ctx.to(torch.bfloat16).cpu().clone()}
torch.save(out, sys.argv[2])
print(f"saved {len(out)} prompt embeddings -> {sys.argv[2]}; e.g. task_0001 {tuple(out['task_0001']['embed'].shape)}")
