"""Build a lossless HTML viewer for the v2 camera-conditioning eval (infer_multiview.py output).

Every frame is written as PNG (no lossy video compression): for each scene and frame one PNG
per row (GT, then each model), the views side by side at native resolution. index.html plays
them with a frame slider; data.js carries metrics + task text so it also works over file://.

An eval dir is either one infer_multiview.py output (metrics.json at its root) or a dir of
per-model outputs <dir>/<model>/ (eval_v2_worker.sh), which are merged here.

Usage: python scripts/rh20t/build_v2_eval_viewer.py --out VIEWER_DIR \
           --runs "no guidance=EVAL/noguide,text 2 + hist 1=EVAL/guide_t2_h1"
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

TASKS = "/n/lab_storage/ydu_lab/Lab/akiruga/rh20t_meta/task_descriptions.json"


def load_report(d):
    """metrics.json of one eval dir, or merged over its per-model subdirs. Also returns
    {row: dir holding <scene>/<row>.npy}."""
    if (d / "metrics.json").exists():
        rep = json.loads((d / "metrics.json").read_text())
        return rep, {r: d for r in ["gt"] + [f"pred_{m}" for m in rep["ckpt"]]}
    parts = sorted(p for p in d.iterdir() if (p / "metrics.json").exists())
    order = ["concat", "cat4d", "prope", "gta", "camtoken"]
    parts.sort(key=lambda p: order.index(p.name) if p.name in order else len(order))
    reps = [json.loads((p / "metrics.json").read_text()) for p in parts]
    rep = {"scenes": reps[0]["scenes"], "ckpt": {}, "mean": {}, "guidance": reps[0].get("guidance", {}),
           "per_scene": {s: {} for s in reps[0]["scenes"]}}
    src = {"gt": parts[0]}
    for p, r in zip(parts, reps):
        assert r["scenes"] == rep["scenes"], f"{p}: different scenes"
        rep["ckpt"].update(r["ckpt"]); rep["mean"].update(r["mean"])
        for s in rep["scenes"]:
            rep["per_scene"][s].update(r["per_scene"][s])
        for m in r["ckpt"]:
            src[f"pred_{m}"] = p
    return rep, src


def write_rows(job):
    srcs, scene, dst, rows = job
    dst.mkdir(parents=True, exist_ok=True)
    for r in rows:
        a = np.load(srcs[r] / scene / f"{r}.npy")                     # V, T, H, W, 3 uint8
        for t in range(a.shape[1]):
            Image.fromarray(np.concatenate(list(a[:, t]), axis=1)).save(
                dst / f"{r}_{t:02d}.png", compress_level=1)
    return a.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", required=True, help="comma list of label=eval_dir")
    args = ap.parse_args()
    out = Path(args.out)
    tasks = json.load(open(TASKS))
    data = {"runs": []}
    for ri, (label, d) in enumerate(r.split("=", 1) for r in args.runs.split(",")):
        d = Path(d)
        rep, srcs = load_report(d)
        models = list(rep["ckpt"])
        rows = ["gt"] + [f"pred_{m}" for m in models]
        jobs = [(srcs, s, out / f"run{ri}" / s, rows) for s in rep["scenes"]]
        with ThreadPoolExecutor(16) as ex:
            shapes = list(ex.map(write_rows, jobs))
        V, T, H, W, _ = shapes[0]
        scenes = []
        for s in rep["scenes"]:
            ep = s.split("_", 1)[1]
            task = ep.split("_user")[0]
            scenes.append({"name": s, "episode": ep, "task": task,
                           "text": tasks.get(task, ""), "metrics": rep["per_scene"][s]})
        data["runs"].append({
            "label": label, "dir": f"run{ri}", "models": models, "rows": rows,
            "V": int(V), "T": int(T), "H": int(H), "W": int(W),
            "steps": {m: rep["ckpt"][m]["global_step"] for m in models},
            "guidance": rep.get("guidance", {}), "mean": rep["mean"], "scenes": scenes})
        print(f"{label}: {len(scenes)} scenes, {len(rows)} rows, {T} frames of {V}x{H}x{W}", flush=True)
    (out / "data.js").write_text("window.EVAL = " + json.dumps(data) + ";\n")
    (out / "index.html").write_text(HTML)
    print(out / "index.html")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera Conditioning Eval</title>
<style>
:root{--bg:#f7f7f5;--fg:#1c1c1c;--mut:#6b6b6b;--card:#fff;--line:#e2e2de;--acc:#2d6cdf;--best:#e7f0ff}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e8;--mut:#9a9a9a;--card:#1d1d1d;--line:#333;--acc:#7aa7ff;--best:#1f2c45}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 4px}
.mut{color:var(--mut)}main{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 70px)}
nav{border-right:1px solid var(--line);overflow:auto;max-height:calc(100vh - 70px);position:sticky;top:0}
nav button{display:block;width:100%;text-align:left;padding:8px 14px;border:0;border-bottom:1px solid var(--line);background:none;color:inherit;cursor:pointer;font:inherit}
nav button.on{background:var(--best)}nav .sub{font-size:12px;color:var(--mut)}
section{padding:16px 20px;overflow:auto}
table{border-collapse:collapse;margin:8px 0 16px;background:var(--card)}th,td{border:1px solid var(--line);padding:4px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}td.best{background:var(--best);font-weight:600}
.ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0}
select,button.b{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:inherit;cursor:pointer}
input[type=range]{width:260px}
.grid{display:grid;grid-template-columns:auto auto;gap:4px 10px;align-items:center}
.grid .lab{font-weight:600;text-align:right;white-space:nowrap}
.grid img{display:block}.pix img{image-rendering:pixelated}
.prompt{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0;font-size:15px}
.load{color:var(--mut)}
@media (max-width:800px){main{grid-template-columns:1fr}nav{max-height:240px;position:static}}
</style></head><body>
<header><h1>Multiview camera-conditioning ablation: held-out RH20T</h1>
<div class="mut" id="sub"></div></header>
<main><nav id="nav"></nav><section>
<div class="ctl"><label>Sampling <select id="run"></select></label>
<label>Scale <select id="scale"><option>1</option><option selected>2</option><option>3</option><option>4</option></select></label>
<label><input type="checkbox" id="pix" checked> nearest-neighbour</label></div>
<h3 style="margin:8px 0 0">Mean over scenes</h3><div id="mean"></div>
<h3 style="margin:8px 0 0" id="stitle"></h3><div class="prompt" id="prompt"></div><div id="smet"></div>
<div class="ctl"><button class="b" id="play">Pause</button>
<input type="range" id="fr" min="0" value="0"><span id="frl"></span>
<label>fps <select id="fps"><option>1</option><option>2</option><option selected>3</option><option>6</option></select></label>
<span class="load" id="load"></span></div>
<div class="grid" id="grid"></div>
</section></main>
<script src="data.js"></script>
<script>
const E=window.EVAL,$=id=>document.getElementById(id);
let ri=0,si=0,t=0,playing=true,timer=null,imgs=[];
const fmt=(v,k)=>k==="psnr"?v.toFixed(2):v.toFixed(3);
function table(met,models){
  const cols=[...models,"copy0"],ks=[["psnr","PSNR ↑",1],["ssim","SSIM ↑",1],["lpips","LPIPS ↓",-1]];
  let h="<table><tr><th>metric</th>"+cols.map(c=>`<th>${c==="copy0"?"copy frame 0":c}</th>`).join("")+"</tr>";
  for(const [k,n,s] of ks){const vals=models.map(m=>met[m][k]);const b=s>0?Math.max(...vals):Math.min(...vals);
    h+=`<tr><td>${n}</td>`+cols.map(c=>`<td class="${c!=="copy0"&&met[c][k]===b?"best":""}">${fmt(met[c][k],k)}</td>`).join("")+"</tr>";}
  return h+"</table>";}
function R(){return E.runs[ri]}
function nav(){const r=R();$("nav").innerHTML=r.scenes.map((s,i)=>`<button class="${i===si?"on":""}" data-i="${i}">${i+1}. ${s.task}<div class="sub">${s.text||""}</div></button>`).join("");
  $("nav").querySelectorAll("button").forEach(b=>b.onclick=()=>{si=+b.dataset.i;scene()});}
function scene(){
  const r=R(),s=r.scenes[si];nav();
  $("stitle").textContent=`Scene ${si+1}/${r.scenes.length}: ${s.episode}`;
  $("prompt").textContent="Task text: "+(s.text||"(none)");
  $("smet").innerHTML=table(s.metrics,r.models);
  $("fr").max=r.T-1;const sc=+$("scale").value;
  const names={gt:"ground truth"};r.models.forEach(m=>names["pred_"+m]=m+` (step ${r.steps[m]})`);
  $("grid").innerHTML="";imgs=[];
  const cache=r.rows.map(row=>Array.from({length:r.T},(_,k)=>{const im=new Image();im.src=`${r.dir}/${s.name}/${row}_${String(k).padStart(2,"0")}.png`;return im}));
  let done=0,total=r.rows.length*r.T;$("load").textContent=`loading 0/${total}`;
  cache.flat().forEach(im=>{const f=()=>{done++;$("load").textContent=done<total?`loading ${done}/${total}`:""};if(im.complete)f();else{im.onload=f;im.onerror=f}});
  r.rows.forEach((row,j)=>{const l=document.createElement("div");l.className="lab";l.textContent=names[row];
    const im=document.createElement("img");im.width=r.V*r.W*sc;im.height=r.H*sc;im.src=cache[j][t].src;
    $("grid").append(l,im);imgs.push(im);});
  window._cache=cache;show();}
function show(){const r=R();t=Math.min(t,r.T-1);imgs.forEach((im,j)=>im.src=window._cache[j][t].src);$("fr").value=t;$("frl").textContent=`frame ${t}/${r.T-1}${t===0?" (history)":""}`;}
function loop(){clearInterval(timer);if(playing)timer=setInterval(()=>{t=(t+1)%R().T;show()},1000/+$("fps").value);}
function init(){
  $("run").innerHTML=E.runs.map((r,i)=>`<option value="${i}">${r.label}</option>`).join("");
  $("run").onchange=()=>{ri=+$("run").value;head();scene()};
  $("scale").onchange=scene;$("pix").onchange=()=>$("grid").classList.toggle("pix",$("pix").checked);$("grid").classList.add("pix");
  $("fr").oninput=()=>{t=+$("fr").value;show()};$("fps").onchange=loop;
  $("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";loop()};
  document.addEventListener("keydown",e=>{if(e.key==="ArrowRight"){t=(t+1)%R().T;show()}else if(e.key==="ArrowLeft"){t=(t-1+R().T)%R().T;show()}
    else if(e.key==="ArrowDown"){si=Math.min(si+1,R().scenes.length-1);scene()}else if(e.key==="ArrowUp"){si=Math.max(si-1,0);scene()}});
  head();scene();loop();}
function head(){const r=R();$("mean").innerHTML=table(r.mean,r.models);
  $("sub").textContent=`${r.scenes.length} held-out scenes · 1-frame history (frame 0 of all ${r.V} views) + task text · ${r.T} frames per view at ${r.W}×${r.H} · lossless PNG frames · ←/→ frame, ↑/↓ scene`;}
init();
</script></body></html>
"""

if __name__ == "__main__":
    main()
