"""Self-contained P9 intervention/no-intervention risk comparison player."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from indoor_fire_world.p9_config import load_p9_parameters
from indoor_fire_world.scene import load_scene
from indoor_fire_world.visualization.risk_surface import (
    complete_blocked_exposure,
    exposure_risk_levels,
)


def _run_length_encode(values: np.ndarray) -> list[list[int]]:
    flattened = np.asarray(values).reshape(-1)
    if flattened.size == 0:
        return []
    runs: list[list[int]] = []
    value = int(flattened[0])
    count = 1
    for item in flattened[1:]:
        current = int(item)
        if current == value:
            count += 1
        else:
            runs.append([value, count])
            value, count = current, 1
    runs.append([value, count])
    return runs


def render_p9_dynamic_risk_comparison_html(
    config_path: str | Path,
    summary: dict,
    arrays: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Render 0--480 s risk with switchable post-handoff action branches."""
    p9 = load_p9_parameters(config_path)
    p8 = p9.p8_config
    from indoor_fire_world.p8_v2_config import load_p8_v2_parameters

    p8_parameters = load_p8_v2_parameters(p8)
    scene = load_scene(p8_parameters.scene_config)
    registration = scene.map_registration
    if registration is None:
        raise ValueError("P9 dynamic visualization requires map registration")
    times = np.asarray(arrays["dynamic_evaluation_times_s"], dtype=np.float64)
    expected = np.arange(0.0, 480.0 + 5.0, 5.0)
    if not np.array_equal(times, expected):
        raise ValueError("P9 dynamic comparison requires 0--480 s at 5 s intervals")
    no_action = np.asarray(arrays["dynamic_no_intervention_exposure"])
    intervention = np.asarray(arrays["dynamic_intervention_exposure"])
    if no_action.shape != intervention.shape or no_action.shape[0] != len(times):
        raise ValueError("P9 comparison branches must have aligned grids")
    blocked = np.asarray(scene.occupancy.cells) != 0
    bounds = tuple(
        float(level.upper_bound)
        for level in p8_parameters.exposure_levels
        if level.upper_bound is not None
    )

    def display_levels(sequence: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                exposure_risk_levels(
                    complete_blocked_exposure(frame, blocked), bounds
                )
                for frame in sequence
            ]
        )

    no_levels = display_levels(no_action)
    intervention_levels = display_levels(intervention)
    if not np.array_equal(no_levels[:61], intervention_levels[:61]):
        raise ValueError("P9 branches must remain identical through t=300 s")
    microwave = scene.semantic.by_id("microwave")
    if microwave.position is None:
        raise ValueError("Microwave requires a registered position")
    source_x, source_y = registration.metric_to_pixel(*microwave.position)
    comparison_action = summary["dynamic_risk_comparison"]["comparison_action"]
    payload = {
        "times": [int(value) for value in times],
        "noIntervention": [_run_length_encode(frame) for frame in no_levels],
        "intervention": [_run_length_encode(frame) for frame in intervention_levels],
        "planningBlocked": _run_length_encode(blocked.astype(np.int8)),
        "rows": int(no_levels.shape[1]),
        "cols": int(no_levels.shape[2]),
        "sourcePixel": [source_x, source_y],
        "mapSize": [registration.source_width_px, registration.source_height_px],
        "floorRoi": list(registration.floor_roi_px),
        "handoffTime": int(p9.firefighter_availability_time_s),
        "comparisonAction": comparison_action,
        "levels": [
            {"id": level.id, "color": level.color, "upper": level.upper_bound}
            for level in p8_parameters.exposure_levels
        ],
    }
    image_b64 = base64.b64encode(registration.source_image.read_bytes()).decode(
        "ascii"
    )
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P9 intervention risk comparison</title><style>
:root{--ink:#172033;--muted:#5c667a;--line:#d8deea;--accent:#6f4bd8}*{box-sizing:border-box}
body{margin:0;background:#eef2f7;color:var(--ink);font:15px/1.35 Inter,Segoe UI,Arial,sans-serif}.player{width:min(1540px,100%);margin:auto;background:#fff;padding:18px 22px;box-shadow:0 10px 28px #20304a1c}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:18px}h1{margin:0;font-size:clamp(21px,2.2vw,32px)}.subtitle,.note{color:var(--muted)}.readout{text-align:right;white-space:nowrap}.time{font-size:30px;font-weight:800}.phase{display:block;color:var(--accent);font-weight:700}
.branch-controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.branch-controls button,.controls button,select{border:1px solid #aeb8ca;background:#fff;border-radius:7px;padding:7px 12px;font:inherit;font-weight:700;cursor:pointer}.branch-controls button.active{background:#283593;color:#fff;border-color:#283593}.branch-controls button:disabled{opacity:.45;cursor:not-allowed}
.maps{display:grid;grid-template-columns:1fr;gap:12px}.maps.compare{grid-template-columns:1fr 1fr}.panel{border:1px solid var(--line);background:#f8fafc}.panel[hidden]{display:none}.panel-label{padding:7px 10px;font-weight:800}.panel canvas{display:block;width:100%;height:auto}
.controls{display:grid;grid-template-columns:auto minmax(220px,1fr) auto;gap:12px;align-items:center;margin-top:12px}.controls input{width:100%;accent-color:var(--accent)}.ticks{grid-column:2;display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:-8px}.ticks .handoff{color:#9a4d13;font-weight:800}
.legend{display:flex;gap:12px 18px;flex-wrap:wrap;align-items:center;margin-top:10px;font-size:13px}.key{display:inline-flex;gap:6px;align-items:center}.swatch{width:19px;height:13px;border:1px solid #596273}.difference{font-weight:700;margin:8px 0 0}.note{font-size:12px;margin:6px 0 0}
@media(max-width:850px){.maps.compare{grid-template-columns:1fr}.top{align-items:flex-start}.controls{grid-template-columns:auto 1fr}.controls select{grid-column:1}.ticks{grid-column:2}}
</style></head><body>
<main class="player" data-frame-count="97" data-interval-s="5" data-handoff-s="300" data-source="p9-action-conditioned-posterior" data-comparison="no-intervention-vs-intervention">
<header class="top"><div><h1>0–480 s action-conditioned risk</h1><p class="subtitle">The branches are identical through 300 s; intervention changes only the post-handoff forecast.</p></div><div class="readout"><span class="time" id="timeLabel">t = 0 s</span><span class="phase" id="phaseLabel">Pre-entry posterior</span></div></header>
<nav class="branch-controls" aria-label="Risk branch display"><button data-mode="no" type="button">No intervention</button><button data-mode="intervention" type="button">Intervention</button><button data-mode="compare" class="active" type="button">Compare side by side</button></nav>
<section class="maps" id="maps"><article class="panel" id="leftPanel"><div class="panel-label" id="leftLabel">No intervention</div><canvas id="leftCanvas" width="1713" height="849"></canvas></article><article class="panel" id="rightPanel"><div class="panel-label">Intervention: __ACTION_LABEL__</div><canvas id="rightCanvas" width="1713" height="849"></canvas></article></section>
<section class="controls"><button id="playButton" type="button">Pause</button><input id="timeSlider" type="range" min="0" max="96" value="0" step="1"><select id="speedSelect"><option value="320">0.5x</option><option value="180" selected>1x</option><option value="90">2x</option></select><div class="ticks"><span>0 s</span><span class="handoff">300 s handoff</span><span>480 s</span></div></section>
<p class="difference" id="differenceLabel">Shared pre-entry posterior: no action difference yet.</p><section class="legend" id="legend"><strong>Risk level</strong></section>
<p class="note">Every registered grid cell receives one risk level. After t=300 s, use the buttons to view either branch or compare both. The t=420 s observation update is evaluated against the matching action-conditioned model. Hatched cells are planning-blocked but remain visually estimated. Simulation-only planning risk, not certified firefighter safety.</p>
</main><img id="mapImage" hidden alt="" src="data:image/png;base64,__MAP_IMAGE__"><script id="riskData" type="application/json">__RISK_DATA__</script><script>
const data=JSON.parse(document.getElementById('riskData').textContent),image=document.getElementById('mapImage');const slider=document.getElementById('timeSlider'),playButton=document.getElementById('playButton'),speedSelect=document.getElementById('speedSelect'),timeLabel=document.getElementById('timeLabel'),phaseLabel=document.getElementById('phaseLabel'),maps=document.getElementById('maps'),rightPanel=document.getElementById('rightPanel'),leftLabel=document.getElementById('leftLabel'),differenceLabel=document.getElementById('differenceLabel');
const canvases={left:document.getElementById('leftCanvas'),right:document.getElementById('rightCanvas')};const contexts={left:canvases.left.getContext('2d'),right:canvases.right.getContext('2d')};const params=new URLSearchParams(window.location.search);let frameIndex=Math.max(0,Math.min(data.times.length-1,Number(params.get('frame')||0))),mode=['no','intervention','compare'].includes(params.get('mode'))?params.get('mode'):'compare',timer=null;const autoplay=params.get('autoplay')!=='0';
const decode=runs=>{const cells=new Uint8Array(data.rows*data.cols);let p=0;for(const [v,n] of runs){cells.fill(v,p,p+n);p+=n}return cells};const noFrames=data.noIntervention.map(decode),interventionFrames=data.intervention.map(decode),blocked=decode(data.planningBlocked);const rgba=(hex,a)=>{const n=parseInt(hex.slice(1),16);return `rgba(${n>>16},${(n>>8)&255},${n&255},${a})`};
function drawSource(ctx){const [x,y]=data.sourcePixel;ctx.save();ctx.lineWidth=7;ctx.strokeStyle='#fff';ctx.fillStyle='#d7191c';ctx.beginPath();ctx.arc(x,y,17,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#111827';ctx.font='700 19px Segoe UI,Arial';ctx.fillText('Microwave source',x+25,y+7);ctx.restore()}
function drawMap(which,cells){const ctx=contexts[which],canvas=canvases[which];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,...data.mapSize);const [left,top,width,height]=data.floorRoi,cw=width/data.cols,ch=height/data.rows;for(let r=0;r<data.rows;r++){const y=top+height-(r+1)*ch;for(let c=0;c<data.cols;c++){const i=r*data.cols+c,x=left+c*cw;ctx.fillStyle=rgba(data.levels[cells[i]].color,.82);ctx.fillRect(x,y,cw+.25,ch+.25);if(blocked[i]){ctx.strokeStyle='rgba(30,41,59,.18)';ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(x,y+ch);ctx.lineTo(x+cw,y);ctx.stroke()}}}drawSource(ctx)}
function applyMode(){const post=data.times[frameIndex]>data.handoffTime;document.querySelectorAll('.branch-controls button').forEach(b=>{b.disabled=!post&&b.dataset.mode!=='compare';b.classList.toggle('active',b.dataset.mode===mode)});maps.classList.toggle('compare',post&&mode==='compare');rightPanel.hidden=!(post&&mode==='compare');leftLabel.textContent=post&&mode==='intervention'?`Intervention: ${data.comparisonAction.replaceAll('_',' ')}`:'No intervention';draw(frameIndex)}
function draw(i){frameIndex=i;const t=data.times[i],post=t>data.handoffTime;const leftCells=post&&mode==='intervention'?interventionFrames[i]:noFrames[i];drawMap('left',leftCells);if(post&&mode==='compare')drawMap('right',interventionFrames[i]);slider.value=String(i);timeLabel.textContent=`t = ${t} s`;phaseLabel.textContent=t<=data.handoffTime?'Shared pre-entry posterior':'Post-handoff action comparison';let changed=0;for(let k=0;k<noFrames[i].length;k++)if(noFrames[i][k]!==interventionFrames[i][k])changed++;differenceLabel.textContent=post?`${changed} of ${noFrames[i].length} grid cells change risk level under the intervention branch.`:'Shared pre-entry posterior: no action difference yet.'}
function stop(){if(timer!==null){clearInterval(timer);timer=null}playButton.textContent='Play'}function start(){stop();playButton.textContent='Pause';timer=setInterval(()=>{const next=(frameIndex+1)%data.times.length;draw(next);if(data.times[next]===305&&mode==='compare')applyMode()},Number(speedSelect.value))}
document.querySelectorAll('.branch-controls button').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;applyMode()}));playButton.addEventListener('click',()=>timer===null?start():stop());slider.addEventListener('input',()=>{stop();draw(Number(slider.value));applyMode()});speedSelect.addEventListener('change',()=>{if(timer!==null)start()});
let lower=0;for(const level of data.levels){const item=document.createElement('span');item.className='key';const upper=level.upper===null?`> ${lower.toFixed(2)}`:`${lower.toFixed(2)}–${Number(level.upper).toFixed(2)}`;item.innerHTML=`<span class="swatch" style="background:${level.color}"></span>${level.id} (${upper})`;document.getElementById('legend').appendChild(item);if(level.upper!==null)lower=Number(level.upper)}
const init=()=>{applyMode();if(autoplay)setTimeout(start,500);else stop()};image.addEventListener('load',init);if(image.complete)init();
</script></body></html>'''
    html = (
        template.replace("__MAP_IMAGE__", image_b64)
        .replace("__RISK_DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__ACTION_LABEL__", comparison_action.replace("_", " "))
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output.resolve()
