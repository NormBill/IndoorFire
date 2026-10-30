"""Self-contained interactive visualization for the P8 v2 online risk field."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from indoor_fire_world.p8_v2_config import load_p8_v2_parameters
from indoor_fire_world.scene import load_scene
from indoor_fire_world.visualization.risk_surface import complete_blocked_risk_levels


def _run_length_encode(values: np.ndarray) -> list[list[int]]:
    flattened = np.asarray(values, dtype=np.int8).reshape(-1)
    if flattened.size == 0:
        return []
    runs: list[list[int]] = []
    value = int(flattened[0])
    count = 1
    for item in flattened[1:]:
        item_value = int(item)
        if item_value == value:
            count += 1
        else:
            runs.append([value, count])
            value = item_value
            count = 1
    runs.append([value, count])
    return runs


def render_p8_v2_dynamic_risk_html(
    config_path: str | Path,
    arrays: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Render all 0--480 s online risk frames as a self-contained HTML player."""
    parameters = load_p8_v2_parameters(config_path)
    scene = load_scene(parameters.scene_config)
    registration = scene.map_registration
    if registration is None:
        raise ValueError("P8 v2 dynamic risk visualization requires map registration")
    times = np.asarray(arrays["online_evaluation_times_s"], dtype=np.float64)
    prior = np.asarray(arrays["online_prior_risk_level"], dtype=np.int8)
    posterior = np.asarray(arrays["online_posterior_risk_level"], dtype=np.int8)
    expected_times = np.arange(0.0, 480.0 + 5.0, 5.0)
    if not np.array_equal(times, expected_times):
        raise ValueError("Dynamic risk visualization requires 0--480 s at 5 s intervals")
    if prior.shape != posterior.shape or prior.shape[0] != len(times):
        raise ValueError("Online prior and posterior grids must align with evaluation times")
    frames = posterior.copy()
    frames[0] = prior[0]
    if np.any(frames[0] == -3) or np.any(frames[1:] == -3):
        raise ValueError("Dynamic risk frames contain unavailable cells")
    planning_blocked = frames[0] < 0
    if any(not np.array_equal(frame < 0, planning_blocked) for frame in frames[1:]):
        raise ValueError("Online risk frames must share one planning-blocked mask")
    display_frames = np.stack(
        [complete_blocked_risk_levels(frame) for frame in frames]
    )
    rows, cols = display_frames.shape[1:]
    microwave = scene.semantic.by_id("microwave")
    if microwave.position is None:
        raise ValueError("Microwave source requires a registered position")
    source_x, source_y = registration.metric_to_pixel(*microwave.position)
    payload = {
        "times": [int(value) for value in times],
        "frames": [_run_length_encode(frame) for frame in display_frames],
        "planningBlocked": _run_length_encode(planning_blocked.astype(np.int8)),
        "rows": rows,
        "cols": cols,
        "sourcePixel": [source_x, source_y],
        "mapSize": [registration.source_width_px, registration.source_height_px],
        "floorRoi": list(registration.floor_roi_px),
        "handoffTime": int(parameters.handoff_time_s),
        "levels": [
            {
                "id": level.id,
                "color": level.color,
                "upper": level.upper_bound,
            }
            for level in parameters.exposure_levels
        ],
    }
    image_b64 = base64.b64encode(registration.source_image.read_bytes()).decode("ascii")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>0-480 s online fire-risk evolution</title>
<style>
:root{color-scheme:light;--ink:#172033;--muted:#5c667a;--panel:#fff;--line:#d8deea;--accent:#6f4bd8}
*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:var(--ink);font:16px/1.35 Inter,Segoe UI,Arial,sans-serif}
.player{width:min(1440px,100%);margin:auto;background:var(--panel);padding:18px 22px 14px;box-shadow:0 10px 28px #20304a1c}
.top{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:10px}
h1{font-size:clamp(20px,2.2vw,32px);line-height:1.1;margin:0 0 4px}.subtitle{margin:0;color:var(--muted);font-size:14px}
.readout{text-align:right;white-space:nowrap}.time{font-size:32px;font-weight:800;font-variant-numeric:tabular-nums}.phase{display:block;color:#6f4bd8;font-size:14px;font-weight:700}
.stage{position:relative;border:1px solid var(--line);background:#f8fafc;overflow:hidden}.stage canvas{display:block;width:100%;height:auto}
.controls{display:grid;grid-template-columns:auto minmax(180px,1fr) auto;gap:12px;align-items:center;margin-top:12px}
button,select{border:1px solid #aeb8ca;background:#fff;color:var(--ink);border-radius:7px;padding:7px 12px;font:inherit;font-weight:700}
button{min-width:92px;cursor:pointer}input[type=range]{width:100%;accent-color:var(--accent)}
.ticks{grid-column:2;display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:-9px}.ticks .handoff{color:#8f3b20;font-weight:700}
.legend{display:flex;align-items:center;flex-wrap:wrap;gap:13px 18px;margin-top:10px;font-size:13px}.legend strong{font-size:13px}.key{display:inline-flex;align-items:center;gap:6px}.swatch{width:19px;height:13px;border:1px solid #596273}
.note{margin:8px 0 0;color:var(--muted);font-size:12px}
@media(max-width:720px){.top{align-items:flex-start}.time{font-size:24px}.controls{grid-template-columns:auto 1fr}.controls select{grid-column:1}.ticks{grid-column:2}.legend{gap:9px}}
</style>
</head>
<body>
<main class="player" data-frame-count="97" data-interval-s="5" data-source="online-sequential-posterior" data-display-completion="conservative-nearest-air">
  <header class="top">
    <div><h1>0-480 s online fire-risk evolution</h1><p class="subtitle">Model-predicted planning risk on the registered W1 4F grid</p></div>
    <div class="readout"><span class="time" id="timeLabel">t = 0 s</span><span class="phase" id="phaseLabel">Physics prior only</span></div>
  </header>
  <section class="stage"><canvas id="riskCanvas" width="1713" height="849" aria-label="Animated fire-risk map"></canvas></section>
  <section class="controls">
    <button id="playButton" type="button" aria-label="Pause animation">Pause</button>
    <input id="timeSlider" type="range" min="0" max="96" value="0" step="1" aria-label="Simulation time">
    <select id="speedSelect" aria-label="Playback speed"><option value="320">0.5x</option><option value="180" selected>1x</option><option value="90">2x</option></select>
    <div class="ticks"><span>0 s</span><span class="handoff">300 s handoff</span><span>480 s</span></div>
  </section>
  <section class="legend" id="legend"><strong>Planning-risk level</strong></section>
  <p class="note">t=0 uses the physics prior; t=5-300 s uses sequential posterior updates every 5 s; after 300 s, posterior weights are held fixed. Hatched cells are planning-blocked and use conservative nearest-air color completion for display only. Simulation-only, not certified firefighter safety.</p>
</main>
<img id="mapImage" hidden alt="" src="data:image/png;base64,__MAP_IMAGE__">
<script id="riskData" type="application/json">__RISK_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('riskData').textContent);
const canvas=document.getElementById('riskCanvas'),ctx=canvas.getContext('2d');
const image=document.getElementById('mapImage'),slider=document.getElementById('timeSlider');
const playButton=document.getElementById('playButton'),speedSelect=document.getElementById('speedSelect');
const timeLabel=document.getElementById('timeLabel'),phaseLabel=document.getElementById('phaseLabel');
const params=new URLSearchParams(window.location.search);
let frameIndex=Math.max(0,Math.min(data.frames.length-1,Number(params.get('frame')||0))),timer=null;
const autoplay=params.get('autoplay')!=='0';
const rgba=(hex,a)=>{const n=parseInt(hex.slice(1),16);return `rgba(${n>>16},${(n>>8)&255},${n&255},${a})`};
const decode=runs=>{const cells=new Int8Array(data.rows*data.cols);let p=0;for(const [value,count] of runs){cells.fill(value,p,p+count);p+=count}return cells};
const decoded=data.frames.map(decode);
const planningBlocked=decode(data.planningBlocked);
const hatchCanvas=document.createElement('canvas');hatchCanvas.width=14;hatchCanvas.height=14;const hatchContext=hatchCanvas.getContext('2d');hatchContext.strokeStyle='rgba(30,41,59,.18)';hatchContext.lineWidth=.8;hatchContext.beginPath();hatchContext.moveTo(-2,14);hatchContext.lineTo(14,-2);hatchContext.moveTo(5,16);hatchContext.lineTo(16,5);hatchContext.stroke();const blockedPattern=ctx.createPattern(hatchCanvas,'repeat');
function phaseFor(t){if(t===0)return ['Physics prior only','#315f9b'];if(t<=data.handoffTime)return ['Sequential posterior update','#6f4bd8'];return ['Forecast with t=300 posterior fixed','#9a4d13']}
function drawSource(){const [x,y]=data.sourcePixel;ctx.save();ctx.lineWidth=7;ctx.strokeStyle='#fff';ctx.fillStyle='#d7191c';ctx.beginPath();ctx.arc(x,y,17,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#111827';ctx.font='700 19px Segoe UI,Arial';ctx.fillText('Microwave source',x+25,y+7);ctx.restore()}
function draw(index){frameIndex=index;const t=data.times[index],cells=decoded[index];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,data.mapSize[0],data.mapSize[1]);const [left,top,width,height]=data.floorRoi,cw=width/data.cols,ch=height/data.rows;for(let row=0;row<data.rows;row++){const y=top+height-(row+1)*ch;for(let col=0;col<data.cols;col++){const cellIndex=row*data.cols+col,level=cells[cellIndex],x=left+col*cw;ctx.fillStyle=rgba(data.levels[level].color,.82);ctx.fillRect(x,y,cw+.25,ch+.25);if(planningBlocked[cellIndex]){ctx.fillStyle=blockedPattern;ctx.fillRect(x,y,cw+.25,ch+.25)}}}drawSource();slider.value=String(index);timeLabel.textContent=`t = ${t} s`;const [label,color]=phaseFor(t);phaseLabel.textContent=label;phaseLabel.style.color=color}
function stop(){if(timer!==null){clearInterval(timer);timer=null}playButton.textContent='Play';playButton.setAttribute('aria-label','Play animation')}
function start(){stop();playButton.textContent='Pause';playButton.setAttribute('aria-label','Pause animation');timer=setInterval(()=>draw((frameIndex+1)%data.frames.length),Number(speedSelect.value))}
playButton.addEventListener('click',()=>timer===null?start():stop());slider.addEventListener('input',()=>{stop();draw(Number(slider.value))});speedSelect.addEventListener('change',()=>{if(timer!==null)start()});
document.addEventListener('keydown',event=>{if(event.code==='Space'){event.preventDefault();timer===null?start():stop()}else if(event.code==='ArrowRight'){stop();draw(Math.min(frameIndex+1,data.frames.length-1))}else if(event.code==='ArrowLeft'){stop();draw(Math.max(frameIndex-1,0))}});
let lower=0;for(const level of data.levels){const item=document.createElement('span');item.className='key';const upper=level.upper===null?`> ${lower.toFixed(2)}`:`${lower.toFixed(2)}-${Number(level.upper).toFixed(2)}`;item.innerHTML=`<span class="swatch" style="background:${level.color}"></span>${level.id} (${upper})`;document.getElementById('legend').appendChild(item);if(level.upper!==null)lower=Number(level.upper)}const blockedKey=document.createElement('span');blockedKey.className='key';blockedKey.innerHTML='<span class="swatch" style="background:repeating-linear-gradient(135deg,transparent 0 4px,#475569 4px 5px)"></span>planning-blocked';document.getElementById('legend').appendChild(blockedKey);
const initialize=()=>{draw(frameIndex);if(autoplay)setTimeout(start,500);else stop()};image.addEventListener('load',initialize);if(image.complete)initialize()
</script>
</body>
</html>
'''
    html = template.replace("__MAP_IMAGE__", image_b64).replace(
        "__RISK_DATA__", json.dumps(payload, separators=(",", ":"))
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output.resolve()
