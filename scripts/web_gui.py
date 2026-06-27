import asyncio
import json
import pickle
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

from utils.colorspace.color_space import yuv_to_rgb_linear_np

CKPT_DIR = Path("runs/production")
TEMP_DIR = Path(__file__).resolve().parent.parent / ".cache"
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ArtRT")
jobs: dict[str, dict] = {}


# ── helpers ──

def srgb_gamma(x: np.ndarray) -> np.ndarray:
    return np.clip(x ** (1 / 2.2), 0, 1)


def scan_checkpoints():
    if not CKPT_DIR.exists():
        return []
    files = sorted(CKPT_DIR.rglob("*.pkl"), reverse=True)
    rels = sorted(set(f.relative_to(CKPT_DIR).as_posix() for f in files))
    # root items first, then subdir items
    root = [r for r in rels if "/" not in r]
    sub = [r for r in rels if "/" in r]
    return root + sorted(sub, reverse=True)


def load_model(ckpt_name):
    ckpt_path = CKPT_DIR / ckpt_name
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    state = ckpt.get("model", ckpt.get("params", ckpt))
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    dim = state["head.weight"].shape[0]
    ppb = 7
    n1 = sum(1 for k in state if k.startswith("body.enc1.")) // ppb
    n2 = sum(1 for k in state if k.startswith("body.enc2.")) // ppb
    n3 = sum(1 for k in state if k.startswith("body.dec2.")) // ppb
    nmid = sum(1 for k in state if k.startswith("body.mid.")) // ppb
    from core import ArtRT
    model = ArtRT(dim=dim, n1=n1, n2=n2, n3=n3, nmid=nmid)
    model.load_state_dict(state, strict=True)
    return model, dim, n1, n2, n3, nmid


def load_video(path):
    path = Path(path)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30)
    w, h = map(int, result.stdout.strip().split(","))
    proc = subprocess.run([
        "ffmpeg", "-vsync", "0", "-hide_banner", "-i", str(path),
        "-vf", "setparams=color_primaries=bt2020:color_trc=bt709, zscale=matrix=bt2020nc:transfer=bt709:primaries=bt709:range=full",
        "-f", "rawvideo", "-pix_fmt", "yuv444p12le",
        "-s", f"{w}x{h}", "pipe:1",
    ], capture_output=True, timeout=120)
    raw = np.frombuffer(proc.stdout, dtype=np.uint16)
    nf = raw.size // (3 * h * w)
    planes = raw[:nf * 3 * h * w].reshape(nf, 3, h, w)
    return np.transpose(planes, (0, 2, 3, 1)), w, h


# ── SSE progress ──

async def event_stream(job_id: str):
    while True:
        job = jobs.get(job_id)
        if job is None:
            break
        data = json.dumps({
            "progress": job.get("progress", 0),
            "status": job.get("status", ""),
            "done": job.get("done", False),
            "error": job.get("error", ""),
            "result": job.get("result", ""),
        })
        yield f"data: {data}\n\n"
        if job.get("done", False) or job.get("error", ""):
            break
        await asyncio.sleep(0.2)


# ── inference runner ──

PREVIEW_MAX_W = 960


def save_preview_png(path, arr):
    """Save float32 [0,1] HWC numpy as PNG, resize if wider than PREVIEW_MAX_W."""
    from PIL import Image
    h, w = arr.shape[:2]
    if w > PREVIEW_MAX_W:
        scale = PREVIEW_MAX_W / w
        img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
            (int(w * scale), int(h * scale)), Image.NEAREST)
    else:
        img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(str(path), format="PNG")


def run_inference(job_id: str, video_path: str, ckpt_name: str):
    job = jobs[job_id]
    try:
        job["status"] = "加载模型..."
        job["progress"] = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16
        model, dim, n1, n2, n3, nmid = load_model(ckpt_name)
        model = model.to(device, dtype=dtype)
        model.eval()

        job["status"] = "加载视频..."
        job["progress"] = 0.1
        frames, w, h = load_video(video_path)
        n_frames = len(frames)

        out_name = Path(video_path).stem
        out_mkv = str(TEMP_DIR / f"{out_name}_{uuid.uuid4().hex[:8]}.mkv")

        # job dir for preview frames
        job_dir = TEMP_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        # determine frame sampling for preview (max 50 frames)
        preview_step = max(1, n_frames // 50)
        preview_indices = list(range(0, n_frames, preview_step))
        if preview_indices[-1] != n_frames - 1:
            preview_indices.append(n_frames - 1)

        job["nframes"] = n_frames
        job["preview_indices"] = preview_indices

        job["status"] = "推理中..."
        job["progress"] = 0.2
        ff = subprocess.Popen([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s", f"{w}x{h}", "-r", "30",
            "-i", "pipe:0",
            "-vf", "lutrgb=r=gammaval(0.45):g=gammaval(0.45):b=gammaval(0.45)",
            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1",
            "-pix_fmt", "yuv444p12le",
            out_mkv,
        ], stdin=subprocess.PIPE)

        preview_set = set(preview_indices)
        batch_size = 4
        with torch.no_grad():
            for i in range(0, n_frames, batch_size):
                batch = frames[i:i + batch_size]
                rgb = yuv_to_rgb_linear_np(batch, bits=12)
                inp = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device, dtype=dtype)
                pred = model(inp)
                pred_np = pred.float().cpu().permute(0, 2, 3, 1).numpy()
                pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)

                # save preview frames for this batch
                for j in range(len(batch)):
                    idx = i + j
                    if idx in preview_set:
                        before_path = job_dir / f"b_{idx:04d}.png"
                        save_preview_png(before_path, srgb_gamma(rgb[j]))
                        after_path = job_dir / f"a_{idx:04d}.png"
                        save_preview_png(after_path, srgb_gamma(pred_np[j]))

                raw_out = (pred_np * 65535).clip(0, 65535).astype(np.uint16)
                ff.stdin.write(raw_out.tobytes())
                job["progress"] = 0.2 + 0.7 * min((i + batch_size) / n_frames, 1.0)

        ff.stdin.close()
        ff.wait()

        job["progress"] = 1.0
        job["status"] = "完成"
        job["done"] = True
        job["result"] = out_mkv

    except Exception as e:
        job["error"] = str(e)
        job["status"] = "错误"
        job["done"] = True


# ── API routes ──

@app.get("/api/models")
def api_models():
    return scan_checkpoints()


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix if file.filename else ".mkv"
    dst = TEMP_DIR / f"upload_{uuid.uuid4().hex}{ext}"
    content = await file.read()
    dst.write_bytes(content)
    return {"path": str(dst), "name": file.filename}


@app.post("/api/infer")
async def api_infer(video_path: str = Form(...), ckpt_name: str = Form(...)):
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"progress": 0, "status": "排队中", "done": False, "error": "", "result": ""}
    # run in background thread
    import threading
    t = threading.Thread(target=run_inference, args=(job_id, video_path, ckpt_name), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/api/progress/{job_id}")
async def api_progress(job_id: str):
    return StreamingResponse(event_stream(job_id), media_type="text/event-stream")


@app.get("/api/frame/{name}")
async def api_frame(name: str):
    path = TEMP_DIR / name
    if path.exists():
        arr = np.load(str(path))
        import io
        from PIL import Image
        img = Image.fromarray((arr * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    return HTMLResponse("", status_code=404)


@app.get("/api/preview/{job_id}/{idx}/{typ}")
async def api_preview(job_id: str, idx: int, typ: str):
    path = TEMP_DIR / job_id / f"{typ}_{idx:04d}.png"
    if not path.exists():
        return HTMLResponse("", status_code=404)
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/nframes/{job_id}")
async def api_nframes(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return {"nframes": 0, "indices": []}
    return {"nframes": job.get("nframes", 0), "indices": job.get("preview_indices", [])}


@app.get("/api/download/{filename:path}")
async def api_download(filename: str):
    path = TEMP_DIR / filename
    if not path.exists():
        return HTMLResponse("File not found", status_code=404)
    return FileResponse(str(path), filename=filename, media_type="video/x-matroska")


# ── HTML page ──

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ArtRT 推理工具</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #0f0f13;
  color: #e0e0e0;
  height: 100vh;
  display: flex;
  flex-direction: column;
}
header {
  padding: 16px 24px;
  background: #1a1a24;
  border-bottom: 1px solid #2a2a3a;
  font-size: 20px;
  font-weight: 600;
  color: #fff;
  letter-spacing: 0.5px;
}
header span { color: #7c7cff; }
.main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── sidebar ── */
.sidebar {
  width: 500px;
  min-width: 500px;
  background: #14141e;
  border-right: 1px solid #2a2a3a;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
}
.sidebar h3 {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #888;
  margin-bottom: 4px;
}
.model-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.model-item {
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.15s;
  background: transparent;
  border: 1px solid transparent;
  color: #ccc;
}
.model-item:hover { background: #1e1e30; }
.model-item.active {
  background: #2a2a50;
  border-color: #5c5cff;
  color: #fff;
}
.btn {
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.15s;
  background: #252535;
  color: #ccc;
}
.btn:hover { background: #30304a; }
.btn-primary {
  background: #5c5cff;
  color: #fff;
  font-weight: 500;
}
.btn-primary:hover { background: #6e6eff; }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

.file-area {
  border: 2px dashed #333;
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  font-size: 13px;
  color: #888;
  cursor: pointer;
  transition: 0.15s;
}
.file-area:hover, .file-area.dragover { border-color: #5c5cff; background: #1a1a2a; }
.file-area.has-file { border-color: #4a4; background: #0f1f0f; }

/* ── content ── */
.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
  gap: 16px;
  overflow: auto;
}

/* ── comparison slider ── */
.comparison-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 0;
  overflow: hidden;
}
.comparison-container {
  position: relative;
  max-width: 100%;
  max-height: 100%;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  display: none;
  transform-origin: center center;
}
.comparison-container.visible {
  display: inline-block;
}
.comparison-container img {
  display: block;
  max-width: 80vw;
  max-height: 70vh;
  object-fit: contain;
  image-rendering: pixelated;
  user-select: none;
  -webkit-user-drag: none;
}
.after-wrapper {
  position: absolute;
  top: 0; left: 0;
  width: 50%;
  height: 100%;
  overflow: hidden;
}
.after-wrapper img {
  position: absolute;
  top: 0; left: 0;
  image-rendering: pixelated;
  user-select: none;
  -webkit-user-drag: none;
}
.slider-handle {
  position: absolute;
  top: 0; bottom: 0;
  width: 4px;
  background: #fff;
  cursor: ew-resize;
  box-shadow: 0 0 8px rgba(0,0,0,0.5);
  left: 50%;
  z-index: 2;
}
.slider-handle::before {
  content: "↔";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: #fff;
  border-radius: 50%;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* ── zoom controls ── */
.zoom-controls {
  display: none;
  justify-content: center;
  gap: 8px;
  margin-top: 8px;
}
.zoom-controls.visible { display: flex; }
.zoom-controls button {
  background: #252535;
  border: 1px solid #3a3a4a;
  color: #ccc;
  width: 36px; height: 36px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  transition: 0.15s;
}
.zoom-controls button:hover {
  background: #3a3a50;
  border-color: #5c5cff;
}
.zoom-controls .zoom-label {
  font-size: 13px;
  color: #888;
  line-height: 36px;
  min-width: 48px;
  text-align: center;
}

/* ── unified timeline / progress bar ── */
.timeline-wrap {
  display: none;
  flex-direction: column;
  gap: 4px;
}
.timeline-wrap.visible { display: flex; }
.timeline-wrap input[type="range"] {
  width: 100%;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: #252535;
  border-radius: 3px;
  outline: none;
  cursor: pointer;
  transition: background 0.3s;
}
.timeline-wrap input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px; height: 16px;
  border-radius: 50%;
  background: #7c7cff;
  cursor: pointer;
  border: 2px solid #fff;
  box-shadow: 0 0 6px rgba(0,0,0,0.3);
  transition: opacity 0.3s;
}
/* progress mode: hide thumb, show colored fill via gradient */
.timeline-wrap.progress input[type="range"] {
  pointer-events: none;
  background: linear-gradient(to right, #5c5cff, #8c8cff var(--pct, 0%), #252535 var(--pct, 0%));
}
.timeline-wrap.progress input[type="range"]::-webkit-slider-thumb {
  opacity: 0;
}
.timeline-labels {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #666;
}

/* ── status ── */
.status {
  font-size: 14px;
  color: #888;
  padding: 8px 0;
}
</style>
</head>
<body>
<header>ArtRT <span>推理工具</span></header>
<div class="main">
  <div class="sidebar">
    <div>
      <h3>模型</h3>
      <div class="model-list" id="modelList"></div>
      <button class="btn" id="refreshBtn" style="width:100%;margin-top:8px">刷新</button>
    </div>
    <div>
      <h3>视频输入</h3>
      <div class="file-area" id="fileArea">
        <div id="filePlaceholder">点击或拖拽选择视频</div>
        <input type="file" id="fileInput" accept=".mkv,.mp4,.avi,.mov,.webm" hidden>
      </div>
    </div>
    <a class="btn btn-primary" id="downloadBtn" style="display:none;text-decoration:none" download>下载无损结果</a>
  </div>

  <div class="content">
    <div class="comparison-wrap">
      <div class="comparison-container" id="comparisonContainer">
        <img id="beforeImg" crossorigin="anonymous" />
        <div class="after-wrapper" id="afterWrapper">
          <img id="afterImg" crossorigin="anonymous" />
        </div>
        <div class="slider-handle" id="sliderHandle"></div>
      </div>
    </div>
    <div class="zoom-controls" id="zoomControls">
      <button id="zoomOutBtn">−</button>
      <button id="zoomResetBtn">⟲</button>
      <button id="zoomInBtn">+</button>
      <span class="zoom-label" id="zoomLabel">100%</span>
    </div>
    <div class="timeline-wrap" id="timelineWrap">
      <input type="range" id="timeline" min="0" max="100" value="0" step="1">
      <div class="timeline-labels">
        <span id="timelineLabel">等待中...</span>
        <span id="framesTotal"></span>
      </div>
    </div>
    <div class="status" id="status">选择模型和视频后自动开始推理</div>
  </div>
</div>

<script>
// ── state ──
let selectedModel = null;
let videoPath = null;
let videoName = null;
let jobId = null;
let resultPath = null;
let isDragging = false;
let sliderPos = 50;
let zoom = 1;
const ZOOM_STEP = 0.25;
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 4;

// ── DOM ──
const modelList = document.getElementById('modelList');
const refreshBtn = document.getElementById('refreshBtn');
const fileArea = document.getElementById('fileArea');
const fileInput = document.getElementById('fileInput');
const filePlaceholder = document.getElementById('filePlaceholder');
const downloadBtn = document.getElementById('downloadBtn');
const comparisonContainer = document.getElementById('comparisonContainer');
const beforeImg = document.getElementById('beforeImg');
const afterImg = document.getElementById('afterImg');
const afterWrapper = document.getElementById('afterWrapper');
const sliderHandle = document.getElementById('sliderHandle');
const zoomControls = document.getElementById('zoomControls');
const zoomLabel = document.getElementById('zoomLabel');
const timelineWrap = document.getElementById('timelineWrap');
const timeline = document.getElementById('timeline');
const timelineLabel = document.getElementById('timelineLabel');
const framesTotal = document.getElementById('framesTotal');
const status = document.getElementById('status');

// ── unified timeline (progress + frame scrubber) ──
function setProgress(pct, label) {
  timelineWrap.classList.add('visible', 'progress');
  timeline.value = Math.round(pct * 100);
  timeline.style.setProperty('--pct', (pct * 100) + '%');
  timelineLabel.textContent = label;
}

function setTimeline(maxVal, val, label, totalLabel) {
  timelineWrap.classList.remove('progress');
  timeline.max = maxVal;
  timeline.value = val;
  timelineLabel.textContent = label;
  framesTotal.textContent = totalLabel;
}
function applyZoom() {
  comparisonContainer.style.transform = 'scale(' + zoom + ')';
  zoomLabel.textContent = Math.round(zoom * 100) + '%';
}

document.getElementById('zoomInBtn').onclick = () => {
  zoom = Math.min(ZOOM_MAX, zoom + ZOOM_STEP);
  applyZoom();
};
document.getElementById('zoomOutBtn').onclick = () => {
  zoom = Math.max(ZOOM_MIN, zoom - ZOOM_STEP);
  applyZoom();
};
document.getElementById('zoomResetBtn').onclick = () => {
  zoom = 1;
  applyZoom();
};

// ── slider logic ──
function updateSlider(pct) {
  sliderPos = Math.max(0, Math.min(100, pct));
  afterWrapper.style.width = sliderPos + '%';
  sliderHandle.style.left = sliderPos + '%';
}

comparisonContainer.addEventListener('mousedown', (e) => {
  const rect = comparisonContainer.getBoundingClientRect();
  isDragging = true;
  updateSlider((e.clientX - rect.left) / rect.width * 100);
});
document.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  const rect = comparisonContainer.getBoundingClientRect();
  updateSlider((e.clientX - rect.left) / rect.width * 100);
});
document.addEventListener('mouseup', () => { isDragging = false; });

comparisonContainer.addEventListener('touchstart', (e) => {
  const rect = comparisonContainer.getBoundingClientRect();
  const touch = e.touches[0];
  updateSlider((touch.clientX - rect.left) / rect.width * 100);
}, { passive: true });
document.addEventListener('touchmove', (e) => {
  if (!isDragging) return;
  const rect = comparisonContainer.getBoundingClientRect();
  const touch = e.touches[0];
  updateSlider((touch.clientX - rect.left) / rect.width * 100);
}, { passive: true });

// ── sync image dimensions ──
function syncImageSizes() {
  const bw = beforeImg.clientWidth;
  const bh = beforeImg.clientHeight;
  if (bw > 0 && bh > 0) {
    afterImg.style.width = bw + 'px';
    afterImg.style.height = bh + 'px';
    afterWrapper.style.height = bh + 'px';
  }
}

beforeImg.onload = () => {
  syncImageSizes();
  updateSlider(sliderPos);
  comparisonContainer.classList.add('visible');
  zoomControls.classList.add('visible');
};

afterImg.onload = () => {
  syncImageSizes();
};

window.addEventListener('resize', syncImageSizes);

// ── file upload ──
fileArea.addEventListener('click', () => fileInput.click());

fileArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  fileArea.classList.add('dragover');
});
fileArea.addEventListener('dragleave', () => fileArea.classList.remove('dragover'));
fileArea.addEventListener('drop', (e) => {
  e.preventDefault();
  fileArea.classList.remove('dragover');
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  filePlaceholder.textContent = '上传中...';
  const form = new FormData();
  form.append('file', file);
  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await resp.json();
    videoPath = data.path;
    videoName = data.name;
    fileArea.classList.add('has-file');
    filePlaceholder.textContent = '✓ ' + videoName;
    status.textContent = '视频已选择，点击模型开始推理';
    if (selectedModel) startInfer();
  } catch (e) {
    filePlaceholder.textContent = '上传失败: ' + e.message;
  }
}

// ── model list ──
async function loadModels() {
  const resp = await fetch('/api/models');
  const models = await resp.json();
  modelList.innerHTML = '';
  models.forEach(m => {
    const div = document.createElement('div');
    div.className = 'model-item' + (m === selectedModel ? ' active' : '');
    div.textContent = m;
    div.onclick = () => selectModel(m, div);
    modelList.appendChild(div);
  });
}

function selectModel(name, el) {
  document.querySelectorAll('.model-item').forEach(e => e.classList.remove('active'));
  el.classList.add('active');
  selectedModel = name;
  status.textContent = '已选择: ' + name;
  if (videoPath) startInfer();
}

refreshBtn.onclick = loadModels;

// ── inference ──
async function startInfer() {
  if (!selectedModel || !videoPath) return;

  zoom = 1;
  applyZoom();
  setProgress(0, '排队中...');
  comparisonContainer.classList.remove('visible');
  zoomControls.classList.remove('visible');
  downloadBtn.style.display = 'none';
  status.textContent = '启动推理...';

  const form = new FormData();
  form.append('video_path', videoPath);
  form.append('ckpt_name', selectedModel);

  try {
    const resp = await fetch('/api/infer', { method: 'POST', body: form });
    const data = await resp.json();
    jobId = data.job_id;
    listenProgress(jobId);
  } catch (e) {
    status.textContent = '错误: ' + e.message;
  }
}

async function listenProgress(jid) {
  const evtSource = new EventSource('/api/progress/' + jid);
  evtSource.onmessage = (e) => {
    const data = JSON.parse(e.data);
    setProgress(data.progress, data.status);

    if (data.done && data.error) {
      status.textContent = '错误: ' + data.error;
      evtSource.close();
      return;
    }

    if (data.done && data.result) {
      evtSource.close();
      resultPath = data.result;
      status.textContent = '推理完成';

      // switch timeline to frame scrubber
      fetch('/api/nframes/' + jid).then(r => r.json()).then(info => {
        const nf = info.nframes;
        const indices = info.indices;
        timelineWrap.classList.remove('progress');
        timeline.max = indices.length - 1;
        timeline.value = 0;
        timeline.style.setProperty('--pct', '0%');
        timelineLabel.textContent = '帧 ' + indices[0] + ' / ' + nf;
        framesTotal.textContent = '总帧数: ' + nf;
        loadFrame(jid, indices, 0);
      });

      const fname = resultPath.split('/').pop();
      downloadBtn.href = '/api/download/' + fname;
      downloadBtn.style.display = 'block';
    }
  };
}

// ── frame scrubbing ──
let previewIndices = [];
let currentJobId = null;

function loadFrame(jid, indices, idx) {
  if (idx < 0 || idx >= indices.length) return;
  previewIndices = indices;
  currentJobId = jid;
  const fi = indices[idx];
  timelineLabel.textContent = '帧 ' + fi + ' / ' + (indices.length > 0 ? indices[indices.length-1] : 0);
  beforeImg.src = '/api/preview/' + jid + '/' + fi + '/b?' + Date.now();
  afterImg.src = '/api/preview/' + jid + '/' + fi + '/a?' + Date.now();
}

timeline.addEventListener('input', () => {
  loadFrame(currentJobId, previewIndices, parseInt(timeline.value));
});

// ── init ──
loadModels();
</script>
</body>
</html>"""


@app.get("/")
def index():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
