#!/usr/bin/env python3
"""
Modern Photo Editing Server using InSPyReNet
4-Step Workflow: Upload → Choose Size → Adjust Position → Remove Background
"""

import os, io, base64, uuid, logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
CORS(app)

temp_images = {}
_model = None
_model_loaded = False

try:
    from transparent_background import Remover
    INSPYRENET_AVAILABLE = True
    logger.info("✅ InSPyReNet available")
except ImportError:
    INSPYRENET_AVAILABLE = False
    logger.error("❌ InSPyReNet not available")

def get_model():
    global _model, _model_loaded
    if not INSPYRENET_AVAILABLE:
        raise ValueError("InSPyReNet not available")
    if not _model_loaded:
        logger.info("🚀 Loading InSPyReNet model...")
        _model = Remover(mode='base')
        _model_loaded = True
        logger.info("✅ Model loaded")
    return _model

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file'}), 400
        allowed = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            return jsonify({'error': 'Invalid format'}), 400
        data = file.read()
        if len(data) > 50 * 1024 * 1024:
            return jsonify({'error': 'File too large'}), 400
        session_id = str(uuid.uuid4())
        temp_images[session_id] = {'original': data, 'filename': file.filename, 'size_choice': None, 'crop_settings': None, 'processed': None}
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        preview = base64.b64encode(data).decode('utf-8')
        logger.info(f"✅ Uploaded: {w}x{h}, session: {session_id}")
        return jsonify({'success': True, 'session_id': session_id, 'width': w, 'height': h, 'preview': f'data:image/{ext[1:]};base64,{preview}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/set-size', methods=['POST'])
def set_size():
    data = request.get_json()
    sid = data.get('session_id')
    if not sid or sid not in temp_images:
        return jsonify({'error': 'Invalid session'}), 400
    temp_images[sid]['size_choice'] = {'type': data.get('size'), 'custom_width': data.get('custom_width'), 'custom_height': data.get('custom_height')}
    return jsonify({'success': True})

@app.route('/set-crop', methods=['POST'])
def set_crop():
    data = request.get_json()
    sid = data.get('session_id')
    if not sid or sid not in temp_images:
        return jsonify({'error': 'Invalid session'}), 400
    temp_images[sid]['crop_settings'] = data.get('crop_settings')
    return jsonify({'success': True})

@app.route('/remove-background', methods=['POST'])
def remove_background():
    if not INSPYRENET_AVAILABLE:
        return jsonify({'error': 'Model not available'}), 500
    try:
        data = request.get_json()
        sid = data.get('session_id')
        bg_color = data.get('background_color', '#ffffff')
        if not sid or sid not in temp_images:
            return jsonify({'error': 'Invalid session'}), 400
        session = temp_images[sid]
        image = Image.open(io.BytesIO(session['original']))
        size_choice = session.get('size_choice', {})
        size_type = size_choice.get('type', 'original')
        sizes = {'passport_us': (600,600), 'passport_eu': (413,531), 'linkedin': (400,400), 'square_1000': (1000,1000)}
        target = sizes.get(size_type)
        if size_type == 'custom':
            target = (int(size_choice.get('custom_width', 400)), int(size_choice.get('custom_height', 400)))
        crop = session.get('crop_settings')
        if crop and target:
            image = apply_crop(image, crop, target)
        elif target:
            image = resize_crop(image, target)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        model = get_model()
        result = model.process(image, type='rgba')
        if bg_color:
            c = bg_color.lstrip('#')
            bg = Image.new('RGB', result.size, (int(c[0:2],16), int(c[2:4],16), int(c[4:6],16)))
            bg.paste(result, (0,0), result)
            result = bg
        out = io.BytesIO()
        fmt = 'PNG' if result.mode == 'RGBA' else 'JPEG'
        result.save(out, format=fmt, quality=95)
        session['processed'] = out.getvalue()
        b64 = base64.b64encode(out.getvalue()).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/{fmt.lower()};base64,{b64}', 'width': result.width, 'height': result.height})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>')
def download(session_id):
    if session_id not in temp_images or not temp_images[session_id].get('processed'):
        return jsonify({'error': 'Not found'}), 400
    s = temp_images[session_id]
    return send_file(io.BytesIO(s['processed']), mimetype='image/png', as_attachment=True, download_name=f"{Path(s['filename']).stem}_no_bg.png")

def apply_crop(img, crop, target):
    scale = crop.get('scale', 1.0)
    ox, oy = crop.get('offsetX', 0), crop.get('offsetY', 0)
    cw, ch = crop.get('canvasW', 400), crop.get('canvasH', 400)
    x, y = max(0, -ox/scale), max(0, -oy/scale)
    w, h = cw/scale, ch/scale
    iw, ih = img.size
    if x+w > iw: w = iw - x
    if y+h > ih: h = ih - y
    return img.crop((int(x), int(y), int(x+max(1,w)), int(y+max(1,h)))).resize(target, Image.Resampling.LANCZOS)

def resize_crop(img, target):
    tw, th = target
    iw, ih = img.size
    s = max(tw/iw, th/ih)
    nw, nh = int(iw*s), int(ih*s)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    l, t = (nw-tw)//2, (nh-th)//2
    return img.crop((l, t, l+tw, t+th))

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Passport Photo Editor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--t:0.2s ease}
[data-theme="dark"]{--bg:#09090b;--bg2:#18181b;--bg3:#27272a;--tx:#fafafa;--tx2:#a1a1aa;--tx3:#71717a;--bd:#27272a;--ac:#6366f1;--ac2:#818cf8;--acbg:rgba(99,102,241,0.1);--ok:#22c55e;--okbg:rgba(34,197,94,0.1);--err:#ef4444;--sh:0 4px 12px rgba(0,0,0,0.4)}
[data-theme="light"]{--bg:#fff;--bg2:#fafafa;--bg3:#f4f4f5;--tx:#18181b;--tx2:#52525b;--tx3:#a1a1aa;--bd:#e4e4e7;--ac:#6366f1;--ac2:#4f46e5;--acbg:rgba(99,102,241,0.08);--ok:#16a34a;--okbg:rgba(22,163,74,0.08);--err:#dc2626;--sh:0 4px 12px rgba(0,0,0,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;transition:background var(--t),color var(--t);display:flex;flex-direction:column}
.app{max-width:720px;margin:0 auto;padding:48px 24px;flex:1;width:100%}
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px}
.hdr-l{display:flex;align-items:center;gap:12px}
.logo{width:40px;height:40px;background:var(--ac);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px}
.hdr h1{font-size:1.25rem;font-weight:600}
.thm{width:44px;height:24px;background:var(--bg3);border-radius:12px;cursor:pointer;position:relative;border:1px solid var(--bd)}
.thm::after{content:'';position:absolute;width:18px;height:18px;background:var(--tx);border-radius:50%;top:2px;left:2px;transition:transform var(--t)}
[data-theme="light"] .thm::after{transform:translateX(20px)}
.steps{display:flex;justify-content:center;gap:8px;margin-bottom:32px;flex-wrap:wrap}
.step{display:flex;align-items:center;gap:6px;padding:8px 14px;background:var(--bg2);border:1px solid var(--bd);border-radius:20px;font-size:13px;font-weight:500;color:var(--tx3);transition:all var(--t)}
.step.active{background:var(--acbg);border-color:var(--ac);color:var(--ac)}
.step.done{background:var(--okbg);border-color:var(--ok);color:var(--ok);cursor:pointer}
.snum{width:20px;height:20px;border-radius:50%;background:var(--bg3);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600}
.step.active .snum{background:var(--ac);color:#fff}
.step.done .snum{background:var(--ok);color:#fff}
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:16px;padding:32px;box-shadow:var(--sh)}
.sec{display:none}.sec.active{display:block;animation:fade .3s ease}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.upz{border:2px dashed var(--bd);border-radius:12px;padding:48px 24px;text-align:center;cursor:pointer;background:var(--bg);transition:all var(--t)}
.upz:hover,.upz.drag{border-color:var(--ac);background:var(--acbg)}
.upi{font-size:40px;margin-bottom:16px;opacity:0.8}
.upz h3{font-size:1rem;font-weight:600;margin-bottom:4px}
.upz p{color:var(--tx3);font-size:0.875rem}
#fi{display:none}
.prev{margin-top:24px;text-align:center;display:none}.prev.vis{display:block}
.prevc{position:relative;display:inline-block;border-radius:12px;overflow:hidden;box-shadow:var(--sh)}
.prevc img{max-width:100%;max-height:300px;display:block}
.badge{position:absolute;top:12px;right:12px;background:var(--ok);color:#fff;padding:4px 10px;border-radius:12px;font-size:12px;font-weight:500}
.info{margin-top:12px;color:var(--tx3);font-size:0.8rem}
.szg{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:24px}
.szo{background:var(--bg);border:1px solid var(--bd);border-radius:10px;padding:16px 12px;cursor:pointer;text-align:center;transition:all var(--t)}
.szo:hover{border-color:var(--ac);transform:translateY(-2px)}
.szo.sel{border-color:var(--ac);background:var(--acbg)}
.szi{font-size:24px;margin-bottom:8px}
.szn{font-size:0.8rem;font-weight:600;margin-bottom:2px}
.szd{color:var(--tx3);font-size:0.7rem}
.cust{display:none;margin-top:16px;padding:16px;background:var(--bg);border-radius:10px;border:1px solid var(--bd)}
.cust.vis{display:flex;gap:12px;align-items:center;justify-content:center}
.cust input{width:80px;padding:10px;border:1px solid var(--bd);border-radius:8px;background:var(--bg2);color:var(--tx);font-size:0.9rem;text-align:center}
.cust input:focus{outline:none;border-color:var(--ac)}
.cust span{color:var(--tx3)}
.posed{text-align:center}
.posed h3{margin-bottom:20px;font-size:1rem;font-weight:500}
.cropc{display:inline-block;margin-bottom:16px;border-radius:12px;overflow:hidden;border:1px solid var(--bd)}
.cropc canvas{display:block;cursor:grab}
.cropc canvas:active{cursor:grabbing}
.zctrl{display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:16px;padding:12px 16px;background:var(--bg);border-radius:10px;border:1px solid var(--bd)}
.zbtn{width:32px;height:32px;border:1px solid var(--bd);border-radius:8px;background:var(--bg2);color:var(--tx);font-size:1.1rem;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all var(--t)}
.zbtn:hover{border-color:var(--ac);background:var(--acbg)}
.zslide{width:120px;height:4px;border-radius:2px;background:var(--bg3);appearance:none;cursor:pointer}
.zslide::-webkit-slider-thumb{appearance:none;width:16px;height:16px;border-radius:50%;background:var(--ac);cursor:pointer}
.zlbl{color:var(--tx3);font-size:0.8rem;min-width:50px}
.pctrl{display:flex;gap:8px;justify-content:center;margin-bottom:20px}
.pbtn{padding:8px 14px;border:1px solid var(--bd);border-radius:8px;background:var(--bg);color:var(--tx2);font-size:0.8rem;font-weight:500;cursor:pointer;transition:all var(--t)}
.pbtn:hover{border-color:var(--ac);color:var(--ac)}
.proc{text-align:center}
.bgsel{margin-bottom:24px}
.bgsel h4{font-size:0.875rem;margin-bottom:12px;color:var(--tx2);font-weight:500}
.colors{display:flex;justify-content:center;gap:8px;flex-wrap:wrap}
.col{width:40px;height:40px;border-radius:10px;cursor:pointer;border:2px solid transparent;transition:all var(--t);position:relative}
.col:hover{transform:scale(1.08)}
.col.sel{border-color:var(--ac);box-shadow:0 0 0 2px var(--acbg)}
.col.sel::after{content:'✓';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:14px;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,0.3)}
.trans{background:repeating-conic-gradient(#808080 0% 25%,#c0c0c0 0% 50%) 50%/12px 12px}
.progc{display:none;margin:24px 0}.progc.vis{display:block}
.progbg{height:4px;background:var(--bg3);border-radius:2px;overflow:hidden}
.prog{height:100%;width:0%;background:linear-gradient(90deg,var(--ac),var(--ac2));border-radius:2px;transition:width 0.3s}
.progt{margin-top:10px;color:var(--tx3);font-size:0.8rem}
.res{display:none;text-align:center}.res.vis{display:block;animation:fade .5s ease}
.resok{background:var(--okbg);color:var(--ok);padding:10px 20px;border-radius:20px;font-weight:500;display:inline-flex;align-items:center;gap:8px;margin-bottom:20px;font-size:0.9rem}
.resprev{display:inline-block;margin-bottom:20px}
.resprev img{max-width:100%;max-height:350px;border-radius:12px;box-shadow:var(--sh)}
.chk{background:repeating-conic-gradient(var(--bg3) 0% 25%,var(--bg2) 0% 50%) 50%/16px 16px;border-radius:12px;padding:12px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 24px;border:none;border-radius:10px;font-size:0.9rem;font-weight:500;cursor:pointer;transition:all var(--t);font-family:inherit}
.btn-p{background:var(--ac);color:#fff}
.btn-p:hover:not(:disabled){background:var(--ac2);transform:translateY(-1px);box-shadow:var(--sh)}
.btn-ok{background:var(--ok);color:#fff}
.btn-ok:hover:not(:disabled){filter:brightness(1.1);transform:translateY(-1px)}
.btn-s{background:var(--bg);color:var(--tx);border:1px solid var(--bd)}
.btn-s:hover:not(:disabled){border-color:var(--ac);color:var(--ac)}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.btng{display:flex;gap:12px;justify-content:center;margin-top:24px;flex-wrap:wrap}
.err{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:var(--err);padding:12px 16px;border-radius:10px;margin-bottom:20px;display:none;font-size:0.875rem}
.err.vis{display:flex;align-items:center;gap:10px}
.footer{padding:16px 24px;text-align:center;font-size:0.75rem;color:var(--tx3);border-top:1px solid var(--bd)}
.footer span{margin:0 6px;opacity:0.5}
.footer a{color:var(--tx3);text-decoration:none}
.footer a:hover{color:var(--ac)}
@media(max-width:640px){.app{padding:24px 16px}.card{padding:24px 20px}.hdr{flex-direction:column;gap:16px;text-align:center}.steps{gap:6px}.step{padding:6px 10px;font-size:12px}.snum{width:18px;height:18px;font-size:10px}.szg{grid-template-columns:repeat(2,1fr)}.btng{flex-direction:column}.btn{width:100%}.zctrl,.pctrl{flex-wrap:wrap;gap:8px}}
</style>
</head>
<body>
<div class="app">
<header class="hdr">
<div class="hdr-l"><div class="logo">📷</div><h1>Passport Photo Editor</h1></div>
<div class="thm" onclick="toggleTheme()"></div>
</header>
<div class="steps">
<div class="step active" id="s1"><div class="snum">1</div><span>Upload</span></div>
<div class="step" id="s2"><div class="snum">2</div><span>Size</span></div>
<div class="step" id="s3"><div class="snum">3</div><span>Position</span></div>
<div class="step" id="s4"><div class="snum">4</div><span>Remove</span></div>
</div>
<div class="card">
<div class="err" id="err"><span>⚠️</span><span id="errtxt"></span></div>
<div class="sec active" id="sec1">
<div class="upz" id="upz"><div class="upi">📷</div><h3>Drop your photo here</h3><p>or click to browse • JPG, PNG, WebP</p></div>
<input type="file" id="fi" accept="image/*">
<div class="prev" id="prev"><div class="prevc"><img id="pimg" src="" alt=""><div class="badge">✓ Uploaded</div></div><div class="info" id="pinfo"></div>
<div class="btng"><button class="btn btn-s" onclick="reset()">Choose Different</button><button class="btn btn-p" onclick="go(2)">Next →</button></div></div>
</div>
<div class="sec" id="sec2">
<div class="szg">
<div class="szo" data-sz="original" onclick="selSz('original')"><div class="szi">📐</div><div class="szn">Original</div><div class="szd">Keep size</div></div>
<div class="szo" data-sz="passport_us" onclick="selSz('passport_us')"><div class="szi">🪪</div><div class="szn">US Passport</div><div class="szd">600×600px</div></div>
<div class="szo" data-sz="passport_eu" onclick="selSz('passport_eu')"><div class="szi">🛂</div><div class="szn">EU Passport</div><div class="szd">413×531px</div></div>
<div class="szo" data-sz="linkedin" onclick="selSz('linkedin')"><div class="szi">💼</div><div class="szn">LinkedIn</div><div class="szd">400×400px</div></div>
<div class="szo" data-sz="square_1000" onclick="selSz('square_1000')"><div class="szi">⬛</div><div class="szn">Square HD</div><div class="szd">1000×1000px</div></div>
<div class="szo" data-sz="custom" onclick="selSz('custom')"><div class="szi">✏️</div><div class="szn">Custom</div><div class="szd">Enter size</div></div>
</div>
<div class="cust" id="cust"><input type="number" id="cw" placeholder="Width" min="50" max="5000"><span>×</span><input type="number" id="ch" placeholder="Height" min="50" max="5000"><span>px</span></div>
<div class="btng"><button class="btn btn-s" onclick="go(1)">← Back</button><button class="btn btn-p" id="s2btn" onclick="go(3)" disabled>Next →</button></div>
</div>
<div class="sec" id="sec3">
<div class="posed"><h3>Adjust Position</h3>
<div class="cropc"><canvas id="canvas"></canvas></div>
<div class="zctrl"><button class="zbtn" onclick="zout()">−</button><input type="range" class="zslide" id="zslide" min="10" max="200" value="100"><button class="zbtn" onclick="zin()">+</button><span class="zlbl" id="zlbl">100%</span></div>
<div class="pctrl"><button class="pbtn" onclick="resetPos()">↺ Reset</button><button class="pbtn" onclick="center()">⊙ Center</button><button class="pbtn" onclick="fit()">⊡ Fit</button></div>
<div class="btng"><button class="btn btn-s" onclick="go(2)">← Back</button><button class="btn btn-p" onclick="saveCrop()">Next →</button></div></div>
</div>
<div class="sec" id="sec4">
<div class="proc" id="proc">
<div class="bgsel"><h4>Background color</h4>
<div class="colors">
<div class="col sel" data-c="#ffffff" style="background:#fff" onclick="selCol('#ffffff')"></div>
<div class="col trans" data-c="transparent" onclick="selCol('transparent')"></div>
<div class="col" data-c="#f0f0f0" style="background:#f0f0f0" onclick="selCol('#f0f0f0')"></div>
<div class="col" data-c="#87CEEB" style="background:#87CEEB" onclick="selCol('#87CEEB')"></div>
<div class="col" data-c="#90EE90" style="background:#90EE90" onclick="selCol('#90EE90')"></div>
<div class="col" data-c="#FFB6C1" style="background:#FFB6C1" onclick="selCol('#FFB6C1')"></div>
</div></div>
<div class="btng"><button class="btn btn-s" onclick="go(3)">← Back</button><button class="btn btn-ok" id="procbtn" onclick="process()">🚀 Remove Background</button></div>
<div class="progc" id="progc"><div class="progbg"><div class="prog" id="prog"></div></div><div class="progt" id="progt">Processing...</div></div>
</div>
<div class="res" id="res"><div class="resok">✨ Done!</div><div class="resprev chk"><img id="resimg" src="" alt=""></div>
<div class="btng"><button class="btn btn-s" onclick="startOver()">Start Over</button><button class="btn btn-ok" onclick="dl()">⬇️ Download</button></div></div>
</div>
</div>
</div>
<footer class="footer">© 2026 Passport Photo Editor<span>·</span><a href="#">Privacy</a><span>·</span><a href="#">Terms</a><span>·</span>Made with ❤️</footer>
<script>
let S={step:1,sid:null,img:null,sz:null,col:'#ffffff',iw:0,ih:0,sc:1,ox:0,oy:0,tw:400,th:400};
let cv,ctx,limg,drag=false,dx,dy;
const upz=document.getElementById('upz'),fi=document.getElementById('fi');
upz.onclick=()=>fi.click();
upz.ondragover=e=>{e.preventDefault();upz.classList.add('drag')};
upz.ondragleave=()=>upz.classList.remove('drag');
upz.ondrop=e=>{e.preventDefault();upz.classList.remove('drag');if(e.dataTransfer.files.length)upload(e.dataTransfer.files[0])};
fi.onchange=e=>{if(e.target.files.length)upload(e.target.files[0])};

function toggleTheme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark'}

async function upload(f){
hide();if(f.size>50*1024*1024){err('File too large');return}
const fd=new FormData();fd.append('image',f);
const r=await fetch('/upload',{method:'POST',body:fd});
const d=await r.json();
if(d.success){S.sid=d.session_id;S.img=d.preview;S.iw=d.width;S.ih=d.height;
document.getElementById('pimg').src=d.preview;document.getElementById('pinfo').textContent=d.width+'×'+d.height+'px';
upz.style.display='none';document.getElementById('prev').classList.add('vis')}
else err(d.error)}

function reset(){upz.style.display='block';document.getElementById('prev').classList.remove('vis');fi.value='';S.sid=null}

function selSz(sz){S.sz=sz;document.querySelectorAll('.szo').forEach(o=>o.classList.remove('sel'));
document.querySelector('[data-sz="'+sz+'"]').classList.add('sel');
document.getElementById('cust').classList.toggle('vis',sz==='custom');
document.getElementById('s2btn').disabled=false;
const szs={passport_us:[600,600],passport_eu:[413,531],linkedin:[400,400],square_1000:[1000,1000]};
if(szs[sz]){S.tw=szs[sz][0];S.th=szs[sz][1]}else if(sz==='original'){S.tw=S.iw;S.th=S.ih}}

function selCol(c){S.col=c;document.querySelectorAll('.col').forEach(o=>o.classList.remove('sel'));
document.querySelector('[data-c="'+c+'"]').classList.add('sel')}

function initCanvas(){
cv=document.getElementById('canvas');ctx=cv.getContext('2d');
limg=new Image();limg.onload=()=>resetPos();limg.src=S.img;
if(S.sz==='custom'){S.tw=parseInt(document.getElementById('cw').value)||400;S.th=parseInt(document.getElementById('ch').value)||400}
const mw=Math.min(400,window.innerWidth-80),r=S.tw/S.th;
cv.width=mw;cv.height=mw/r;
cv.onmousedown=e=>startDrag(e);cv.onmousemove=e=>doDrag(e);cv.onmouseup=endDrag;cv.onmouseleave=endDrag;
cv.ontouchstart=e=>{e.preventDefault();startDrag(e.touches[0])};cv.ontouchmove=e=>{e.preventDefault();doDrag(e.touches[0])};cv.ontouchend=endDrag;
cv.onwheel=e=>{e.preventDefault();zoomBy(e.deltaY>0?-10:10)};
document.getElementById('zslide').oninput=e=>{S.sc=e.target.value/100;draw();updZ()}}

function resetPos(){const sw=cv.width/S.iw,sh=cv.height/S.ih;S.sc=Math.max(sw,sh);
S.ox=(cv.width-S.iw*S.sc)/2;S.oy=(cv.height-S.ih*S.sc)/2;
document.getElementById('zslide').value=S.sc*100;updZ();draw()}

function center(){S.ox=(cv.width-S.iw*S.sc)/2;S.oy=(cv.height-S.ih*S.sc)/2;draw()}
function fit(){resetPos()}
function startDrag(e){drag=true;dx=e.clientX-S.ox;dy=e.clientY-S.oy}
function doDrag(e){if(!drag)return;S.ox=e.clientX-dx;S.oy=e.clientY-dy;draw()}
function endDrag(){drag=false}
function zin(){zoomBy(10)}
function zout(){zoomBy(-10)}
function zoomBy(d){const os=S.sc;S.sc=Math.max(0.1,Math.min(2,S.sc+d/100));
const cx=cv.width/2,cy=cv.height/2;S.ox=cx-(cx-S.ox)*(S.sc/os);S.oy=cy-(cy-S.oy)*(S.sc/os);
document.getElementById('zslide').value=S.sc*100;updZ();draw()}
function updZ(){document.getElementById('zlbl').textContent=Math.round(S.sc*100)+'%'}
function draw(){ctx.fillStyle='#1a1a1a';ctx.fillRect(0,0,cv.width,cv.height);
if(limg.complete)ctx.drawImage(limg,S.ox,S.oy,S.iw*S.sc,S.ih*S.sc)}

async function saveCrop(){
await fetch('/set-crop',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,crop_settings:{scale:S.sc,offsetX:S.ox,offsetY:S.oy,canvasW:cv.width,canvasH:cv.height}})});
go(4)}

async function process(){
hide();document.getElementById('procbtn').disabled=true;
document.getElementById('progc').classList.add('vis');
const pb=document.getElementById('prog');let p=0;
const iv=setInterval(()=>{p=Math.min(90,p+Math.random()*15);pb.style.width=p+'%'},500);
await fetch('/set-size',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,size:S.sz,custom_width:document.getElementById('cw').value,custom_height:document.getElementById('ch').value})});
document.getElementById('progt').textContent='Removing background...';
const r=await fetch('/remove-background',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,background_color:S.col==='transparent'?null:S.col})});
const d=await r.json();clearInterval(iv);pb.style.width='100%';
if(d.success){document.getElementById('resimg').src=d.image;
setTimeout(()=>{document.getElementById('proc').style.display='none';document.getElementById('res').classList.add('vis')},400)}
else{err(d.error);document.getElementById('procbtn').disabled=false;document.getElementById('progc').classList.remove('vis')}}

function dl(){if(S.sid)window.location.href='/download/'+S.sid}

function go(n){
if(n===2&&!S.sid){err('Upload an image first');return}
if(n===3&&!S.sz){err('Select a size');return}
S.step=n;updSteps();
document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
document.getElementById('sec'+n).classList.add('active');
if(n===3)initCanvas()}

function updSteps(){for(let i=1;i<=4;i++){
const s=document.getElementById('s'+i);s.classList.remove('active','done');
if(i<S.step){s.classList.add('done');s.onclick=()=>go(i)}
else if(i===S.step)s.classList.add('active')}}

function startOver(){
S={step:1,sid:null,img:null,sz:null,col:'#ffffff',iw:0,ih:0,sc:1,ox:0,oy:0,tw:400,th:400};
reset();document.querySelectorAll('.szo').forEach(o=>o.classList.remove('sel'));
document.getElementById('cust').classList.remove('vis');document.getElementById('s2btn').disabled=true;
document.getElementById('proc').style.display='block';document.getElementById('res').classList.remove('vis');
document.getElementById('progc').classList.remove('vis');document.getElementById('prog').style.width='0%';
document.getElementById('procbtn').disabled=false;go(1)}

function err(m){document.getElementById('errtxt').textContent=m;document.getElementById('err').classList.add('vis')}
function hide(){document.getElementById('err').classList.remove('vis')}
</script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"\\n🚀 Passport Photo Editor Server\\n📍 http://localhost:{port}\\n")
    app.run(host='0.0.0.0', port=port, debug=False)
