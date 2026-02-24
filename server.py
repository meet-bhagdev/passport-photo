#!/usr/bin/env python3
"""
Modern Photo Editing Server using InSPyReNet
4-Step Workflow: Upload → Choose Size → Adjust Position → Remove Background
"""

import os, io, base64, uuid, logging
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string, send_from_directory
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

# Face detection using OpenCV
_face_cascade = None
def get_face_detector():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade

def detect_face(image):
    """Detect face in image and return bounding box"""
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_detector()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Return largest face
    largest = max(faces, key=lambda f: f[2] * f[3])
    return tuple(largest)

# Country-specific photo specifications
PHOTO_SPECS = {
    # Passport Photos
    'passport_us': {'size': (600, 600), 'head_ratio': 0.60, 'eye_offset': 0.35, 'head_margin': 0.15, 'chin_margin': 0.10, 'name': '🇺🇸 US Passport (2×2 inch)'},
    'passport_eu': {'size': (413, 531), 'head_ratio': 0.65, 'eye_offset': 0.40, 'head_margin': 0.25, 'chin_margin': 0.15, 'name': '🇪🇺 EU/Schengen Passport (35×45mm)'},
    'passport_uk': {'size': (413, 531), 'head_ratio': 0.70, 'eye_offset': 0.38, 'head_margin': 0.20, 'chin_margin': 0.12, 'name': '🇬🇧 UK Passport (35×45mm)'},
    'passport_canada': {'size': (591, 827), 'head_ratio': 0.62, 'eye_offset': 0.45, 'head_margin': 0.20, 'chin_margin': 0.15, 'name': '🇨🇦 Canada Passport (50×70mm)'},
    'passport_india': {'size': (600, 600), 'head_ratio': 0.65, 'eye_offset': 0.35, 'head_margin': 0.18, 'chin_margin': 0.12, 'name': '🇮🇳 India Passport (51×51mm)'},
    'passport_china': {'size': (390, 567), 'head_ratio': 0.70, 'eye_offset': 0.40, 'head_margin': 0.22, 'chin_margin': 0.15, 'name': '🇨🇳 China Passport (33×48mm)'},
    # Visa Photos
    'visa_australia': {'size': (413, 531), 'head_ratio': 0.68, 'eye_offset': 0.40, 'head_margin': 0.22, 'chin_margin': 0.15, 'name': '🇦🇺 Australia Visa (35×45mm)'},
    'visa_japan': {'size': (413, 531), 'head_ratio': 0.70, 'eye_offset': 0.38, 'head_margin': 0.20, 'chin_margin': 0.12, 'name': '🇯🇵 Japan Visa (35×45mm)'},
    'visa_brazil': {'size': (591, 827), 'head_ratio': 0.60, 'eye_offset': 0.42, 'head_margin': 0.18, 'chin_margin': 0.12, 'name': '🇧🇷 Brazil Visa (50×70mm)'},
    'visa_saudi': {'size': (472, 709), 'head_ratio': 0.65, 'eye_offset': 0.40, 'head_margin': 0.20, 'chin_margin': 0.15, 'name': '🇸🇦 Saudi Arabia Visa (40×60mm)'},
    # Generic Passport Sizes
    'passport_40x50': {'size': (472, 591), 'head_ratio': 0.68, 'eye_offset': 0.40, 'head_margin': 0.20, 'chin_margin': 0.12, 'name': '📐 Generic 40×50mm (Russia/CIS)'},
    'passport_35x35': {'size': (413, 413), 'head_ratio': 0.65, 'eye_offset': 0.38, 'head_margin': 0.18, 'chin_margin': 0.12, 'name': '📐 Generic 35×35mm (Square)'},
    'passport_30x40': {'size': (354, 472), 'head_ratio': 0.70, 'eye_offset': 0.40, 'head_margin': 0.20, 'chin_margin': 0.12, 'name': '📐 Generic 30×40mm (Compact)'},
    # Generic Visa Sizes
    'visa_45x45': {'size': (531, 531), 'head_ratio': 0.62, 'eye_offset': 0.38, 'head_margin': 0.18, 'chin_margin': 0.12, 'name': '📐 Generic 45×45mm (Malaysia)'},
    'visa_47x47': {'size': (555, 555), 'head_ratio': 0.62, 'eye_offset': 0.38, 'head_margin': 0.18, 'chin_margin': 0.12, 'name': '📐 Generic 47×47mm (Square)'},
    'visa_50x50': {'size': (591, 591), 'head_ratio': 0.60, 'eye_offset': 0.38, 'head_margin': 0.18, 'chin_margin': 0.12, 'name': '📐 Generic 50×50mm (2 inch)'},
    # Other
    'linkedin': {'size': (400, 400), 'head_ratio': 0.55, 'eye_offset': 0.35, 'head_margin': 0.15, 'chin_margin': 0.10, 'name': 'LinkedIn (400×400px)'},
    'square_1000': {'size': (1000, 1000), 'head_ratio': 0.50, 'eye_offset': 0.35, 'head_margin': 0.15, 'chin_margin': 0.10, 'name': 'Square HD (1000×1000px)'},
}

def auto_crop_passport(image, target_size, standard='us'):
    """Auto-crop image to passport standards based on face detection"""
    face = detect_face(image)
    if face is None:
        return resize_crop(image, target_size)
    
    fx, fy, fw, fh = face
    iw, ih = image.size
    tw, th = target_size
    
    # Get specs for this standard
    spec = PHOTO_SPECS.get(standard, PHOTO_SPECS.get('passport_' + standard, PHOTO_SPECS['passport_us']))
    head_ratio = spec['head_ratio']
    eye_offset_ratio = spec['eye_offset']
    head_top_margin = spec['head_margin']
    chin_margin = spec['chin_margin']
    
    # Estimate head dimensions
    head_top = fy - int(fh * head_top_margin)
    head_bottom = fy + fh + int(fh * chin_margin)
    actual_head_height = head_bottom - head_top
    
    # Calculate required frame height to achieve target head ratio
    frame_height = int(actual_head_height / head_ratio)
    frame_width = int(frame_height * (tw / th))
    
    # Calculate crop position to put eyes at correct position
    face_center_x = fx + fw // 2
    face_center_y = fy + int(fh * 0.35)  # Eyes are roughly at 35% of face height
    
    # Position frame so eyes are at correct vertical position
    crop_top = int(face_center_y - (frame_height * eye_offset_ratio))
    crop_left = face_center_x - frame_width // 2
    
    # Ensure crop stays within image bounds
    crop_left = max(0, min(crop_left, iw - frame_width))
    crop_top = max(0, min(crop_top, ih - frame_height))
    
    # Adjust dimensions if they exceed image bounds
    if crop_left + frame_width > iw:
        frame_width = iw - crop_left
    if crop_top + frame_height > ih:
        frame_height = ih - crop_top
    
    # Crop and resize
    cropped = image.crop((crop_left, crop_top, crop_left + frame_width, crop_top + frame_height))
    return cropped.resize(target_size, Image.Resampling.LANCZOS)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/privacy-policy')
def privacy_policy():
    return render_template_string(PRIVACY_TEMPLATE)

@app.route('/terms-of-service')
def terms_of_service():
    return render_template_string(TERMS_TEMPLATE)

@app.route('/about')
def about():
    return render_template_string(ABOUT_TEMPLATE)

@app.route('/contact')
def contact():
    return render_template_string(CONTACT_TEMPLATE)

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

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
        # Ensure RGBA mode for proper PNG compatibility with macOS Finder
        if result.mode != 'RGBA':
            result = result.convert('RGBA')
        out = io.BytesIO()
        result.save(out, format='PNG', optimize=False)
        out.seek(0)
        session['processed'] = out.getvalue()
        b64 = base64.b64encode(out.getvalue()).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/png;base64,{b64}', 'width': result.width, 'height': result.height})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>')
def download(session_id):
    if session_id not in temp_images or not temp_images[session_id].get('processed'):
        return jsonify({'error': 'Not found'}), 400
    s = temp_images[session_id]
    return send_file(io.BytesIO(s['processed']), mimetype='image/png', as_attachment=True, download_name=f"{Path(s['filename']).stem}_no_bg.png")

@app.route('/download-cropped', methods=['POST'])
def download_cropped():
    """Download the cropped/resized image without background removal"""
    try:
        data = request.get_json()
        sid = data.get('session_id')
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
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        out = io.BytesIO()
        image.save(out, format='JPEG', quality=95)
        out.seek(0)
        session['processed'] = out.getvalue()
        b64 = base64.b64encode(out.getvalue()).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/jpeg;base64,{b64}', 'width': image.width, 'height': image.height})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-original/<session_id>')
def download_original(session_id):
    """Download the cropped image without background removal"""
    if session_id not in temp_images or not temp_images[session_id].get('processed'):
        return jsonify({'error': 'Not found'}), 400
    s = temp_images[session_id]
    return send_file(io.BytesIO(s['processed']), mimetype='image/jpeg', as_attachment=True, download_name=f"{Path(s['filename']).stem}_cropped.jpg")

@app.route('/auto-process', methods=['POST'])
def auto_process():
    """Auto-detect face, crop to passport standards, and remove background in one step"""
    if not INSPYRENET_AVAILABLE:
        return jsonify({'error': 'Model not available'}), 500
    try:
        data = request.get_json()
        sid = data.get('session_id')
        size_type = data.get('size', 'passport_us')
        bg_color = data.get('background_color', '#ffffff')
        if not sid or sid not in temp_images:
            return jsonify({'error': 'Invalid session'}), 400
        session = temp_images[sid]
        session['size_type'] = size_type  # Store for photo sheet
        image = Image.open(io.BytesIO(session['original']))
        # Determine target size and standard - use PHOTO_SPECS
        spec = PHOTO_SPECS.get(size_type, PHOTO_SPECS['passport_us'])
        target = spec['size']
        # Auto-crop based on face detection using the size_type as standard
        cropped = auto_crop_passport(image, target, size_type)
        if cropped.mode != 'RGB':
            cropped = cropped.convert('RGB')
        # Remove background
        model = get_model()
        result = model.process(cropped, type='rgba')
        if bg_color and bg_color != 'transparent':
            c = bg_color.lstrip('#')
            bg = Image.new('RGB', result.size, (int(c[0:2],16), int(c[2:4],16), int(c[4:6],16)))
            bg.paste(result, (0,0), result)
            result = bg
        if result.mode != 'RGBA':
            result = result.convert('RGBA')
        out = io.BytesIO()
        result.save(out, format='PNG', optimize=False)
        out.seek(0)
        session['processed'] = out.getvalue()
        # Also create photo sheet
        result_for_sheet = Image.open(io.BytesIO(session['processed']))
        if result_for_sheet.mode == 'RGBA':
            bg_sheet = Image.new('RGB', result_for_sheet.size, (255, 255, 255))
            bg_sheet.paste(result_for_sheet, (0, 0), result_for_sheet)
            result_for_sheet = bg_sheet
        sheet, count = create_photo_sheet(result_for_sheet, size_type)
        sheet_out = io.BytesIO()
        sheet.save(sheet_out, format='JPEG', quality=95)
        sheet_out.seek(0)
        session['photo_sheet'] = sheet_out.getvalue()
        session['sheet_count'] = count
        b64 = base64.b64encode(session['processed']).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/png;base64,{b64}', 'width': result.width, 'height': result.height, 'sheet_count': count})
    except Exception as e:
        logger.error(f"Auto-process error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-sheet/<session_id>')
def download_sheet(session_id):
    """Download the 4x6 photo print sheet"""
    if session_id not in temp_images or not temp_images[session_id].get('photo_sheet'):
        return jsonify({'error': 'Not found'}), 400
    s = temp_images[session_id]
    return send_file(io.BytesIO(s['photo_sheet']), mimetype='image/jpeg', as_attachment=True, download_name=f"{Path(s['filename']).stem}_4x6_sheet.jpg")

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

def create_photo_sheet(photo, size_type):
    """Create a 4x6 inch print sheet with multiple copies of the photo"""
    # 4x6 inch at 300 DPI = 1800 x 1200 pixels (landscape)
    sheet_width, sheet_height = 1800, 1200
    
    # Get photo size from specs or use photo's actual size
    spec = PHOTO_SPECS.get(size_type)
    if spec:
        pw, ph = spec['size']
    else:
        pw, ph = photo.size
    
    # Scale down large photos for sheet
    if pw > 600 or ph > 600:
        scale = 600 / max(pw, ph)
        pw, ph = int(pw * scale), int(ph * scale)
    
    # Resize photo if needed
    if photo.size != (pw, ph):
        photo = photo.resize((pw, ph), Image.Resampling.LANCZOS)
    
    # Calculate grid layout with small gaps for cutting guides
    gap = 4  # pixels between photos for cut lines
    cols = (sheet_width + gap) // (pw + gap)
    rows = (sheet_height + gap) // (ph + gap)
    
    # Create white sheet
    sheet = Image.new('RGB', (sheet_width, sheet_height), (255, 255, 255))
    
    # Calculate starting position to center the grid
    total_w = cols * pw + (cols - 1) * gap
    total_h = rows * ph + (rows - 1) * gap
    start_x = (sheet_width - total_w) // 2
    start_y = (sheet_height - total_h) // 2
    
    # Paste photos in grid
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * (pw + gap)
            y = start_y + row * (ph + gap)
            sheet.paste(photo, (x, y))
    
    # Draw cutting guide lines around each photo
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    line_color = (180, 180, 180)  # Light gray
    
    # Draw vertical lines (left and right of each column)
    for col in range(cols + 1):
        x = start_x + col * (pw + gap) - gap // 2
        if col == 0:
            x = start_x - 1
        elif col == cols:
            x = start_x + cols * pw + (cols - 1) * gap
        draw.line([(x, 0), (x, sheet_height)], fill=line_color, width=1)
    
    # Draw horizontal lines (top and bottom of each row)
    for row in range(rows + 1):
        y = start_y + row * (ph + gap) - gap // 2
        if row == 0:
            y = start_y - 1
        elif row == rows:
            y = start_y + rows * ph + (rows - 1) * gap
        draw.line([(0, y), (sheet_width, y)], fill=line_color, width=1)
    
    return sheet, cols * rows

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%236366f1'/%3E%3Crect x='6' y='8' width='20' height='16' rx='2' fill='%23fff'/%3E%3Ccircle cx='16' cy='16' r='5' fill='%236366f1'/%3E%3Ccircle cx='16' cy='16' r='3' fill='%23818cf8'/%3E%3Crect x='20' y='10' width='4' height='2' rx='1' fill='%236366f1'/%3E%3C/svg%3E">
<title>Free Passport Photo Editor - Create Professional Photos Online</title>
<meta name="description" content="Create professional passport and visa photos for free. Skip expensive CVS/Walgreens fees. AI-powered background removal, US &amp; EU passport sizes, instant download.">
<meta name="keywords" content="passport photo, visa photo, free passport photo, passport photo maker, background removal, passport photo online, US passport photo, EU passport photo">
<meta name="robots" content="index, follow">
<meta name="author" content="Passport Photo Editor">
<link rel="canonical" href="https://passport-photo-app.blueforest-5a95b458.westus2.azurecontainerapps.io/">
<meta property="og:type" content="website">
<meta property="og:title" content="Free Passport Photo Editor - No CVS Fees">
<meta property="og:description" content="Create professional passport photos for free. AI background removal, US &amp; EU sizes, instant download. Save $20+ vs stores.">
<meta property="og:url" content="https://passport-photo-app.blueforest-5a95b458.westus2.azurecontainerapps.io/">
<meta property="og:site_name" content="Passport Photo Editor">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Free Passport Photo Editor">
<meta name="twitter:description" content="Create professional passport photos for free. AI background removal, instant download.">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"WebApplication","name":"Passport Photo Editor","description":"Free online passport photo editor with AI background removal. Create US and EU passport photos, visa photos, and professional headshots instantly.","url":"https://passport-photo-app.blueforest-5a95b458.westus2.azurecontainerapps.io/","applicationCategory":"Photography","operatingSystem":"Web Browser","offers":{"@type":"Offer","price":"0","priceCurrency":"USD"},"featureList":["AI Background Removal","US Passport Photo (600x600)","EU Passport Photo (413x531)","Custom Sizes","Instant Download"]}
</script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--t:0.2s ease}
[data-theme="dark"]{--bg:#09090b;--bg2:#18181b;--bg3:#27272a;--tx:#fafafa;--tx2:#a1a1aa;--tx3:#71717a;--bd:#27272a;--ac:#6366f1;--ac2:#818cf8;--acbg:rgba(99,102,241,0.1);--ok:#22c55e;--okbg:rgba(34,197,94,0.1);--err:#ef4444;--sh:0 4px 12px rgba(0,0,0,0.4)}
[data-theme="light"]{--bg:#fff;--bg2:#fafafa;--bg3:#f4f4f5;--tx:#18181b;--tx2:#52525b;--tx3:#a1a1aa;--bd:#e4e4e7;--ac:#6366f1;--ac2:#4f46e5;--acbg:rgba(99,102,241,0.08);--ok:#16a34a;--okbg:rgba(22,163,74,0.08);--err:#dc2626;--sh:0 4px 12px rgba(0,0,0,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;transition:background var(--t),color var(--t);display:flex;flex-direction:column}
.navbar{position:sticky;top:0;z-index:100;display:flex;justify-content:space-between;align-items:center;padding:12px 24px;background:var(--bg);border-bottom:1px solid var(--bd);backdrop-filter:blur(8px)}
.nav-left{display:flex;align-items:center;gap:10px}
.nav-logo{width:32px;height:32px;background:var(--ac);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}
.nav-brand{font-weight:600;font-size:0.95rem}
.nav-right{display:flex;align-items:center;gap:20px}
.nav-link{color:var(--tx2);text-decoration:none;font-size:0.85rem;font-weight:500;transition:color var(--t)}
.nav-link:hover{color:var(--ac)}
.nav-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:linear-gradient(135deg,#ec4899,#f472b6);color:#fff;text-decoration:none;border-radius:20px;font-size:0.8rem;font-weight:500;transition:all var(--t)}
.nav-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(236,72,153,0.3)}
.app{max-width:720px;margin:0 auto;padding:48px 24px;flex:1;width:100%}
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px}
.hdr-l{display:flex;align-items:center;gap:12px}
.logo{width:40px;height:40px;background:var(--ac);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px}
.hdr h1{font-size:1.25rem;font-weight:600}
.hero{display:flex;align-items:center;justify-content:center;gap:48px;margin-bottom:32px;padding:0 16px}
.hero-text{flex:1;max-width:400px}
.hero h2{font-size:1.5rem;font-weight:700;margin-bottom:8px;background:linear-gradient(135deg,var(--ac),var(--ac2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero p{color:var(--tx2);font-size:0.95rem;margin:0 0 16px;line-height:1.6}
.features{display:flex;gap:16px;flex-wrap:wrap;font-size:0.85rem;color:var(--ok);font-weight:500}
.hero-demo{display:flex;gap:20px;flex-shrink:0}
.demo-card{position:relative;border-radius:12px;overflow:hidden;box-shadow:var(--sh);background:var(--bg)}
.demo-card img{width:140px;height:175px;object-fit:cover;display:block}
.demo-after img{background:#fff}
.demo-label{position:absolute;bottom:0;left:0;right:0;padding:8px;background:rgba(75,85,99,0.9);color:#fff;font-size:0.75rem;font-weight:700;text-align:center;letter-spacing:1px}
.thm{width:44px;height:24px;background:var(--bg3);border-radius:12px;cursor:pointer;position:relative;border:1px solid var(--bd)}
.thm::after{content:'';position:absolute;width:18px;height:18px;background:var(--tx);border-radius:50%;top:2px;left:2px;transition:transform var(--t)}
[data-theme="light"] .thm::after{transform:translateX(20px)}
.size-sel{margin-bottom:16px}
.size-sel label{display:block;font-size:0.85rem;font-weight:600;margin-bottom:8px;color:var(--tx2)}
.size-dropdown{width:100%;padding:14px 16px;border:1px solid var(--bd);border-radius:12px;background:var(--bg);color:var(--tx);font-size:0.95rem;font-family:inherit;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2371717a' d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 16px center;transition:border-color var(--t)}
.size-dropdown:hover{border-color:var(--ac)}
.size-dropdown:focus{outline:none;border-color:var(--ac)}
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
.auto-toggle{display:flex;align-items:center;gap:12px;padding:12px 16px;background:var(--bg);border:1px solid var(--bd);border-radius:12px;margin-bottom:16px;cursor:pointer;transition:all var(--t)}
.auto-toggle:hover{border-color:var(--ac)}
.auto-toggle.on{border-color:var(--ac);background:var(--acbg)}
.auto-label{font-weight:600;font-size:0.9rem;color:var(--tx)}
.auto-sw{width:44px;height:24px;background:var(--bg3);border-radius:12px;position:relative;border:1px solid var(--bd);transition:all var(--t)}
.auto-toggle.on .auto-sw{background:var(--ac);border-color:var(--ac)}
.auto-dot{position:absolute;width:18px;height:18px;background:#fff;border-radius:50%;top:2px;left:2px;transition:transform var(--t)}
.auto-toggle.on .auto-dot{transform:translateX(20px)}
.auto-hint{color:var(--tx3);font-size:0.8rem}
.upz{border:2px dashed var(--bd);border-radius:16px;padding:60px 32px;min-height:280px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;cursor:pointer;background:var(--bg);transition:all 0.3s ease;position:relative;overflow:hidden}
.upz::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--acbg) 0%,transparent 60%);opacity:0;transition:opacity 0.3s ease}
.upz:hover,.upz.drag{border-color:var(--ac);background:var(--bg);transform:translateY(-2px);box-shadow:0 8px 30px rgba(99,102,241,0.15)}
.upz:hover::before,.upz.drag::before{opacity:1}
.upi{font-size:56px;margin-bottom:20px;transition:transform 0.3s ease;position:relative;z-index:1}
.upz:hover .upi{transform:scale(1.1)}
.upz h3{font-size:1.1rem;font-weight:600;margin-bottom:8px;position:relative;z-index:1}
.upz p{color:var(--tx3);font-size:0.9rem;position:relative;z-index:1}
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
.cropc{display:inline-block;margin-bottom:16px;border-radius:12px;overflow:hidden;border:1px solid var(--bd);position:relative}
.cropc canvas{display:block;cursor:grab}
.cropc canvas:active{cursor:grabbing}
.silhouette{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;display:none}
.silhouette.vis{display:block}
.silhouette svg{width:100%;height:100%}
.zctrl{display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:16px;padding:12px 16px;background:var(--bg);border-radius:10px;border:1px solid var(--bd)}
.zbtn{width:32px;height:32px;border:1px solid var(--bd);border-radius:8px;background:var(--bg2);color:var(--tx);font-size:1.1rem;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all var(--t)}
.zbtn:hover{border-color:var(--ac);background:var(--acbg)}
.zslide{width:120px;height:4px;border-radius:2px;background:var(--bg3);appearance:none;cursor:pointer}
.zslide::-webkit-slider-thumb{appearance:none;width:16px;height:16px;border-radius:50%;background:var(--ac);cursor:pointer}
.zlbl{color:var(--tx3);font-size:0.8rem;min-width:50px}
.pctrl{display:flex;gap:8px;justify-content:center;margin-bottom:20px}
.pbtn{padding:8px 14px;border:1px solid var(--bd);border-radius:8px;background:var(--bg);color:var(--tx2);font-size:0.8rem;font-weight:500;cursor:pointer;transition:all var(--t)}
.pbtn:hover{border-color:var(--ac);color:var(--ac)}
.guide-legend{display:none;justify-content:center;gap:16px;margin-bottom:12px;font-size:0.75rem;color:var(--tx2)}
.guide-legend.vis{display:flex}
.gl-item{display:flex;align-items:center;gap:5px}
.gl-dot{width:10px;height:10px;border-radius:3px}
.gl-purple{background:rgba(99,102,241,0.5)}
.gl-green{background:rgba(34,197,94,0.5)}
.gl-orange{background:rgba(249,115,22,0.5)}
.gl-pink{background:rgba(236,72,153,0.5)}
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
.gen-prog{display:none;margin-top:20px;padding:20px 24px;background:var(--bg);border:1px solid var(--bd);border-radius:14px}.gen-prog.vis{display:block;animation:fade .3s ease}
.gen-bar{height:6px;background:var(--bg3);border-radius:3px;overflow:hidden}
.gen-fill{height:100%;width:0%;background:linear-gradient(90deg,#6366f1,#8b5cf6,#a855f7);border-radius:3px;transition:width 0.5s ease}
.gen-status{margin-top:12px;color:var(--tx2);font-size:0.85rem;font-weight:500;text-align:center}
.res{display:none;text-align:center}.res.vis{display:block;animation:fade .5s ease}
.resok{color:var(--ok);font-weight:600;font-size:0.85rem;margin-bottom:24px;display:flex;align-items:center;justify-content:center;gap:6px}
.resprev{display:inline-block;margin-bottom:28px;position:relative}
.resprev img{max-width:100%;max-height:350px;border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,0.12)}
.imgframe{background:linear-gradient(145deg,var(--bg),var(--bg3));border-radius:20px;padding:16px;border:1px solid var(--bd)}
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
.about-section,.contact-section{padding:60px 24px;max-width:720px;margin:0 auto}
.about-content,.contact-content{background:var(--bg2);border:1px solid var(--bd);border-radius:16px;padding:40px;box-shadow:var(--sh)}
.about-section h2,.contact-section h2{font-size:1.5rem;font-weight:700;margin-bottom:24px;background:linear-gradient(135deg,var(--ac),var(--ac2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.about-story h3{font-size:1.1rem;font-weight:600;margin-bottom:16px;color:var(--tx)}
.about-story p{color:var(--tx2);line-height:1.8;margin-bottom:16px;font-size:0.95rem}
.about-story strong{color:var(--tx);font-weight:600}
.about-cta{margin-top:32px;text-align:center}
.contact-content p{color:var(--tx2);line-height:1.7;margin-bottom:12px}
.contact-content a{color:var(--ac);text-decoration:none}
.contact-content a:hover{text-decoration:underline}
.footer a:hover{color:var(--ac)}
.how-it-works{margin-top:60px;padding:60px 32px;background:var(--bg);border-radius:0;border-top:1px solid var(--bd)}
.how-it-works h3{text-align:center;font-size:1.75rem;font-weight:700;margin-bottom:16px;color:var(--tx)}
.how-it-works .hiw-sub{text-align:center;color:var(--tx3);font-size:0.95rem;margin-bottom:48px}
.hiw-steps{display:flex;justify-content:center;align-items:flex-start;gap:0;position:relative;max-width:900px;margin:0 auto}
.hiw-step{flex:1;text-align:center;position:relative;padding:0 16px;max-width:200px}
.hiw-num{width:48px;height:48px;border-radius:50%;background:var(--ac);color:#fff;display:flex;align-items:center;justify-content:center;font-size:1.1rem;font-weight:700;margin:0 auto 20px;position:relative;z-index:2;box-shadow:0 4px 12px rgba(99,102,241,0.25);transition:transform 0.2s ease,box-shadow 0.2s ease}
.hiw-step:hover .hiw-num{transform:scale(1.1);box-shadow:0 6px 20px rgba(99,102,241,0.35)}
.hiw-line{position:absolute;top:24px;left:calc(50% + 24px);right:calc(-50% + 24px);height:2px;background:linear-gradient(90deg,var(--ac),var(--bd));z-index:1}
.hiw-step:last-child .hiw-line{display:none}
.hiw-title{font-size:1rem;font-weight:600;color:var(--tx);margin-bottom:8px}
.hiw-desc{font-size:0.85rem;color:var(--tx3);line-height:1.6}
@media(max-width:640px){.app{padding:24px 16px}.card{padding:24px 20px}.hdr{flex-direction:column;gap:16px;text-align:center}.hero{flex-direction:column;text-align:center}.hero-text{text-align:center}.hero-demo{display:none}.features{justify-content:center}.steps{gap:6px}.step{padding:6px 10px;font-size:12px}.snum{width:18px;height:18px;font-size:10px}.szg{grid-template-columns:repeat(2,1fr)}.btng{flex-direction:column}.btn{width:100%}.zctrl,.pctrl{flex-wrap:wrap;gap:8px}.hiw-steps{flex-wrap:wrap;gap:32px}.hiw-step{flex:none;width:45%}.hiw-line{display:none}.hiw-num{width:40px;height:40px;font-size:1rem}}
</style>
</head>
<body>
<nav class="navbar">
<div class="nav-left"><div class="nav-logo">📷</div><span class="nav-brand">Passport Photo Editor</span></div>
<div class="nav-right">
<a href="/about" class="nav-link">About</a>
<a href="/contact" class="nav-link">Contact</a>
<a href="https://buymeacoffee.com/onestopshoppassportphotos" target="_blank" class="nav-btn">☕ Buy me a coffee</a>
<div class="thm" onclick="toggleTheme()"></div>
</div>
</nav>
<div class="app">
<div class="hero">
<div class="hero-text">
<h2>Create Professional Passport Photos — Free</h2>
<p>Skip the $20+ CVS/Walgreens fees. Upload your photo, resize to official specs, remove the background, and download instantly.</p>
<div class="features"><span>✓ US & EU Passport Sizes</span><span>✓ AI Background Removal</span><span>✓ 100% Free</span></div>
</div>
<div class="hero-demo">
<div class="demo-card demo-before"><img src="/static/images/before.jpg" alt="Before - Original photo"><span class="demo-label">BEFORE</span></div>
<div class="demo-card demo-after"><img src="/static/images/after.jpg" alt="After - Passport photo"><span class="demo-label">AFTER</span></div>
</div>
</div>
<div class="steps">
<div class="step active" id="s1"><div class="snum">1</div><span>Upload</span></div>
<div class="step" id="s2"><div class="snum">2</div><span>Process</span></div>
<div class="step" id="s3"><div class="snum">3</div><span>Download</span></div>
</div>
<div class="card">
<div class="err" id="err"><span>⚠️</span><span id="errtxt"></span></div>
<div class="sec active" id="sec1">
<div class="size-sel">
<label>Photo Specification</label>
<select class="size-dropdown" id="szsel" onchange="selSzDrop(this.value)">
<optgroup label="📸 Passport Photos">
<option value="passport_us">🇺🇸 US Passport (2×2 inch / 600×600px)</option>
<option value="passport_eu">🇪🇺 EU/Schengen Passport (35×45mm)</option>
<option value="passport_uk">🇬🇧 UK Passport (35×45mm)</option>
<option value="passport_canada">🇨🇦 Canada Passport (50×70mm)</option>
<option value="passport_india">🇮🇳 India Passport (51×51mm)</option>
<option value="passport_china">🇨🇳 China Passport (33×48mm)</option>
<option value="passport_40x50">📐 Generic 40×50mm (Russia/CIS)</option>
<option value="passport_35x35">📐 Generic 35×35mm (Square)</option>
<option value="passport_30x40">📐 Generic 30×40mm (Compact)</option>
</optgroup>
<optgroup label="🛂 Visa Photos">
<option value="visa_australia">🇦🇺 Australia Visa (35×45mm)</option>
<option value="visa_japan">🇯🇵 Japan Visa (35×45mm)</option>
<option value="visa_brazil">🇧🇷 Brazil Visa (50×70mm)</option>
<option value="visa_saudi">🇸🇦 Saudi Arabia Visa (40×60mm)</option>
<option value="visa_45x45">📐 Generic 45×45mm (Malaysia)</option>
<option value="visa_47x47">📐 Generic 47×47mm (Square)</option>
<option value="visa_50x50">📐 Generic 50×50mm (2 inch)</option>
</optgroup>
<optgroup label="📷 Other">
<option value="linkedin">LinkedIn (400×400px)</option>
<option value="square_1000">Square HD (1000×1000px)</option>
<option value="original">Original Size</option>
</optgroup>
</select>
</div>
<div class="auto-toggle" id="autoToggle" onclick="toggleAuto()"><span class="auto-label">🔧 Manual Mode</span><div class="auto-sw"><div class="auto-dot"></div></div><span class="auto-hint">Customize positioning</span></div>
<div class="upz" id="upz"><div class="upi">📷</div><h3>Drop your photo here</h3><p>or click to browse • JPG, PNG, WebP</p></div>
<input type="file" id="fi" accept="image/*">
<div class="prev" id="prev"><div class="prevc"><img id="pimg" src="" alt=""><div class="badge">✓ Uploaded</div></div><div class="info" id="pinfo"></div>
<div class="btng"><button class="btn btn-s" onclick="reset()">Choose Different</button><button class="btn btn-p" id="s1btn" onclick="processFromUpload()">✨ Generate Photo</button></div></div>
</div>
<div class="sec" id="sec2">
<div class="posed"><h3>Adjust Position</h3>
<div class="cropc"><canvas id="canvas"></canvas>
<div class="silhouette" id="silhouette">
<svg viewBox="0 0 100 100" preserveAspectRatio="none">
<!-- Zone bands only - no text labels inside -->
<!-- Head top zone (5-15% - where top of head should be) -->
<rect x="0" y="5" width="100" height="10" fill="rgba(99,102,241,0.12)"/>
<!-- Eye zone (31-44% from top) -->
<rect x="0" y="31" width="100" height="13" fill="rgba(34,197,94,0.12)"/>
<!-- Chin zone (65-75% from top) -->
<rect x="0" y="65" width="100" height="10" fill="rgba(249,115,22,0.1)"/>
<!-- Center vertical guide (nose alignment) -->
<line x1="50" y1="0" x2="50" y2="100" stroke="rgba(236,72,153,0.35)" stroke-width="0.5" stroke-dasharray="3,3"/>
</svg>
</div>
</div>
<div class="zctrl"><button class="zbtn" onclick="zout()">−</button><input type="range" class="zslide" id="zslide" min="0" max="100" value="100"><button class="zbtn" onclick="zin()">+</button><span class="zlbl" id="zlbl">100%</span></div>
<div class="guide-legend" id="guideLegend"><span class="gl-item"><span class="gl-dot gl-purple"></span>Top of head</span><span class="gl-item"><span class="gl-dot gl-green"></span>Eyes</span><span class="gl-item"><span class="gl-dot gl-orange"></span>Chin</span><span class="gl-item"><span class="gl-dot gl-pink"></span>Center</span></div>
<div class="pctrl"><button class="pbtn" onclick="resetPos()">↺ Reset</button><button class="pbtn" onclick="center()">⊙ Center</button><button class="pbtn" onclick="fit()">⊡ Fit</button></div>
<div class="btng"><button class="btn btn-s" onclick="go(1)">← Back</button><button class="btn btn-p" onclick="saveCrop()">Next →</button></div></div>
</div>
<div class="sec" id="sec3">
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
<div class="btng"><button class="btn btn-s" onclick="go(S.manual?2:1)">← Back</button><button class="btn btn-p" id="skipbtn" onclick="skipAndDownload()">⬇️ Skip & Download</button><button class="btn btn-ok" id="procbtn" onclick="process()">🚀 Remove Background</button></div>
<div class="progc" id="progc"><div class="progbg"><div class="prog" id="prog"></div></div><div class="progt" id="progt">Processing...</div></div>
</div>
<div class="res" id="res">
<div class="resok">✨ Background removed successfully</div>
<div class="resprev imgframe"><img id="resimg" src="" alt=""></div>
<div class="btng"><button class="btn btn-s" onclick="startOver()">Start Over</button><button class="btn btn-ok" onclick="dl()">⬇️ Download PNG</button></div>
</div>
</div>
</div>
<div class="how-it-works">
<h3>How it Works</h3>
<p class="hiw-sub">Professional passport photos in four simple steps</p>
<div class="hiw-steps">
<div class="hiw-step"><div class="hiw-num">1</div><div class="hiw-line"></div><div class="hiw-title">Upload Photo</div><div class="hiw-desc">Drop any photo from your phone or computer</div></div>
<div class="hiw-step"><div class="hiw-num">2</div><div class="hiw-line"></div><div class="hiw-title">Select Size</div><div class="hiw-desc">Choose US, EU passport or custom dimensions</div></div>
<div class="hiw-step"><div class="hiw-num">3</div><div class="hiw-line"></div><div class="hiw-title">AI Processing</div><div class="hiw-desc">Face detection, alignment, and background removal</div></div>
<div class="hiw-step"><div class="hiw-num">4</div><div class="hiw-title">Download</div><div class="hiw-desc">Get your professional photo instantly - free</div></div>
</div>
</div>
</div>
<footer class="footer">© 2026 Passport Photo Editor<span>·</span><a href="/privacy-policy">Privacy</a><span>·</span><a href="/terms-of-service">Terms</a><span>·</span>Made with ❤️</footer>
<script>
let S={step:1,sid:null,img:null,sz:'passport_us',col:'#ffffff',iw:0,ih:0,sc:1,ox:0,oy:0,tw:600,th:600,manual:false};

function toggleAuto(){S.manual=!S.manual;document.getElementById('autoToggle').classList.toggle('on',S.manual);
if(S.sid){document.getElementById('s1btn').textContent=S.manual?'Next →':'✨ Generate Photo'}}

function selSzDrop(sz){S.sz=sz;
const szs={passport_us:[600,600],passport_eu:[413,531],passport_uk:[413,531],passport_canada:[591,827],passport_india:[600,600],passport_china:[390,567],passport_40x50:[472,591],passport_35x35:[413,413],passport_30x40:[354,472],visa_australia:[413,531],visa_japan:[413,531],visa_brazil:[591,827],visa_saudi:[472,709],visa_45x45:[531,531],visa_47x47:[555,555],visa_50x50:[591,591],linkedin:[400,400],square_1000:[1000,1000]};
if(szs[sz]){S.tw=szs[sz][0];S.th=szs[sz][1]}else if(sz==='original'){S.tw=S.iw;S.th=S.ih}}

async function processFromUpload(){
if(!S.sid){err('Upload an image first');return}
if(S.manual){go(2);return}
hide();document.getElementById('s1btn').disabled=true;document.getElementById('s1btn').textContent='Generating...';
const progc=document.createElement('div');progc.className='gen-prog vis';progc.innerHTML='<div class="gen-bar"><div class="gen-fill" id="gen-fill"></div></div><div class="gen-status" id="gen-status">Analyzing photo...</div>';
document.getElementById('prev').appendChild(progc);
setTimeout(()=>{document.getElementById('gen-fill').style.width='25%';document.getElementById('gen-status').textContent='Detecting face...'},300);
setTimeout(()=>{document.getElementById('gen-fill').style.width='50%';document.getElementById('gen-status').textContent='Cropping to size...'},800);
const r=await fetch('/auto-process',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,size:S.sz,background_color:S.col})});
document.getElementById('gen-fill').style.width='85%';document.getElementById('gen-status').textContent='Removing background...';
await new Promise(r=>setTimeout(r,400));document.getElementById('gen-fill').style.width='100%';document.getElementById('gen-status').textContent='Done!';
const d=await r.json();
if(d.success){document.getElementById('resimg').src=d.image;S.sheetCount=d.sheet_count;S.step=3;updSteps();
document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
document.getElementById('sec3').classList.add('active');
document.getElementById('proc').style.display='none';document.getElementById('res').classList.add('vis');
document.getElementById('res').querySelector('.btng').innerHTML='<button class="btn btn-s" onclick="startOver()">Start Over</button><button class="btn btn-p" onclick="dl()">⬇️ Single Photo</button><button class="btn btn-ok" onclick="dlSheet()">🖨️ 4×6 Print Sheet ('+d.sheet_count+' photos)</button>'}
else{err(d.error);document.getElementById('s1btn').disabled=false;document.getElementById('s1btn').textContent='✨ Generate Photo';progc.remove()}}
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
upz.style.display='none';document.getElementById('prev').classList.add('vis');
document.getElementById('s1btn').textContent=S.manual?'Next →':'✨ Generate Photo'}
else err(d.error)}

function reset(){upz.style.display='block';document.getElementById('prev').classList.remove('vis');fi.value='';S.sid=null}


function selCol(c){S.col=c;document.querySelectorAll('.col').forEach(o=>o.classList.remove('sel'));
document.querySelector('[data-c="'+c+'"]').classList.add('sel')}

function initCanvas(){
cv=document.getElementById('canvas');ctx=cv.getContext('2d');
limg=new Image();limg.onload=()=>resetPos();limg.src=S.img;
const mw=Math.min(400,window.innerWidth-80),r=S.tw/S.th;
cv.width=mw;cv.height=mw/r;
// Show silhouette guide for passport sizes only
const sil=document.getElementById('silhouette');
const leg=document.getElementById('guideLegend');
if(['passport_us','passport_eu'].includes(S.sz)){sil.classList.add('vis');leg.classList.add('vis')}else{sil.classList.remove('vis');leg.classList.remove('vis')}
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
function zin(){zoomBy(1)}
function zout(){zoomBy(-1)}
function zoomBy(d){const os=S.sc;S.sc=Math.max(0.1,Math.min(2,S.sc+d/100));
const cx=cv.width/2,cy=cv.height/2;S.ox=cx-(cx-S.ox)*(S.sc/os);S.oy=cy-(cy-S.oy)*(S.sc/os);
document.getElementById('zslide').value=S.sc*100;updZ();draw()}
function updZ(){document.getElementById('zlbl').textContent=Math.round(S.sc*100)+'%'}
function draw(){ctx.fillStyle='#1a1a1a';ctx.fillRect(0,0,cv.width,cv.height);
if(limg.complete)ctx.drawImage(limg,S.ox,S.oy,S.iw*S.sc,S.ih*S.sc)}

async function saveCrop(){
await fetch('/set-crop',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,crop_settings:{scale:S.sc,offsetX:S.ox,offsetY:S.oy,canvasW:cv.width,canvasH:cv.height}})});
go(3)}

async function process(){
hide();document.getElementById('procbtn').disabled=true;
document.getElementById('progc').classList.add('vis');
const pb=document.getElementById('prog');let p=0;
const iv=setInterval(()=>{p=Math.min(90,p+Math.random()*15);pb.style.width=p+'%'},500);
await fetch('/set-size',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,size:S.sz})});
document.getElementById('progt').textContent='Removing background...';
const r=await fetch('/remove-background',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,background_color:S.col==='transparent'?null:S.col})});
const d=await r.json();clearInterval(iv);pb.style.width='100%';
if(d.success){document.getElementById('resimg').src=d.image;
setTimeout(()=>{document.getElementById('proc').style.display='none';document.getElementById('res').classList.add('vis')},400)}
else{err(d.error);document.getElementById('procbtn').disabled=false;document.getElementById('progc').classList.remove('vis')}}

function dl(){if(S.sid)window.location.href='/download/'+S.sid}
function dlSheet(){if(S.sid)window.location.href='/download-sheet/'+S.sid}
function dlOrig(){if(S.sid)window.location.href='/download-original/'+S.sid}

async function skipAndDownload(){
hide();document.getElementById('skipbtn').disabled=true;
document.getElementById('progc').classList.add('vis');
document.getElementById('progt').textContent='Processing image...';
const pb=document.getElementById('prog');pb.style.width='50%';
await fetch('/set-size',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid,size:S.sz})});
const r=await fetch('/download-cropped',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({session_id:S.sid})});
const d=await r.json();pb.style.width='100%';
if(d.success){document.getElementById('resimg').src=d.image;
document.getElementById('res').querySelector('.resok').innerHTML='✅ Image ready (original background)';
document.getElementById('res').querySelector('.btng').innerHTML='<button class="btn btn-s" onclick="startOver()">Start Over</button><button class="btn btn-ok" onclick="dlOrig()">⬇️ Download JPG</button>';
setTimeout(()=>{document.getElementById('proc').style.display='none';document.getElementById('res').classList.add('vis')},400)}
else{err(d.error);document.getElementById('skipbtn').disabled=false;document.getElementById('progc').classList.remove('vis')}}

function go(n){
if(n===2&&!S.sid){err('Upload an image first');return}
S.step=n;updSteps();
document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
document.getElementById('sec'+n).classList.add('active');
if(n===2)initCanvas()}

function updSteps(){for(let i=1;i<=3;i++){
const s=document.getElementById('s'+i);s.classList.remove('active','done');
if(i<S.step){s.classList.add('done');s.onclick=()=>go(i)}
else if(i===S.step)s.classList.add('active')}}

function startOver(){
S={step:1,sid:null,img:null,sz:'passport_us',col:'#ffffff',iw:0,ih:0,sc:1,ox:0,oy:0,tw:600,th:600,manual:false};
document.getElementById('autoToggle').classList.remove('on');
document.getElementById('szsel').value='passport_us';
// Clean up auto-mode progress bar if exists
const gp=document.querySelector('.gen-prog');if(gp)gp.remove();
// Reset s1btn
document.getElementById('s1btn').disabled=false;
document.getElementById('s1btn').textContent='✨ Generate Photo';
reset();
document.getElementById('proc').style.display='block';document.getElementById('res').classList.remove('vis');
document.getElementById('procbtn').disabled=false;go(1)}

function err(m){document.getElementById('errtxt').textContent=m;document.getElementById('err').classList.add('vis')}
function hide(){document.getElementById('err').classList.remove('vis')}
</script>
</body>
</html>
'''

ABOUT_TEMPLATE = '''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%236366f1'/%3E%3Crect x='6' y='8' width='20' height='16' rx='2' fill='%23fff'/%3E%3Ccircle cx='16' cy='16' r='5' fill='%236366f1'/%3E%3Ccircle cx='16' cy='16' r='3' fill='%23818cf8'/%3E%3Crect x='20' y='10' width='4' height='2' rx='1' fill='%236366f1'/%3E%3C/svg%3E">
<title>About - Passport Photo Editor</title>
<meta name="description" content="Learn about Passport Photo Editor - why we built it and our mission to provide free passport photos.">
<meta name="robots" content="index, follow">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--t:0.2s ease}
[data-theme="dark"]{--bg:#09090b;--bg2:#18181b;--bg3:#27272a;--tx:#fafafa;--tx2:#a1a1aa;--tx3:#71717a;--bd:#27272a;--ac:#6366f1;--ac2:#818cf8;--sh:0 4px 12px rgba(0,0,0,0.4)}
[data-theme="light"]{--bg:#fff;--bg2:#fafafa;--bg3:#f4f4f5;--tx:#18181b;--tx2:#52525b;--tx3:#a1a1aa;--bd:#e4e4e7;--ac:#6366f1;--ac2:#4f46e5;--sh:0 4px 12px rgba(0,0,0,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;transition:background var(--t),color var(--t);display:flex;flex-direction:column}
.navbar{position:sticky;top:0;z-index:100;display:flex;justify-content:space-between;align-items:center;padding:12px 24px;background:var(--bg);border-bottom:1px solid var(--bd);backdrop-filter:blur(8px)}
.nav-left{display:flex;align-items:center;gap:10px}
.nav-left a{text-decoration:none;display:flex;align-items:center;gap:10px}
.nav-logo{width:32px;height:32px;background:var(--ac);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}
.nav-brand{font-weight:600;font-size:0.95rem;color:var(--tx)}
.nav-right{display:flex;align-items:center;gap:20px}
.nav-link{color:var(--tx2);text-decoration:none;font-size:0.85rem;font-weight:500;transition:color var(--t)}
.nav-link:hover{color:var(--ac)}
.nav-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:linear-gradient(135deg,#ec4899,#f472b6);color:#fff;text-decoration:none;border-radius:20px;font-size:0.8rem;font-weight:500;transition:all var(--t)}
.nav-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(236,72,153,0.3)}
.thm{width:44px;height:24px;background:var(--bg3);border-radius:12px;cursor:pointer;position:relative;border:1px solid var(--bd)}
.thm::after{content:'';position:absolute;width:18px;height:18px;background:var(--tx);border-radius:50%;top:2px;left:2px;transition:transform var(--t)}
[data-theme="light"] .thm::after{transform:translateX(20px)}
.content{max-width:720px;margin:0 auto;padding:60px 24px;flex:1;width:100%}
.about-card{background:var(--bg2);border:1px solid var(--bd);border-radius:16px;padding:48px;box-shadow:var(--sh)}
.about-card h1{font-size:1.75rem;font-weight:700;margin-bottom:24px;background:linear-gradient(135deg,var(--ac),var(--ac2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center}
.about-story h3{font-size:1.1rem;font-weight:600;margin-bottom:16px;color:var(--tx)}
.about-story p{color:var(--tx2);line-height:1.8;margin-bottom:16px;font-size:0.95rem}
.about-story strong{color:var(--tx);font-weight:600}
.about-cta{margin-top:32px;text-align:center}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 24px;border:none;border-radius:10px;font-size:0.9rem;font-weight:500;cursor:pointer;transition:all var(--t);font-family:inherit;text-decoration:none}
.btn-p{background:var(--ac);color:#fff}
.btn-p:hover{background:var(--ac2);transform:translateY(-1px);box-shadow:var(--sh)}
.footer{padding:16px 24px;text-align:center;font-size:0.75rem;color:var(--tx3);border-top:1px solid var(--bd)}
.footer span{margin:0 6px;opacity:0.5}
.footer a{color:var(--tx3);text-decoration:none}
.footer a:hover{color:var(--ac)}
@media(max-width:640px){.content{padding:32px 16px}.about-card{padding:32px 24px}}
</style>
</head>
<body>
<nav class="navbar">
<div class="nav-left"><a href="/"><div class="nav-logo">📷</div><span class="nav-brand">Passport Photo Editor</span></a></div>
<div class="nav-right">
<a href="/about" class="nav-link">About</a>
<a href="/contact" class="nav-link">Contact</a>
<a href="https://buymeacoffee.com/onestopshoppassportphotos" target="_blank" class="nav-btn">☕ Buy me a coffee</a>
<div class="thm" onclick="toggleTheme()"></div>
</div>
</nav>
<div class="content">
<div class="about-card">
<h1>About</h1>
<div class="about-story">
<h3>Why I Built This</h3>
<p>It all started with my 2-month-old son needing a passport photo.</p>
<p>Like most parents, we really didn't want to bundle up a newborn and drag him to Walgreens just for a simple photo. So I searched for an online solution — something that could resize to official specs and remove the background to white.</p>
<p><strong>What I found? Nothing worked well.</strong></p>
<p>Every app I tried either couldn't handle the size requirements, did a terrible job with background removal, or required multiple tools stitched together. None of them gave me a clean, compliant passport photo in one simple flow.</p>
<p>So I built this.</p>
<p>I'm a software engineer, and I created this tool as a side project for my own use. It combines proper sizing (US, EU passport specs), AI-powered background removal, and an intuitive interface — all in one place.</p>
<p>I shared it with friends who had the same frustration. They loved it. So now I'm making it available to everyone.</p>
<p><strong>No trips to the store. No expensive fees. No hassle.</strong></p>
<p>Just upload your photo, position it, and download — done.</p>
</div>
<div class="about-cta">
<a href="/" class="btn btn-p">Try It Free →</a>
</div>
</div>
</div>
<footer class="footer">© 2026 Passport Photo Editor<span>·</span><a href="/privacy-policy">Privacy</a><span>·</span><a href="/terms-of-service">Terms</a><span>·</span>Made with ❤️</footer>
<script>
function toggleTheme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark'}
</script>
</body>
</html>
'''

CONTACT_TEMPLATE = '''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%236366f1'/%3E%3Crect x='6' y='8' width='20' height='16' rx='2' fill='%23fff'/%3E%3Ccircle cx='16' cy='16' r='5' fill='%236366f1'/%3E%3Ccircle cx='16' cy='16' r='3' fill='%23818cf8'/%3E%3Crect x='20' y='10' width='4' height='2' rx='1' fill='%236366f1'/%3E%3C/svg%3E">
<title>Contact - Passport Photo Editor</title>
<meta name="description" content="Contact Passport Photo Editor. Get in touch with questions or feedback.">
<meta name="robots" content="index, follow">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--t:0.2s ease}
[data-theme="dark"]{--bg:#09090b;--bg2:#18181b;--bg3:#27272a;--tx:#fafafa;--tx2:#a1a1aa;--tx3:#71717a;--bd:#27272a;--ac:#6366f1;--ac2:#818cf8;--sh:0 4px 12px rgba(0,0,0,0.4)}
[data-theme="light"]{--bg:#fff;--bg2:#fafafa;--bg3:#f4f4f5;--tx:#18181b;--tx2:#52525b;--tx3:#a1a1aa;--bd:#e4e4e7;--ac:#6366f1;--ac2:#4f46e5;--sh:0 4px 12px rgba(0,0,0,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;transition:background var(--t),color var(--t);display:flex;flex-direction:column}
.navbar{position:sticky;top:0;z-index:100;display:flex;justify-content:space-between;align-items:center;padding:12px 24px;background:var(--bg);border-bottom:1px solid var(--bd);backdrop-filter:blur(8px)}
.nav-left{display:flex;align-items:center;gap:10px}
.nav-left a{text-decoration:none;display:flex;align-items:center;gap:10px}
.nav-logo{width:32px;height:32px;background:var(--ac);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}
.nav-brand{font-weight:600;font-size:0.95rem;color:var(--tx)}
.nav-right{display:flex;align-items:center;gap:20px}
.nav-link{color:var(--tx2);text-decoration:none;font-size:0.85rem;font-weight:500;transition:color var(--t)}
.nav-link:hover{color:var(--ac)}
.nav-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:linear-gradient(135deg,#ec4899,#f472b6);color:#fff;text-decoration:none;border-radius:20px;font-size:0.8rem;font-weight:500;transition:all var(--t)}
.nav-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(236,72,153,0.3)}
.thm{width:44px;height:24px;background:var(--bg3);border-radius:12px;cursor:pointer;position:relative;border:1px solid var(--bd)}
.thm::after{content:'';position:absolute;width:18px;height:18px;background:var(--tx);border-radius:50%;top:2px;left:2px;transition:transform var(--t)}
[data-theme="light"] .thm::after{transform:translateX(20px)}
.content{max-width:720px;margin:0 auto;padding:60px 24px;flex:1;width:100%}
.contact-card{background:var(--bg2);border:1px solid var(--bd);border-radius:16px;padding:48px;box-shadow:var(--sh)}
.contact-card h1{font-size:1.75rem;font-weight:700;margin-bottom:24px;background:linear-gradient(135deg,var(--ac),var(--ac2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center}
.contact-card p{color:var(--tx2);line-height:1.8;margin-bottom:16px;font-size:0.95rem;text-align:center}
.contact-card a{color:var(--ac);text-decoration:none}
.contact-card a:hover{text-decoration:underline}
.contact-cta{margin-top:32px;text-align:center}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 24px;border:none;border-radius:10px;font-size:0.9rem;font-weight:500;cursor:pointer;transition:all var(--t);font-family:inherit;text-decoration:none}
.btn-p{background:var(--ac);color:#fff}
.btn-p:hover{background:var(--ac2);transform:translateY(-1px);box-shadow:var(--sh)}
.footer{padding:16px 24px;text-align:center;font-size:0.75rem;color:var(--tx3);border-top:1px solid var(--bd)}
.footer span{margin:0 6px;opacity:0.5}
.footer a{color:var(--tx3);text-decoration:none}
.footer a:hover{color:var(--ac)}
@media(max-width:640px){.content{padding:32px 16px}.contact-card{padding:32px 24px}}
</style>
</head>
<body>
<nav class="navbar">
<div class="nav-left"><a href="/"><div class="nav-logo">📷</div><span class="nav-brand">Passport Photo Editor</span></a></div>
<div class="nav-right">
<a href="/about" class="nav-link">About</a>
<a href="/contact" class="nav-link">Contact</a>
<a href="https://buymeacoffee.com/onestopshoppassportphotos" target="_blank" class="nav-btn">☕ Buy me a coffee</a>
<div class="thm" onclick="toggleTheme()"></div>
</div>
</nav>
<div class="content">
<div class="contact-card">
<h1>Contact</h1>
<p>Have questions or feedback? I'd love to hear from you.</p>
<p>Email: <a href="mailto:hello@passportphotoeditor.com">hello@passportphotoeditor.com</a></p>
<div class="contact-cta">
<a href="/" class="btn btn-p">← Back to Home</a>
</div>
</div>
</div>
<footer class="footer">© 2026 Passport Photo Editor<span>·</span><a href="/privacy-policy">Privacy</a><span>·</span><a href="/terms-of-service">Terms</a><span>·</span>Made with ❤️</footer>
<script>
function toggleTheme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark'}
</script>
</body>
</html>
'''

PRIVACY_TEMPLATE = '''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%236366f1'/%3E%3Crect x='6' y='8' width='20' height='16' rx='2' fill='%23fff'/%3E%3Ccircle cx='16' cy='16' r='5' fill='%236366f1'/%3E%3Ccircle cx='16' cy='16' r='3' fill='%23818cf8'/%3E%3Crect x='20' y='10' width='4' height='2' rx='1' fill='%236366f1'/%3E%3C/svg%3E">
<title>Privacy Policy - Passport Photo Editor</title>
<meta name="description" content="Privacy Policy for Passport Photo Editor. Learn how we handle your data and protect your privacy.">
<meta name="robots" content="index, follow">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--t:0.2s ease}
[data-theme="dark"]{--bg:#09090b;--bg2:#18181b;--bg3:#27272a;--tx:#fafafa;--tx2:#a1a1aa;--tx3:#71717a;--bd:#27272a;--ac:#6366f1;--ac2:#818cf8;--sh:0 4px 12px rgba(0,0,0,0.4)}
[data-theme="light"]{--bg:#fff;--bg2:#fafafa;--bg3:#f4f4f5;--tx:#18181b;--tx2:#52525b;--tx3:#a1a1aa;--bd:#e4e4e7;--ac:#6366f1;--ac2:#4f46e5;--sh:0 4px 12px rgba(0,0,0,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;transition:background var(--t),color var(--t);display:flex;flex-direction:column}
.navbar{position:sticky;top:0;z-index:100;display:flex;justify-content:space-between;align-items:center;padding:12px 24px;background:var(--bg);border-bottom:1px solid var(--bd);backdrop-filter:blur(8px)}
.nav-left{display:flex;align-items:center;gap:10px}
.nav-left a{text-decoration:none;display:flex;align-items:center;gap:10px}
.nav-logo{width:32px;height:32px;background:var(--ac);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}
.nav-brand{font-weight:600;font-size:0.95rem;color:var(--tx)}
.nav-right{display:flex;align-items:center;gap:20px}
.nav-link{color:var(--tx2);text-decoration:none;font-size:0.85rem;font-weight:500;transition:color var(--t)}
.nav-link:hover{color:var(--ac)}
.nav-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:linear-gradient(135deg,#ec4899,#f472b6);color:#fff;text-decoration:none;border-radius:20px;font-size:0.8rem;font-weight:500;transition:all var(--t)}
.nav-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(236,72,153,0.3)}
.thm{width:44px;height:24px;background:var(--bg3);border-radius:12px;cursor:pointer;position:relative;border:1px solid var(--bd)}
.thm::after{content:'';position:absolute;width:18px;height:18px;background:var(--tx);border-radius:50%;top:2px;left:2px;transition:transform var(--t)}
[data-theme="light"] .thm::after{transform:translateX(20px)}
.content{max-width:720px;margin:0 auto;padding:60px 24px;flex:1;width:100%}
.policy-card{background:var(--bg2);border:1px solid var(--bd);border-radius:16px;padding:48px;box-shadow:var(--sh)}
.policy-card h1{font-size:1.75rem;font-weight:700;margin-bottom:8px;background:linear-gradient(135deg,var(--ac),var(--ac2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center}
.effective-date{text-align:center;color:var(--tx3);font-size:0.875rem;margin-bottom:32px}
.policy-intro{color:var(--tx2);line-height:1.8;margin-bottom:32px;font-size:0.95rem}
.policy-section{margin-bottom:28px}
.policy-section h2{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:var(--tx)}
.policy-section p{color:var(--tx2);line-height:1.8;margin-bottom:12px;font-size:0.95rem}
.policy-section ul{color:var(--tx2);line-height:1.8;margin-left:24px;margin-bottom:12px;font-size:0.95rem}
.policy-section li{margin-bottom:6px}
.policy-section a{color:var(--ac);text-decoration:none}
.policy-section a:hover{text-decoration:underline}
.policy-footer{margin-top:40px;padding-top:24px;border-top:1px solid var(--bd);text-align:center}
.policy-footer p{color:var(--tx3);font-size:0.875rem;margin-bottom:16px}
.policy-footer a{color:var(--ac);text-decoration:none;font-weight:500}
.policy-footer a:hover{text-decoration:underline}
.footer{padding:16px 24px;text-align:center;font-size:0.75rem;color:var(--tx3);border-top:1px solid var(--bd)}
.footer span{margin:0 6px;opacity:0.5}
.footer a{color:var(--tx3);text-decoration:none}
.footer a:hover{color:var(--ac)}
@media(max-width:640px){.content{padding:32px 16px}.policy-card{padding:32px 24px}}
</style>
</head>
<body>
<nav class="navbar">
<div class="nav-left"><a href="/"><div class="nav-logo">📷</div><span class="nav-brand">Passport Photo Editor</span></a></div>
<div class="nav-right">
<a href="/about" class="nav-link">About</a>
<a href="/contact" class="nav-link">Contact</a>
<a href="https://buymeacoffee.com/onestopshoppassportphotos" target="_blank" class="nav-btn">☕ Buy me a coffee</a>
<div class="thm" onclick="toggleTheme()"></div>
</div>
</nav>
<div class="content">
<div class="policy-card">
<h1>Privacy Policy</h1>
<p class="effective-date"><strong>Effective Date:</strong> February 18, 2026</p>

<p class="policy-intro">This Privacy Policy describes how Passport Photo Editor collects, uses, and shares information when you use this website. I value your privacy and am committed to protecting your personal information.</p>

<div class="policy-section">
<h2>1. Information Collection</h2>
<p><strong>Photo Processing:</strong> All photo processing happens directly in your browser and on our secure servers during your session. Your uploaded images are temporarily stored only for the duration of your editing session and are automatically deleted when you close your browser or start a new session. We do not permanently store, collect, or transmit your photos to any third parties.</p>
</div>

<div class="policy-section">
<h2>2. Analytics and Cookies</h2>
<p>I may use analytics tools to understand how visitors interact with the website. This helps me improve the user experience and functionality. Analytics may collect:</p>
<ul>
<li>IP addresses (anonymized)</li>
<li>Browser type and version</li>
<li>Device information</li>
<li>Page visit duration</li>
<li>Pages visited</li>
<li>Referring websites</li>
</ul>
</div>

<div class="policy-section">
<h2>3. Data Security</h2>
<p>I take reasonable measures to help protect information about you from loss, theft, misuse and unauthorized access, disclosure, alteration and destruction. However, as no Internet transmission is ever fully secure, I cannot guarantee the security of your information.</p>
</div>

<div class="policy-section">
<h2>4. Children's Privacy</h2>
<p>This website is not intended for children under 13 years of age. I do not knowingly collect personal information from children under 13.</p>
</div>

<div class="policy-section">
<h2>5. Third-Party Services</h2>
<p>This website may contain links to third-party websites or services. I am not responsible for the privacy practices of these third parties. I encourage you to read the privacy policies of any third-party services you use.</p>
</div>

<div class="policy-section">
<h2>6. Changes to This Privacy Policy</h2>
<p>I may update this Privacy Policy from time to time. I will notify you of any changes by posting the new Privacy Policy on this page and updating the effective date at the top of this page.</p>
</div>

<div class="policy-section">
<h2>7. Contact Information</h2>
<p>If you have any questions or concerns about this Privacy Policy, please visit the <a href="/#contact">Contact Page</a> to get in touch with me.</p>
</div>

<div class="policy-footer">
<p>This Privacy Policy should be read in conjunction with our <a href="/terms-of-service">Terms of Service</a>.</p>
<a href="/">Return to Home</a>
</div>
</div>
</div>
<footer class="footer">© 2026 Passport Photo Editor<span>·</span><a href="/privacy-policy">Privacy</a><span>·</span><a href="/terms-of-service">Terms</a><span>·</span>Made with ❤️</footer>
<script>
function toggleTheme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark'}
</script>
</body>
</html>
'''

TERMS_TEMPLATE = '''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%236366f1'/%3E%3Crect x='6' y='8' width='20' height='16' rx='2' fill='%23fff'/%3E%3Ccircle cx='16' cy='16' r='5' fill='%236366f1'/%3E%3Ccircle cx='16' cy='16' r='3' fill='%23818cf8'/%3E%3Crect x='20' y='10' width='4' height='2' rx='1' fill='%236366f1'/%3E%3C/svg%3E">
<title>Terms of Service - Passport Photo Editor</title>
<meta name="description" content="Terms of Service for Passport Photo Editor. Read our terms and conditions for using the service.">
<meta name="robots" content="index, follow">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--t:0.2s ease}
[data-theme="dark"]{--bg:#09090b;--bg2:#18181b;--bg3:#27272a;--tx:#fafafa;--tx2:#a1a1aa;--tx3:#71717a;--bd:#27272a;--ac:#6366f1;--ac2:#818cf8;--sh:0 4px 12px rgba(0,0,0,0.4)}
[data-theme="light"]{--bg:#fff;--bg2:#fafafa;--bg3:#f4f4f5;--tx:#18181b;--tx2:#52525b;--tx3:#a1a1aa;--bd:#e4e4e7;--ac:#6366f1;--ac2:#4f46e5;--sh:0 4px 12px rgba(0,0,0,0.08)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;transition:background var(--t),color var(--t);display:flex;flex-direction:column}
.navbar{position:sticky;top:0;z-index:100;display:flex;justify-content:space-between;align-items:center;padding:12px 24px;background:var(--bg);border-bottom:1px solid var(--bd);backdrop-filter:blur(8px)}
.nav-left{display:flex;align-items:center;gap:10px}
.nav-left a{text-decoration:none;display:flex;align-items:center;gap:10px}
.nav-logo{width:32px;height:32px;background:var(--ac);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px}
.nav-brand{font-weight:600;font-size:0.95rem;color:var(--tx)}
.nav-right{display:flex;align-items:center;gap:20px}
.nav-link{color:var(--tx2);text-decoration:none;font-size:0.85rem;font-weight:500;transition:color var(--t)}
.nav-link:hover{color:var(--ac)}
.nav-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;background:linear-gradient(135deg,#ec4899,#f472b6);color:#fff;text-decoration:none;border-radius:20px;font-size:0.8rem;font-weight:500;transition:all var(--t)}
.nav-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(236,72,153,0.3)}
.thm{width:44px;height:24px;background:var(--bg3);border-radius:12px;cursor:pointer;position:relative;border:1px solid var(--bd)}
.thm::after{content:'';position:absolute;width:18px;height:18px;background:var(--tx);border-radius:50%;top:2px;left:2px;transition:transform var(--t)}
[data-theme="light"] .thm::after{transform:translateX(20px)}
.content{max-width:720px;margin:0 auto;padding:60px 24px;flex:1;width:100%}
.policy-card{background:var(--bg2);border:1px solid var(--bd);border-radius:16px;padding:48px;box-shadow:var(--sh)}
.policy-card h1{font-size:1.75rem;font-weight:700;margin-bottom:8px;background:linear-gradient(135deg,var(--ac),var(--ac2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center}
.effective-date{text-align:center;color:var(--tx3);font-size:0.875rem;margin-bottom:32px}
.policy-intro{color:var(--tx2);line-height:1.8;margin-bottom:32px;font-size:0.95rem}
.policy-section{margin-bottom:28px}
.policy-section h2{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:var(--tx)}
.policy-section p{color:var(--tx2);line-height:1.8;margin-bottom:12px;font-size:0.95rem}
.policy-section ul{color:var(--tx2);line-height:1.8;margin-left:24px;margin-bottom:12px;font-size:0.95rem}
.policy-section li{margin-bottom:6px}
.policy-section a{color:var(--ac);text-decoration:none}
.policy-section a:hover{text-decoration:underline}
.policy-footer{margin-top:40px;padding-top:24px;border-top:1px solid var(--bd);text-align:center}
.policy-footer p{color:var(--tx3);font-size:0.875rem;margin-bottom:16px}
.policy-footer a{color:var(--ac);text-decoration:none;font-weight:500}
.policy-footer a:hover{text-decoration:underline}
.agreement-note{background:var(--bg3);border-radius:10px;padding:16px 20px;margin-top:32px;text-align:center}
.agreement-note p{color:var(--tx2);font-size:0.9rem;margin:0}
.footer{padding:16px 24px;text-align:center;font-size:0.75rem;color:var(--tx3);border-top:1px solid var(--bd)}
.footer span{margin:0 6px;opacity:0.5}
.footer a{color:var(--tx3);text-decoration:none}
.footer a:hover{color:var(--ac)}
@media(max-width:640px){.content{padding:32px 16px}.policy-card{padding:32px 24px}}
</style>
</head>
<body>
<nav class="navbar">
<div class="nav-left"><a href="/"><div class="nav-logo">📷</div><span class="nav-brand">Passport Photo Editor</span></a></div>
<div class="nav-right">
<a href="/about" class="nav-link">About</a>
<a href="/contact" class="nav-link">Contact</a>
<a href="https://buymeacoffee.com/onestopshoppassportphotos" target="_blank" class="nav-btn">☕ Buy me a coffee</a>
<div class="thm" onclick="toggleTheme()"></div>
</div>
</nav>
<div class="content">
<div class="policy-card">
<h1>Terms of Service</h1>
<p class="effective-date"><strong>Effective Date:</strong> February 18, 2026</p>

<p class="policy-intro">Welcome to Passport Photo Editor. These Terms of Service govern your use of our website and services. By accessing or using our service, you agree to these terms.</p>

<div class="policy-section">
<h2>1. Service Description</h2>
<p>Passport Photo Editor offers an online tool for creating passport and visa photos that meet various country requirements. Our service processes your photos using AI-powered background removal and allows you to resize images to official specifications.</p>
</div>

<div class="policy-section">
<h2>2. User Responsibilities</h2>
<ul>
<li><strong>Verification:</strong> You are responsible for verifying that the output photos meet your specific country's requirements. Photo specifications can vary and may change over time.</li>
<li><strong>No Guarantees:</strong> While we strive to meet passport and visa photo standards, we cannot guarantee approval from any government authority. Always verify requirements with the appropriate government agency.</li>
<li><strong>Appropriate Use:</strong> You agree to use our service only for legitimate passport and visa photo creation, and not for any fraudulent or illegal purposes.</li>
</ul>
</div>

<div class="policy-section">
<h2>3. Privacy and Data Security</h2>
<ul>
<li><strong>Temporary Processing:</strong> Your photos are processed on our servers only during your active session. We do not permanently store your photos.</li>
<li><strong>No Data Collection:</strong> We do not collect personal data through the photo processing service beyond what is necessary to provide the service.</li>
<li>For complete information about our data practices, please see our <a href="/privacy-policy">Privacy Policy</a>.</li>
</ul>
</div>

<div class="policy-section">
<h2>4. Service Changes and Limitations</h2>
<ul>
<li><strong>Changes to Service:</strong> We may modify, update, or discontinue our service at any time without notice.</li>
<li><strong>Technical Limitations:</strong> Our service performance depends on your device capabilities and internet connection. Results may vary based on the quality of uploaded photos.</li>
<li><strong>Availability:</strong> We do not guarantee uninterrupted access to the service and are not liable for any downtime or service interruptions.</li>
</ul>
</div>

<div class="policy-section">
<h2>5. Intellectual Property</h2>
<p>All content, features, and functionality of our service are owned by us and are protected by copyright, trademark, and other intellectual property laws. You retain ownership of your uploaded photos.</p>
</div>

<div class="policy-section">
<h2>6. Limitation of Liability</h2>
<p>To the fullest extent permitted by law, Passport Photo Editor shall not be liable for any indirect, incidental, special, consequential, or punitive damages, including but not limited to loss of profits, data, or other intangible losses resulting from your use of the service.</p>
</div>

<div class="policy-section">
<h2>7. Indemnification</h2>
<p>You agree to indemnify and hold harmless Passport Photo Editor from any claims, damages, losses, or expenses arising from your use of the service or violation of these terms.</p>
</div>

<div class="policy-section">
<h2>8. Contact Us</h2>
<p>If you have any questions about these Terms, please <a href="/#contact">contact us</a>.</p>
</div>

<div class="agreement-note">
<p>By using Passport Photo Editor, you acknowledge that you have read and understood these Terms of Service and agree to be bound by them.</p>
</div>

<div class="policy-footer">
<a href="/">Return to Home</a>
</div>
</div>
</div>
<footer class="footer">© 2026 Passport Photo Editor<span>·</span><a href="/privacy-policy">Privacy</a><span>·</span><a href="/terms-of-service">Terms</a><span>·</span>Made with ❤️</footer>
<script>
function toggleTheme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark'}
</script>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"\\n🚀 Passport Photo Editor Server\\n📍 http://localhost:{port}\\n")
    app.run(host='0.0.0.0', port=port, debug=False)
