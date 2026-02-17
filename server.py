#!/usr/bin/env python3
"""
Modern Photo Editing Server using InSPyReNet
High-quality background removal and passport photo editing with sleek interface
"""

import os
import io
import json
import base64
import tempfile
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model loading
_model = None
_model_loaded = False

# Try to import InSPyReNet (transparent-background package)
try:
    from transparent_background import Remover
    INSPYRENET_AVAILABLE = True
    logger.info("✅ InSPyReNet (transparent-background) available")
except ImportError as e:
    INSPYRENET_AVAILABLE = False
    logger.error(f"❌ InSPyReNet not available: {e}")
    logger.error("Install with: pip install transparent-background")

def get_model():
    """Load InSPyReNet model (lazy loading)"""
    global _model, _model_loaded
    
    if not INSPYRENET_AVAILABLE:
        raise ValueError("InSPyReNet not available")
    
    if not _model_loaded:
        logger.info("🚀 Loading InSPyReNet model (first use may take a moment)...")
        try:
            # Load with 'base' mode for highest quality
            _model = Remover(mode='base')  # Options: 'base', 'fast'
            _model_loaded = True
            logger.info("✅ InSPyReNet model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load InSPyReNet model: {e}")
            raise e
    
    return _model

@app.route('/')
def index():
    """Serve the main application"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'inspyrenet_available': INSPYRENET_AVAILABLE,
        'model_loaded': _model_loaded,
        'supported_formats': ['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        'max_image_size': '50MB',
        'processing_modes': ['high_quality', 'fast'] if INSPYRENET_AVAILABLE else []
    })

@app.route('/remove-background', methods=['POST'])
def remove_background():
    """Remove background from uploaded image using InSPyReNet"""
    if not INSPYRENET_AVAILABLE:
        return jsonify({
            'error': 'InSPyReNet not available',
            'message': 'Please install with: pip install transparent-background'
        }), 500

    try:
        # Get parameters
        quality_mode = request.form.get('quality', 'high_quality')  # high_quality or fast
        background_color = request.form.get('background_color')  # hex color like #ffffff
        
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': 'Unsupported file format',
                'message': f'Please use one of: {", ".join(allowed_extensions)}'
            }), 400

        # Read and validate image
        try:
            image_data = file.read()
            if len(image_data) > 50 * 1024 * 1024:  # 50MB limit
                return jsonify({
                    'error': 'File too large',
                    'message': 'Please use images smaller than 50MB'
                }), 400
                
            image = Image.open(io.BytesIO(image_data))
            logger.info(f"Processing image: {image.size[0]}x{image.size[1]} pixels, format: {image.format}")
            
        except Exception as e:
            return jsonify({
                'error': 'Invalid image file',
                'message': str(e)
            }), 400

        # Load InSPyReNet model
        try:
            model = get_model()
        except Exception as e:
            return jsonify({
                'error': 'Model loading failed',
                'message': str(e)
            }), 500

        # Process with InSPyReNet
        try:
            logger.info(f"Processing with InSPyReNet ({quality_mode} mode)...")
            
            # Convert image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process with transparent-background (InSPyReNet wrapper)
            result_image = model.process(image, type='rgba')
            
            # Apply background color if specified
            if background_color:
                try:
                    # Parse hex color (e.g., #ffffff)
                    if background_color.startswith('#'):
                        background_color = background_color[1:]
                    
                    r = int(background_color[0:2], 16)
                    g = int(background_color[2:4], 16) 
                    b = int(background_color[4:6], 16)
                    bg_color = (r, g, b)
                    
                    # Create background and composite
                    background = Image.new('RGB', result_image.size, bg_color)
                    background.paste(result_image, (0, 0), result_image)
                    result_image = background
                    
                except Exception as e:
                    logger.warning(f"Invalid background color {background_color}: {e}")

            # Save result to memory
            output = io.BytesIO()
            if result_image.mode == 'RGBA':
                result_image.save(output, format='PNG')
                mimetype = 'image/png'
            else:
                result_image.save(output, format='JPEG', quality=95)
                mimetype = 'image/jpeg'
            
            output.seek(0)
            
            logger.info("✅ Background removal completed successfully")
            
            # Return processed image
            return send_file(
                output,
                mimetype=mimetype,
                as_attachment=True,
                download_name=f"{Path(file.filename).stem}_no_bg.png"
            )
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return jsonify({
                'error': 'Processing failed',
                'message': str(e)
            }), 500

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/remove-background-base64', methods=['POST'])
def remove_background_base64():
    """Remove background from base64 encoded image"""
    if not INSPYRENET_AVAILABLE:
        return jsonify({
            'error': 'InSPyReNet not available',
            'message': 'Please install with: pip install transparent-background'
        }), 500

    try:
        data = request.get_json()
        
        # Get parameters
        quality_mode = data.get('quality', 'high_quality')
        background_color = data.get('background_color')  # hex color
        
        # Get base64 image data
        image_b64 = data.get('image')
        if not image_b64:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        try:
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400

        # Load and process with InSPyReNet
        model = get_model()
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process with InSPyReNet
        result_image = model.process(image, type='rgba')
        
        # Apply background color if specified
        if background_color:
            try:
                if background_color.startswith('#'):
                    background_color = background_color[1:]
                
                r = int(background_color[0:2], 16)
                g = int(background_color[2:4], 16) 
                b = int(background_color[4:6], 16)
                bg_color = (r, g, b)
                
                background = Image.new('RGB', result_image.size, bg_color)
                background.paste(result_image, (0, 0), result_image)
                result_image = background
                
            except Exception as e:
                logger.warning(f"Invalid background color: {e}")

        # Convert result to base64
        output = io.BytesIO()
        if result_image.mode == 'RGBA':
            result_image.save(output, format='PNG')
            format_type = 'png'
        else:
            result_image.save(output, format='JPEG', quality=95)
            format_type = 'jpeg'
        
        result_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/{format_type};base64,{result_b64}',
            'quality_mode': quality_mode
        })
        
    except Exception as e:
        logger.error(f"Base64 processing failed: {e}")
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500

@app.route('/create-passport-photo', methods=['POST'])
def create_passport_photo():
    """Create passport photo with background removal and 2x2 crop"""
    if not INSPYRENET_AVAILABLE:
        return jsonify({
            'error': 'InSPyReNet not available',
            'message': 'Please install with: pip install transparent-background'
        }), 500

    try:
        # Get parameters
        quality_mode = request.form.get('quality', 'high_quality')
        image_scale = float(request.form.get('scale', '1.0'))
        image_x = float(request.form.get('x', '0'))
        image_y = float(request.form.get('y', '0'))
        
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': 'Unsupported file format',
                'message': f'Please use one of: {", ".join(allowed_extensions)}'
            }), 400

        # Read and validate image
        try:
            image_data = file.read()
            if len(image_data) > 50 * 1024 * 1024:  # 50MB limit
                return jsonify({
                    'error': 'File too large',
                    'message': 'Please use images smaller than 50MB'
                }), 400
                
            original_image = Image.open(io.BytesIO(image_data))
            logger.info(f"Processing passport photo: {original_image.size[0]}x{original_image.size[1]} pixels")
            
        except Exception as e:
            return jsonify({
                'error': 'Invalid image file',
                'message': str(e)
            }), 400

        # Load InSPyReNet model
        try:
            model = get_model()
        except Exception as e:
            return jsonify({
                'error': 'Model loading failed',
                'message': str(e)
            }), 500

        # Process passport photo
        try:
            logger.info(f"Creating passport photo with InSPyReNet ({quality_mode} mode)...")
            
            # Convert to RGB if needed
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            # Apply positioning transform similar to frontend
            # Calculate the crop area based on the positioning parameters
            canvas_size = 300  # Canvas size from frontend
            target_size = 600  # Target 2x2 photo size at 300 DPI
            
            # Calculate scale factor from canvas to actual processing
            scale_factor = target_size / canvas_size
            
            # Apply transforms (scale, then translate)
            img_width, img_height = original_image.size
            
            # Scale the image
            new_width = int(img_width * image_scale * scale_factor)
            new_height = int(img_height * image_scale * scale_factor)
            scaled_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create 2x2 passport photo canvas (600x600 at 300 DPI)
            passport_canvas = Image.new('RGB', (target_size, target_size), 'white')
            
            # Calculate position to paste scaled image
            paste_x = int((target_size / 2) + (image_x * scale_factor) - (new_width / 2))
            paste_y = int((target_size / 2) + (image_y * scale_factor) - (new_height / 2))
            
            # Paste the scaled image onto the canvas
            passport_canvas.paste(scaled_image, (paste_x, paste_y))
            
            # Remove background using InSPyReNet
            result_image = model.process(passport_canvas, type='rgba')
            
            # Apply white background for passport photos
            final_image = Image.new('RGB', (target_size, target_size), 'white')
            final_image.paste(result_image, (0, 0), result_image)
            
            # Save result to memory
            output = io.BytesIO()
            final_image.save(output, format='JPEG', quality=95, dpi=(300, 300))
            output.seek(0)
            
            logger.info("✅ Passport photo created successfully")
            
            # Return processed passport photo
            return send_file(
                output,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=f"{Path(file.filename).stem}_passport_2x2.jpg"
            )
            
        except Exception as e:
            logger.error(f"❌ Passport photo processing failed: {e}")
            return jsonify({
                'error': 'Processing failed',
                'message': str(e)
            }), 500

    except Exception as e:
        logger.error(f"❌ Unexpected error in passport photo processing: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

# HTML Template for the main interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InSPyReNet Background Removal</title>
    <style>
        :root {
            --primary-color: #667eea;
            --primary-dark: #5a67d8;
            --secondary-color: #764ba2;
            --accent-color: #f093fb;
            --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            --shadow-hover: 0 30px 60px rgba(0, 0, 0, 0.15);
            --text-primary: #2d3748;
            --text-secondary: #718096;
            --border-radius: 20px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            line-height: 1.6;
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            padding: 40px;
            max-width: 600px;
            width: 100%;
            transition: var(--transition);
        }

        .container:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-5px);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .logo {
            width: 60px;
            height: 60px;
            background: var(--bg-gradient);
            border-radius: 15px;
            margin: 0 auto 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }

        h1 {
            color: var(--text-primary);
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            background: var(--bg-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 400;
        }

        .mode-selector {
            margin-bottom: 30px;
        }

        .mode-tabs {
            display: flex;
            background: rgba(247, 250, 252, 0.8);
            border-radius: 15px;
            padding: 6px;
            margin-bottom: 20px;
        }

        .mode-tab {
            flex: 1;
            background: transparent;
            border: none;
            padding: 12px 20px;
            border-radius: 12px;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-secondary);
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .mode-tab:hover {
            color: var(--primary-color);
            background: rgba(102, 126, 234, 0.1);
        }

        .mode-tab.active {
            background: white;
            color: var(--primary-color);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .mode-icon {
            font-size: 1.1rem;
        }

        /* Top Processing Indicator */
        .top-processing-indicator {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.95) 0%, rgba(16, 185, 129, 0.95) 100%);
            backdrop-filter: blur(10px);
            color: white;
            padding: 1rem 2rem;
            border-radius: 20px 20px 0 0;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
            display: none;
            align-items: center;
            gap: 1rem;
            z-index: 100;
            transform: translateY(-100%);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .top-processing-indicator.show {
            display: flex;
            transform: translateY(0);
        }

        .top-processing-spinner {
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            flex-shrink: 0;
        }

        .top-processing-content {
            flex: 1;
        }

        .top-processing-title {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .top-processing-subtitle {
            font-size: 0.85rem;
            opacity: 0.9;
            font-weight: 400;
        }

        .processing-model-badge {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: auto;
            flex-shrink: 0;
        }

        /* Adjust main card padding when processing indicator is shown */
        .container.processing {
            padding-top: 5rem;
        }

        /* Step Indicator */
        .step-indicator {
            display: none;
            justify-content: center;
            margin: 2rem 0;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .step {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            background: var(--secondary-color);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            transition: all 0.3s ease;
        }

        .step.active {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }

        .step.completed {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .step-number {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            font-size: 0.9rem;
        }

        /* Interactive Editing Area */
        .editing-area {
            display: none;
            text-align: center;
        }

        .tools-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .tool-card {
            background: rgba(247, 250, 252, 0.8);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
        }

        .tool-card h4 {
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Enhanced Interactive Canvas */
        .interactive-canvas-container {
            position: relative;
            display: inline-block;
            border: 3px solid var(--primary-color);
            border-radius: 12px;
            overflow: hidden;
            background: white;
            box-shadow: var(--shadow-lg);
        }

        .interactive-canvas {
            display: block;
            cursor: move;
            background: white;
        }

        .passport-frame-guide {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }

        /* Passport Photo Silhouette Overlay */
        .silhouette-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .silhouette-overlay svg {
            width: 100%;
            height: 100%;
        }

        .zoom-controls {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            margin: 1rem 0;
            align-items: center;
        }

        .zoom-btn {
            width: 40px;
            height: 40px;
            border: none;
            border-radius: 50%;
            background: var(--primary-color);
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .zoom-btn:hover {
            background: var(--primary-dark);
            transform: scale(1.1);
        }

        .zoom-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .zoom-level {
            margin: 0 1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .position-controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            max-width: 120px;
            margin: 1rem auto;
        }

        .position-btn {
            width: 35px;
            height: 35px;
            border: none;
            border-radius: 8px;
            background: rgba(247, 250, 252, 0.8);
            border: 2px solid var(--border-color);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
        }

        .position-btn:hover {
            border-color: var(--primary-color);
            background: var(--primary-color);
            color: white;
        }

        /* Background Removal Controls */
        .bg-removal-controls {
            display: flex;
            gap: 1rem;
            justify-content: center;
            align-items: center;
            margin: 1rem 0;
            flex-wrap: wrap;
        }

        .toggle-switch-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .toggle-label {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.95rem;
        }

        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }

        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        input:checked + .toggle-slider {
            background-color: var(--primary-color);
        }

        input:checked + .toggle-slider:before {
            transform: translateX(26px);
        }

        .bg-options {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(255,255,255,0.7);
            border-radius: 12px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }

        .bg-options.active {
            background: rgba(102, 126, 234, 0.1);
            border-color: var(--primary-color);
        }

        /* Color Grid for Background Selection */
        .color-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            max-width: 300px;
            margin: 0 auto;
        }

        .color-option {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem;
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }

        .color-option:hover {
            border-color: var(--primary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }

        .color-option input[type="radio"] {
            display: none;
        }

        .color-option input[type="radio"]:checked + .color-preview {
            border-width: 3px !important;
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }

        .color-preview {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 2px solid #e5e7eb;
            transition: all 0.2s ease;
            box-shadow: var(--shadow);
        }

        .color-name {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-primary);
            line-height: 1.2;
        }

        .color-option input[type="radio"]:checked ~ .color-name {
            color: var(--primary-color);
            font-weight: 700;
        }

        .before-after-container {
            display: flex;
            gap: 1rem;
            justify-content: center;
            align-items: center;
            margin: 1rem 0;
        }

        .before-after-toggle {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        .before-after-toggle:hover {
            background: var(--primary-dark);
        }

        .upload-area {
            border: 3px dashed #e2e8f0;
            border-radius: var(--border-radius);
            padding: 60px 30px;
            text-align: center;
            margin-bottom: 30px;
            background: rgba(247, 250, 252, 0.5);
            transition: var(--transition);
            cursor: pointer;
        }

        .upload-area:hover {
            border-color: var(--primary-color);
            background: rgba(102, 126, 234, 0.05);
        }

        .upload-area.dragover {
            border-color: var(--primary-color);
            background: rgba(102, 126, 234, 0.1);
            transform: scale(1.02);
        }

        .upload-icon {
            width: 80px;
            height: 80px;
            background: var(--bg-gradient);
            border-radius: 50%;
            margin: 0 auto 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 32px;
        }

        .upload-text {
            font-size: 1.2rem;
            color: var(--text-primary);
            margin-bottom: 10px;
            font-weight: 600;
        }

        .upload-subtext {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        #fileInput {
            display: none;
        }

        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .control-group {
            display: flex;
            flex-direction: column;
        }

        .control-group label {
            color: var(--text-primary);
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.95rem;
        }

        select, input[type="color"] {
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            background: white;
            color: var(--text-primary);
            font-size: 1rem;
            transition: var(--transition);
        }

        select:focus, input[type="color"]:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .btn {
            background: var(--bg-gradient);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            width: 100%;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .preview {
            display: none;
            text-align: center;
            margin-bottom: 20px;
        }

        .preview img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }

        .progress {
            display: none;
            text-align: center;
            padding: 30px 20px;
            background: rgba(247, 250, 252, 0.8);
            border-radius: var(--border-radius);
            margin-bottom: 20px;
        }

        .progress-container {
            position: relative;
            width: 100%;
            height: 12px;
            background: rgba(226, 232, 240, 0.6);
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .progress-bar {
            height: 100%;
            width: 0%;
            background: var(--bg-gradient);
            border-radius: 10px;
            transition: width 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% {
                transform: translateX(-100%);
            }
            100% {
                transform: translateX(100%);
            }
        }

        .progress-text {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            font-size: 0.95rem;
        }

        .progress-percentage {
            font-weight: 700;
            color: var(--primary-color);
            font-size: 1.1rem;
        }

        .progress-status {
            color: var(--text-secondary);
            font-weight: 500;
        }

        .spinner {
            display: none;
        }

        .result {
            display: none;
            text-align: center;
            margin-top: 20px;
            padding: 30px 20px;
            background: rgba(72, 187, 120, 0.05);
            border-radius: var(--border-radius);
            border: 1px solid rgba(72, 187, 120, 0.2);
        }

        .success-icon {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #48bb78, #38a169);
            border-radius: 50%;
            margin: 0 auto 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
            animation: successPulse 0.6s ease-out;
        }

        @keyframes successPulse {
            0% {
                transform: scale(0.8);
                opacity: 0;
            }
            50% {
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
                opacity: 1;
            }
        }

        .success-text {
            color: var(--text-primary);
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 20px;
        }

        .download-btn {
            background: linear-gradient(135deg, #48bb78, #38a169);
            margin-top: 15px;
            animation: downloadReady 0.8s ease-out;
        }

        .download-btn:hover {
            box-shadow: 0 10px 25px rgba(72, 187, 120, 0.3);
        }

        @keyframes downloadReady {
            0% {
                transform: translateY(20px);
                opacity: 0;
            }
            100% {
                transform: translateY(0);
                opacity: 1;
            }
        }

        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 15px 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: none;
            border: 1px solid #feb2b2;
        }

        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 30px;
            padding-top: 30px;
            border-top: 1px solid #e2e8f0;
        }

        .feature {
            text-align: center;
            padding: 20px 15px;
            background: rgba(247, 250, 252, 0.8);
            border-radius: 12px;
            transition: var(--transition);
        }

        .feature:hover {
            background: rgba(102, 126, 234, 0.05);
            transform: translateY(-3px);
        }

        .feature-icon {
            font-size: 24px;
            margin-bottom: 10px;
            color: var(--primary-color);
        }

        .feature-title {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.9rem;
            margin-bottom: 5px;
        }

        .feature-desc {
            color: var(--text-secondary);
            font-size: 0.8rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 30px 20px;
                margin: 10px;
            }

            h1 {
                font-size: 2rem;
            }

            .controls {
                grid-template-columns: 1fr;
            }

            .upload-area {
                padding: 40px 20px;
            }

            .features {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🎨</div>
            <h1>InSPyReNet</h1>
            <p class="subtitle">Professional Photo Editing & Passport Photos</p>
        </div>

        <div class="mode-selector">
            <div class="mode-tabs">
                <button class="mode-tab active" id="backgroundTab" onclick="switchMode('background')">
                    <span class="mode-icon">✨</span>
                    Background Removal
                </button>
                <button class="mode-tab" id="passportTab" onclick="switchMode('passport')">
                    <span class="mode-icon">📋</span>
                    Passport Photo
                </button>
            </div>
        </div>

        <div class="error" id="errorMessage"></div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📷</div>
            <div class="upload-text">Drop your image here</div>
            <div class="upload-subtext">or click to browse • JPG, PNG, WEBP up to 50MB</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>

        <div class="preview" id="preview">
            <img id="previewImage" alt="Preview">
        </div>

        <!-- Top Processing Indicator -->
        <div class="top-processing-indicator" id="topProcessingIndicator">
            <div class="top-processing-spinner"></div>
            <div class="top-processing-content">
                <div class="top-processing-title" id="topProcessingTitle">
                    <span>🤖 AI Processing</span>
                </div>
                <div class="top-processing-subtitle" id="topProcessingSubtitle">
                    Creating professional passport photo with InSPyReNet...
                </div>
            </div>
            <div class="processing-model-badge" id="topProcessingBadge">
                InSPyReNet
            </div>
        </div>

        <!-- Step Indicator -->
        <div class="step-indicator" id="stepIndicator">
            <div class="step completed" id="step1">
                <div class="step-number">1</div>
                <span>Upload</span>
            </div>
            <div class="step active" id="step2">
                <div class="step-number">2</div>
                <span>Position</span>
            </div>
            <div class="step" id="step3">
                <div class="step-number">3</div>
                <span>Adjust</span>
            </div>
            <div class="step" id="step4">
                <div class="step-number">4</div>
                <span>Download</span>
            </div>
        </div>

        <!-- Interactive Editing Area -->
        <div class="editing-area" id="editingArea">
            <h3>✨ Interactive Photo Editor</h3>
            <div class="tools-panel">
                <div class="tool-card">
                    <h4>🎯 Position & Scale</h4>
                    <div class="interactive-canvas-container" id="canvasContainer">
                        <canvas id="interactiveCanvas" class="interactive-canvas"></canvas>
                        <div class="passport-frame-guide" id="frameGuide">
                            <div class="silhouette-overlay" id="silhouetteOverlay">
                                <svg viewBox="0 0 100 100" preserveAspectRatio="none" id="silhouetteSvg">
                                    <!-- Top dashed guide line -->
                                    <line x1="25" y1="15" x2="75" y2="15" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
                                    
                                    <!-- Simple oval outline for head -->
                                    <ellipse cx="50" cy="45" rx="18" ry="28" fill="none" stroke="#666" stroke-width="0.6"/>
                                    
                                    <!-- Bottom dashed guide line (chin level) -->
                                    <line x1="25" y1="75" x2="75" y2="75" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
                                    
                                    <!-- Measurement arrow - vertical line with arrows -->
                                    <line x1="15" y1="15" x2="15" y2="75" stroke="#888" stroke-width="0.5"/>
                                    <!-- Top arrow head -->
                                    <path d="M 15,15 L 12,20 M 15,15 L 18,20" stroke="#888" stroke-width="0.5" fill="none"/>
                                    <!-- Bottom arrow head -->
                                    <path d="M 15,75 L 12,70 M 15,75 L 18,70" stroke="#888" stroke-width="0.5" fill="none"/>
                                    
                                    <!-- Measurement text -->
                                    <text x="5" y="48" font-size="5" fill="#888" transform="rotate(-90, 8, 45)" id="measurementText">1-1⅜ in</text>
                                    
                                    <!-- Bottom solid line -->
                                    <line x1="20" y1="92" x2="80" y2="92" stroke="#666" stroke-width="1"/>
                                </svg>
                            </div>
                        </div>
                    </div>
                    
                    <div class="zoom-controls">
                        <button class="zoom-btn" id="zoomOut">−</button>
                        <div class="zoom-level" id="zoomLevel">100%</div>
                        <button class="zoom-btn" id="zoomIn">+</button>
                    </div>
                    
                    <div class="position-controls">
                        <button class="position-btn" onclick="moveImage(-10, -10)">↖</button>
                        <button class="position-btn" onclick="moveImage(0, -10)">↑</button>
                        <button class="position-btn" onclick="moveImage(10, -10)">↗</button>
                        <button class="position-btn" onclick="moveImage(-10, 0)">←</button>
                        <button class="position-btn" onclick="centerImage()">⌂</button>
                        <button class="position-btn" onclick="moveImage(10, 0)">→</button>
                        <button class="position-btn" onclick="moveImage(-10, 10)">↙</button>
                        <button class="position-btn" onclick="moveImage(0, 10)">↓</button>
                        <button class="position-btn" onclick="moveImage(10, 10)">↘</button>
                    </div>
                </div>

                <div class="tool-card">
                    <h4>⚙️ Settings</h4>
                    <div class="control-group">
                        <label>📏 Photo Standard</label>
                        <select class="dropdown-select" id="photoStandard" name="standard">
                            <option value="us" selected>🇺🇸 US Standard (2×2 inches)</option>
                            <option value="eu">🇪🇺 EU Standard (35×45mm)</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="btn-group">
                <button class="btn" onclick="resetToStep1()">🔄 Start Over</button>
                <button class="btn" onclick="undoLastChange()">↶ Undo</button>
                <button class="btn btn-success" onclick="downloadPassportPhoto()">⬇️ Download Passport Photo</button>
            </div>
        </div>

        <div class="controls" id="backgroundControls">
            <div class="control-group">
                <label for="qualitySelect">Quality Mode</label>
                <select id="qualitySelect">
                    <option value="high_quality">High Quality (Slower)</option>
                    <option value="fast">Fast (Lower Quality)</option>
                </select>
            </div>
            <div class="control-group">
                <label for="backgroundColorPicker">Background Color</label>
                <input type="color" id="backgroundColorPicker" value="#ffffff">
            </div>
        </div>

        <button class="btn" id="processBtn" onclick="processImage()">
            ✨ Remove Background
        </button>

        <div class="progress" id="progress">
            <div class="progress-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="progress-text">
                <span class="progress-status" id="progressStatus">Preparing...</span>
                <span class="progress-percentage" id="progressPercentage">0%</span>
            </div>
        </div>

        <div class="result" id="result">
            <div class="success-icon">✓</div>
            <div class="success-text">Background removed successfully!</div>
            <button class="btn download-btn" id="downloadBtn">
                💾 Download Result
            </button>
        </div>

        <div class="features">
            <div class="feature">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">High Precision</div>
                <div class="feature-desc">Research-grade accuracy</div>
            </div>
            <div class="feature">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Fast Processing</div>
                <div class="feature-desc">Quick results</div>
            </div>
            <div class="feature">
                <div class="feature-icon">🎨</div>
                <div class="feature-title">Custom Backgrounds</div>
                <div class="feature-desc">Any color you want</div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const previewImage = document.getElementById('previewImage');
        const processBtn = document.getElementById('processBtn');
        const progress = document.getElementById('progress');
        const result = document.getElementById('result');
        const downloadBtn = document.getElementById('downloadBtn');
        const errorMessage = document.getElementById('errorMessage');

        // Upload area click
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file');
                return;
            }

            if (file.size > 50 * 1024 * 1024) {
                showError('File size must be less than 50MB');
                return;
            }

            selectedFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                preview.style.display = 'block';
                processBtn.disabled = false;
                hideError();
            };
            reader.readAsDataURL(file);
        }

        function processImage() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }

            const formData = new FormData();
            formData.append('image', selectedFile);
            
            let endpoint = '/remove-background';
            let successText = 'Background removed successfully!';
            let filename = selectedFile.name.replace(/\\.[^/.]+$/, '_no_bg.png');
            
            if (currentMode === 'passport') {
                // Passport photo mode
                endpoint = '/create-passport-photo';
                successText = 'Passport photo created successfully!';
                filename = selectedFile.name.replace(/\\.[^/.]+$/, '_passport_2x2.jpg');
                
                // Add positioning parameters
                formData.append('scale', currentZoom.toString());
                formData.append('x', imageOffsetX.toString());
                formData.append('y', imageOffsetY.toString());
                formData.append('quality', 'high_quality'); // Always use high quality for passport photos
            } else {
                // Background removal mode
                formData.append('quality', document.getElementById('qualitySelect').value);
                formData.append('background_color', document.getElementById('backgroundColorPicker').value);
            }

            // Show progress and start animation
            processBtn.disabled = true;
            progress.style.display = 'block';
            result.style.display = 'none';
            hideError();
            
            // Start progress animation with appropriate mode
            startProgressAnimation(currentMode);

            fetch(endpoint, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => Promise.reject(err));
                }
                return response.blob();
            })
            .then(blob => {
                // Complete progress animation
                completeProgress();
                
                // Create download URL and set up download button
                const url = URL.createObjectURL(blob);
                
                // Update success message
                document.querySelector('.success-text').textContent = successText;
                
                // Set up download function
                downloadBtn.onclick = () => {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    a.click();
                };
                
                // Show success result after a brief delay
                setTimeout(() => {
                    progress.style.display = 'none';
                    result.style.display = 'block';
                }, 500);
            })
            .catch(error => {
                showError(error.message || 'Processing failed. Please try again.');
                console.error('Error:', error);
                progress.style.display = 'none';
                processBtn.disabled = false;
            });
        }

        function startProgressAnimation() {
            const progressBar = document.getElementById('progressBar');
            const progressPercentage = document.getElementById('progressPercentage');
            const progressStatus = document.getElementById('progressStatus');
            
            let currentProgress = 0;
            const stages = [
                { end: 20, duration: 2000, status: 'Loading InSPyReNet model...' },
                { end: 50, duration: 3000, status: 'Processing image...' },
                { end: 80, duration: 2500, status: 'Removing background...' },
                { end: 95, duration: 1500, status: 'Finalizing...' }
            ];
            
            let currentStage = 0;
            
            function updateProgress() {
                if (currentStage < stages.length) {
                    const stage = stages[currentStage];
                    const increment = (stage.end - currentProgress) / (stage.duration / 100);
                    
                    const interval = setInterval(() => {
                        currentProgress += increment;
                        
                        if (currentProgress >= stage.end) {
                            currentProgress = stage.end;
                            clearInterval(interval);
                            currentStage++;
                            
                            // Update status for next stage
                            if (currentStage < stages.length) {
                                progressStatus.textContent = stages[currentStage].status;
                                setTimeout(updateProgress, 200);
                            }
                        }
                        
                        // Update UI
                        progressBar.style.width = currentProgress + '%';
                        progressPercentage.textContent = Math.round(currentProgress) + '%';
                    }, 100);
                    
                    // Set initial status
                    progressStatus.textContent = stage.status;
                }
            }
            
            updateProgress();
        }

        function completeProgress() {
            const progressBar = document.getElementById('progressBar');
            const progressPercentage = document.getElementById('progressPercentage');
            const progressStatus = document.getElementById('progressStatus');
            
            // Complete the progress bar
            progressBar.style.width = '100%';
            progressPercentage.textContent = '100%';
            progressStatus.textContent = 'Complete!';
            
            // Reset for next use
            setTimeout(() => {
                processBtn.disabled = false;
                progressBar.style.width = '0%';
                progressPercentage.textContent = '0%';
                progressStatus.textContent = 'Preparing...';
            }, 1000);
        }

        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
        }

        function hideError() {
            errorMessage.style.display = 'none';
        }

        // Advanced Interactive Canvas System
        let originalImageData = null;
        let canvas = document.createElement('canvas');
        let ctx = canvas.getContext('2d');
        let interactiveCanvas = null;
        let interactiveCtx = null;
        let currentStandard = 'us';
        let currentZoom = 1.0;
        let imageOffsetX = 0;
        let imageOffsetY = 0;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let currentStep = 1;
        
        // Throttling variables for smooth dragging
        let animationFrameId = null;
        let pendingRender = false;

        // Mode switching functionality
        let currentMode = 'background';
        
        function resetAllState() {
            // Clear passport photo state
            originalImageData = null;
            interactiveCanvas = null;
            interactiveCtx = null;
            currentStandard = 'us';
            currentZoom = 1.0;
            imageOffsetX = 0;
            imageOffsetY = 0;
            isDragging = false;
            dragStartX = 0;
            dragStartY = 0;
            currentStep = 1;
            
            // Reset UI elements
            const zoomLevel = document.getElementById('zoomLevel');
            if (zoomLevel) zoomLevel.textContent = '100%';
            
            const photoStandard = document.getElementById('photoStandard');
            if (photoStandard) photoStandard.value = 'us';
            
            // Clear file input
            fileInput.value = '';
            
            // Hide all sections
            hideError();
            document.getElementById('editingArea').style.display = 'none';
            document.getElementById('stepIndicator').style.display = 'none';
            document.getElementById('preview').style.display = 'none';
            document.getElementById('progress').style.display = 'none';
            document.getElementById('result').style.display = 'none';
            document.getElementById('uploadArea').style.display = 'block';
            
            // Reset button state
            processBtn.disabled = true;
            
            // Clear any canvas contexts
            if (canvas) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }
        
        function switchMode(mode) {
            // Only reset state if actually switching modes
            if (currentMode !== mode) {
                resetAllState();
                selectedFile = null;
            }
            
            currentMode = mode;
            
            // Update tab appearance
            document.getElementById('backgroundTab').classList.toggle('active', mode === 'background');
            document.getElementById('passportTab').classList.toggle('active', mode === 'passport');
            
            // Show/hide relevant sections
            const backgroundControls = document.getElementById('backgroundControls');
            const editingArea = document.getElementById('editingArea');
            const stepIndicator = document.getElementById('stepIndicator');
            const preview = document.getElementById('preview');
            const processBtn = document.getElementById('processBtn');
            
            if (mode === 'passport') {
                backgroundControls.style.display = 'none';
                processBtn.innerHTML = '📋 Create Passport Photo';
                
                // Show passport editing if image is selected
                if (selectedFile) {
                    initializeEditor();
                }
            } else {
                backgroundControls.style.display = 'grid';
                editingArea.style.display = 'none';
                stepIndicator.style.display = 'none';
                processBtn.innerHTML = '✨ Remove Background';
                
                // Show regular preview if image is selected
                if (selectedFile) {
                    preview.style.display = 'block';
                }
            }
        }

        // Initialize Interactive Editor
        function initializeEditor() {
            // Initialize interactive canvas
            interactiveCanvas = document.getElementById('interactiveCanvas');
            interactiveCtx = interactiveCanvas.getContext('2d');
            
            // Reset values - start with better zoom for overview
            currentZoom = 0.5;
            imageOffsetX = 0;
            imageOffsetY = 0;
            currentStep = 2;
            
            // Set up canvas size
            updateCanvasSize();
            setupInteractiveCanvas();
            
            // Show editing interface
            showEditor();
            updateSteps();
        }

        function updateCanvasSize() {
            if (!interactiveCanvas) return;
            
            const dimensions = getPhotoDimensions(currentStandard);
            interactiveCanvas.width = dimensions.width;
            interactiveCanvas.height = dimensions.height;
            
            // Update frame guide size
            const frameGuide = document.getElementById('frameGuide');
            frameGuide.style.width = dimensions.width + 'px';
            frameGuide.style.height = dimensions.height + 'px';
            
            // Update alignment guides for the current standard
            updateAlignmentGuides();
        }
        
        function updateAlignmentGuides() {
            // Update the guide SVG based on the current standard
            const silhouetteSvg = document.getElementById('silhouetteSvg');
            
            if (currentStandard === 'us') {
                // US standard: Square (2x2"), head 1-1.375 inches (25-35mm)
                silhouetteSvg.innerHTML = `
                    <!-- Top dashed guide line -->
                    <line x1="25" y1="15" x2="75" y2="15" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
                    
                    <!-- Simple oval outline for head -->
                    <ellipse cx="50" cy="45" rx="20" ry="28" fill="none" stroke="#666" stroke-width="0.6"/>
                    
                    <!-- Bottom dashed guide line (chin level) -->
                    <line x1="25" y1="75" x2="75" y2="75" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
                    
                    <!-- Measurement arrow - vertical line with arrows -->
                    <line x1="15" y1="15" x2="15" y2="75" stroke="#888" stroke-width="0.5"/>
                    <!-- Top arrow head -->
                    <path d="M 15,15 L 12,20 M 15,15 L 18,20" stroke="#888" stroke-width="0.5" fill="none"/>
                    <!-- Bottom arrow head -->
                    <path d="M 15,75 L 12,70 M 15,75 L 18,70" stroke="#888" stroke-width="0.5" fill="none"/>
                    
                    <!-- Measurement text -->
                    <text x="3" y="48" font-size="4.5" fill="#888" transform="rotate(-90, 8, 45)">1-1⅜ in</text>
                    
                    <!-- Bottom solid line -->
                    <line x1="20" y1="92" x2="80" y2="92" stroke="#666" stroke-width="1"/>
                `;
            } else {
                // EU standard: Rectangular (35x45mm), head height 32-36mm
                silhouetteSvg.innerHTML = `
                    <!-- Top dashed guide line -->
                    <line x1="25" y1="12" x2="75" y2="12" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
                    
                    <!-- Simple oval outline for head (taller for EU) -->
                    <ellipse cx="50" cy="40" rx="18" ry="26" fill="none" stroke="#666" stroke-width="0.6"/>
                    
                    <!-- Bottom dashed guide line (chin level) -->
                    <line x1="25" y1="68" x2="75" y2="68" stroke="#888" stroke-width="0.8" stroke-dasharray="3,2"/>
                    
                    <!-- Measurement arrow - vertical line with arrows -->
                    <line x1="15" y1="12" x2="15" y2="68" stroke="#888" stroke-width="0.5"/>
                    <!-- Top arrow head -->
                    <path d="M 15,12 L 12,17 M 15,12 L 18,17" stroke="#888" stroke-width="0.5" fill="none"/>
                    <!-- Bottom arrow head -->
                    <path d="M 15,68 L 12,63 M 15,68 L 18,63" stroke="#888" stroke-width="0.5" fill="none"/>
                    
                    <!-- Measurement text -->
                    <text x="3" y="43" font-size="4.5" fill="#888" transform="rotate(-90, 8, 40)">32-36mm</text>
                    
                    <!-- Bottom solid line -->
                    <line x1="20" y1="92" x2="80" y2="92" stroke="#666" stroke-width="1"/>
                `;
            }
        }

        function setupInteractiveCanvas() {
            // Add mouse event listeners for dragging
            interactiveCanvas.addEventListener('mousedown', startDrag);
            interactiveCanvas.addEventListener('mousemove', drag);
            interactiveCanvas.addEventListener('mouseup', endDrag);
            interactiveCanvas.addEventListener('mouseleave', endDrag);
            
            // Touch events for mobile
            interactiveCanvas.addEventListener('touchstart', startDrag);
            interactiveCanvas.addEventListener('touchmove', drag);
            interactiveCanvas.addEventListener('touchend', endDrag);
            
            // Zoom controls
            document.getElementById('zoomIn').addEventListener('click', () => adjustZoom(0.01));
            document.getElementById('zoomOut').addEventListener('click', () => adjustZoom(-0.01));
            
            // Photo standard change listener
            document.getElementById('photoStandard').addEventListener('change', function(e) {
                currentStandard = e.target.value;
                updateCanvasSize();
                renderInteractiveCanvas();
            });
            
            renderInteractiveCanvas();
        }

        function renderInteractiveCanvas() {
            if (!originalImageData || !interactiveCanvas) return;
            
            const canvasWidth = interactiveCanvas.width;
            const canvasHeight = interactiveCanvas.height;
            
            // Clear canvas with white background
            interactiveCtx.fillStyle = 'white';
            interactiveCtx.fillRect(0, 0, canvasWidth, canvasHeight);
            
            // Calculate image size and position
            const scaledWidth = originalImageData.width * currentZoom;
            const scaledHeight = originalImageData.height * currentZoom;
            
            const drawX = (canvasWidth - scaledWidth) / 2 + imageOffsetX;
            const drawY = (canvasHeight - scaledHeight) / 2 + imageOffsetY;
            
            // Draw the image
            interactiveCtx.drawImage(originalImageData, drawX, drawY, scaledWidth, scaledHeight);
            
            // Update processed image for download
            updateProcessedImage();
        }
        
        // Throttled rendering function to prevent excessive redraws during drag
        function scheduleRender() {
            if (pendingRender) return;
            
            pendingRender = true;
            animationFrameId = requestAnimationFrame(() => {
                renderInteractiveCanvas();
                pendingRender = false;
            });
        }

        function updateProcessedImage() {
            if (!interactiveCanvas) return;
            
            // Copy interactive canvas to main canvas for download
            canvas.width = interactiveCanvas.width;
            canvas.height = interactiveCanvas.height;
            ctx.drawImage(interactiveCanvas, 0, 0);
        }

        function startDrag(e) {
            isDragging = true;
            const rect = interactiveCanvas.getBoundingClientRect();
            const clientX = e.clientX || e.touches[0].clientX;
            const clientY = e.clientY || e.touches[0].clientY;
            
            // Store the mouse position relative to canvas at start of drag
            dragStartX = clientX - rect.left;
            dragStartY = clientY - rect.top;
            
            interactiveCanvas.style.cursor = 'grabbing';
            e.preventDefault();
        }

        function drag(e) {
            if (!isDragging) return;
            
            const rect = interactiveCanvas.getBoundingClientRect();
            const clientX = e.clientX || e.touches[0].clientX;
            const clientY = e.clientY || e.touches[0].clientY;
            
            // Calculate current mouse position relative to canvas
            const currentX = clientX - rect.left;
            const currentY = clientY - rect.top;
            
            // Calculate the movement delta
            const deltaX = currentX - dragStartX;
            const deltaY = currentY - dragStartY;
            
            // Apply the movement delta to image offset
            imageOffsetX += deltaX;
            imageOffsetY += deltaY;
            
            // Update drag start position for next frame
            dragStartX = currentX;
            dragStartY = currentY;
            
            // Use throttled rendering to prevent fidgeting
            scheduleRender();
            e.preventDefault();
        }

        function endDrag() {
            isDragging = false;
            interactiveCanvas.style.cursor = 'move';
        }

        function adjustZoom(delta) {
            currentZoom = Math.max(0.01, Math.min(1.0, currentZoom + delta));
            document.getElementById('zoomLevel').textContent = Math.round(currentZoom * 100) + '%';
            
            // Update button states
            document.getElementById('zoomOut').disabled = currentZoom <= 0.01;
            document.getElementById('zoomIn').disabled = currentZoom >= 1.0;
            
            // Use throttled rendering to prevent jittering
            scheduleRender();
        }

        function moveImage(deltaX, deltaY) {
            imageOffsetX += deltaX;
            imageOffsetY += deltaY;
            // Use throttled rendering to prevent jittering
            scheduleRender();
        }

        function centerImage() {
            imageOffsetX = 0;
            imageOffsetY = 0;
            // Use throttled rendering to prevent jittering
            scheduleRender();
        }

        function updateSteps() {
            // Update step indicator (4 steps total)
            for (let i = 1; i <= 4; i++) {
                const step = document.getElementById(`step${i}`);
                if (step) {
                    if (i < currentStep) {
                        step.className = 'step completed';
                    } else if (i === currentStep) {
                        step.className = 'step active';
                    } else {
                        step.className = 'step';
                    }
                }
            }
        }

        function showEditor() {
            setTimeout(() => {
                document.getElementById('editingArea').style.display = 'block';
                document.getElementById('editingArea').classList.add('fade-in');
                document.getElementById('stepIndicator').style.display = 'flex';
                document.getElementById('stepIndicator').classList.add('fade-in');
            }, 500);
        }

        function getPhotoDimensions(standard) {
            const dimensions = {
                us: { width: 300, height: 300 }, // 2x2 inches at 150 DPI
                eu: { width: 260, height: 335 }  // 35x45mm at 150 DPI
            };
            return dimensions[standard];
        }

        function resetToStep1() {
            uploadArea.style.display = 'block';
            document.getElementById('editingArea').style.display = 'none';
            document.getElementById('stepIndicator').style.display = 'none';
            document.getElementById('preview').style.display = 'none';
            
            originalImageData = null;
            currentZoom = 0.5;
            imageOffsetX = 0;
            imageOffsetY = 0;
            currentStep = 1;
            fileInput.value = '';
            
            // Reset controls
            currentStandard = 'us';
            document.getElementById('zoomLevel').textContent = '50%';
            
            hideError();
        }

        function undoLastChange() {
            // Simple undo functionality - reset to centered position
            imageOffsetX = 0;
            imageOffsetY = 0;
            currentZoom = 0.5;
            document.getElementById('zoomLevel').textContent = '50%';
            renderInteractiveCanvas();
        }

        function downloadPassportPhoto() {
            if (!canvas) return;
            
            currentStep = 4; // Update to final step (4 steps total)
            updateSteps();
            
            const link = document.createElement('a');
            const standard = currentStandard.toUpperCase();
            link.download = `passport-photo-${standard}-${Date.now()}.jpg`;
            link.href = canvas.toDataURL('image/jpeg', 0.95);
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function renderInteractiveCanvas() {
            if (!originalImageData || !interactiveCanvas) return;
            
            const canvasWidth = interactiveCanvas.width;
            const canvasHeight = interactiveCanvas.height;
            
            // Clear canvas with white background
            interactiveCtx.fillStyle = 'white';
            interactiveCtx.fillRect(0, 0, canvasWidth, canvasHeight);
            
            // Calculate image size and position
            const scaledWidth = originalImageData.width * currentZoom;
            const scaledHeight = originalImageData.height * currentZoom;
            
            const drawX = (canvasWidth - scaledWidth) / 2 + imageOffsetX;
            const drawY = (canvasHeight - scaledHeight) / 2 + imageOffsetY;
            
            // Draw the image
            interactiveCtx.drawImage(originalImageData, drawX, drawY, scaledWidth, scaledHeight);
            
            // Update processed image for download
            updateProcessedImage();
        }

        // Modified handleFile to support both modes with proper state management
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file');
                return;
            }

            if (file.size > 50 * 1024 * 1024) {
                showError('File size must be less than 50MB');
                return;
            }

            selectedFile = file;
            
            // Clear any previous error states
            hideError();
            
            // Process file based on current mode
            if (currentMode === 'passport') {
                // Hide regular preview and results
                preview.style.display = 'none';
                result.style.display = 'none';
                
                // Load image for passport mode
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = new Image();
                    img.onload = function() {
                        originalImageData = img;
                        initializeEditor();
                    };
                    img.src = e.target.result;
                };
                reader.readAsDataURL(file);
            } else {
                // Hide passport editing areas and results
                document.getElementById('editingArea').style.display = 'none';
                document.getElementById('stepIndicator').style.display = 'none';
                result.style.display = 'none';
                
                // Setup background removal mode
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
            
            // Enable process button
            processBtn.disabled = false;
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Check dependencies
    if not INSPYRENET_AVAILABLE:
        print("\n" + "="*60)
        print("❌ ERROR: InSPyReNet (transparent-background) not found!")
        print("Please install it with:")
        print("  pip install transparent-background")
        print("="*60 + "\n")
    
    # Run server
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"\n🚀 InSPyReNet Background Removal Server")
    print(f"📍 Running on http://localhost:{port}")
    print(f"🎯 InSPyReNet available: {INSPYRENET_AVAILABLE}")
    print(f"✨ High-quality background removal ready!")
    print("-" * 50)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
