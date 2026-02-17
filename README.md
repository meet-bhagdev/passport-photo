# InSPyReNet Background Removal

A modern, high-quality background removal application powered by InSPyReNet research model with a sleek drag-and-drop interface.

![InSPyReNet Background Removal](https://img.shields.io/badge/AI-InSPyReNet-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)

## Features

- 🎯 **High Precision**: Research-grade accuracy using InSPyReNet model
- ⚡ **Fast Processing**: Quick background removal with quality options
- 🎨 **Custom Backgrounds**: Choose any background color
- 📱 **Modern Interface**: Sleek drag-and-drop UI with smooth animations
- 🖼️ **Multiple Formats**: Support for JPG, PNG, WEBP, BMP (up to 50MB)
- 💾 **Easy Download**: One-click download of processed images
- 🔒 **Client-Side Processing**: No data stored on server

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the project:**
   ```bash
   cd /Users/mbh/Documents/ecprojects/background-removal
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   python server.py
   ```

4. **Open your browser:**
   ```
   http://localhost:5000
   ```

## Usage

### Web Interface

1. **Upload Image**: Drag and drop an image or click to browse
2. **Choose Quality**: Select between "High Quality" (slower) or "Fast" mode
3. **Pick Background**: Use the color picker to choose a background color
4. **Process**: Click "Remove Background" and wait for processing
5. **Download**: Click "Download Result" to save your image

### API Endpoints

#### Remove Background (File Upload)
```bash
POST /remove-background
Content-Type: multipart/form-data

Parameters:
- image: Image file (required)
- quality: "high_quality" or "fast" (optional, default: "high_quality")
- background_color: Hex color like "#ffffff" (optional, default: transparent)
```

#### Remove Background (Base64)
```bash
POST /remove-background-base64
Content-Type: application/json

Body:
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "quality": "high_quality",
  "background_color": "#ffffff"
}
```

#### Health Check
```bash
GET /health
```

## Technical Details

### InSPyReNet Model

This application uses the InSPyReNet (Inverse Saliency Pyramid Reconstruction Network) model via the `transparent-background` package. InSPyReNet is a research-grade model published in ACCV 2022 that provides superior background removal quality, especially for high-resolution images.

**Paper**: "Revisiting Image Pyramid Structure for High Resolution Salient Object Detection"

### Architecture

- **Backend**: Flask web server with InSPyReNet integration
- **Frontend**: Modern HTML/CSS/JavaScript embedded in server
- **Model**: InSPyReNet via transparent-background package
- **Processing**: PIL/Pillow for image handling

### Performance

- **Model Loading**: ~10-30 seconds on first use (cached afterwards)
- **Processing Time**: 2-10 seconds depending on image size and quality mode
- **Memory Usage**: ~2-4GB for model + image processing
- **Supported Sizes**: Up to 50MB images, optimized for high-resolution

## Configuration

### Environment Variables

- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)

### Quality Modes

- **High Quality (`high_quality`)**: Uses InSPyReNet 'base' mode for maximum accuracy
- **Fast (`fast`)**: Uses InSPyReNet 'fast' mode for quicker processing

## Troubleshooting

### Common Issues

1. **"InSPyReNet not available" Error**
   ```bash
   pip install transparent-background
   ```

2. **Slow Model Loading**
   - First run downloads models (~180MB)
   - Models are cached for subsequent runs
   - Ensure good internet connection

3. **Out of Memory**
   - Reduce image size before processing
   - Use "fast" quality mode
   - Ensure sufficient RAM (4GB+ recommended)

4. **Processing Timeout**
   - Very large images may take several minutes
   - Try smaller images or "fast" mode first

### Dependencies Issues

If you encounter dependency conflicts, create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Development

### Project Structure
```
background-removal/
├── server.py           # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── uploads/           # Temporary upload directory (auto-created)
```

### API Testing

Test with curl:
```bash
curl -X POST -F "image=@test.jpg" -F "quality=high_quality" -F "background_color=#ffffff" http://localhost:5000/remove-background -o result.png
```

## Credits

- **InSPyReNet**: Original research by Kim et al. (ACCV 2022)
- **transparent-background**: Python wrapper by plemeri
- **Model Paper**: https://arxiv.org/abs/2209.09475
- **GitHub**: https://github.com/plemeri/InSPyReNet

## License

This project is for educational and research purposes. Please refer to the InSPyReNet original license for model usage terms.

---

**Made with ❤️ using InSPyReNet and Flask**
# passport-photo
