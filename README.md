# ResCat ML API

Flask-based API service for cat recognition and face detection using MobileNetV3 and YOLO models.

## Features

- Cat image classification (CAT/NON-CAT)
- Cat face detection with smart ROI selection
- Background removal with caching
- Image upload to storage API
- Health check endpoint

## Quick Setup

### Automated Setup (Recommended)

Run the setup script to automatically configure everything:

```powershell
.\setup.ps1
```

This will:
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Create `.env` from template
- ✅ Create necessary directories
- ✅ Verify model files

### Manual Setup

If you prefer manual setup:

#### 1. Clone Repository

```bash
git clone <repository-url>
cd rescat-ml
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

**Important settings to configure:**

- `CONTENT_API_BASE` - Your storage API URL (required for uploads)
- `PORT` - Server port (default: 5000)
- `CORS_ENABLED` - Enable CORS if needed (default: false)

### 5. Run Application

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### `GET /` or `GET /v1`
Web interface with API documentation

### `GET /healthz`
Health check endpoint

### `POST /v1/cat/recognize`
Recognize cat in image and detect faces
- **Input**: multipart/form-data with `file` field
- **Output**: Classification result + face detection

### `POST /v1/cat/faces`
Detect cat faces only
- **Input**: multipart/form-data with `file` field
- **Output**: Face detection result with bounding boxes

### `POST /v1/cat/remove-bg`
Remove background from image
- **Input**: 
  - Mode 1: multipart/form-data with `file` field
  - Mode 2: JSON/form/query with `url` parameter
- **Output**: Processed image URL (cached)

## Configuration

See `.env.example` for all available configuration options.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 5000 |
| `THRESHOLD` | Cat classification threshold | 0.50 |
| `MAX_FILE_MB` | Max upload size (MB) | 8 |
| `CONTENT_API_BASE` | Storage API URL | Required |
| `IMG_SIZE` | YOLO input size | 640 |

## Models

- **MobileNetV3**: Cat image classification
- **YOLO**: Cat face detection

Place model files in `models/` directory:
- `mobilenetv3_small.onnx`
- `cat_head_model.onnx`
- `imagenet_classes.txt`
- `cat_head_classes.json`

## Development

Project structure:
```
├── app.py              # Main Flask application
├── config.py           # Configuration management
├── inference.py        # Cat classifier
├── yolo_face.py        # Face detector
├── utils/
│   ├── validation.py   # Upload validation
│   ├── responses.py    # JSON responses
│   └── uploader.py     # Storage upload
├── models/             # ONNX models
├── static/             # Static files
└── templates/          # HTML templates
```

## License

[Your License]
