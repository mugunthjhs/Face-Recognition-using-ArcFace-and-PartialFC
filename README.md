# Face Recognition System using ArcFace and PartialFC

A robust face recognition system that uses state-of-the-art deep learning models (ArcFace and PartialFC) for accurate face comparison and verification. This project implements models from [InsightFace](https://github.com/deepinsight/insightface), a state-of-the-art 2D and 3D face analysis toolbox.

## Features

- Dual model support: ArcFace and PartialFC
- Real-time face detection and alignment
- Web-based user interface
- REST API endpoints
- Support for multiple image formats and sources (URL, Base64, local files)
- Advanced face detection with multi-angle support
- Memory-optimized processing

## Model Architecture

This project uses two main models from InsightFace:

1. **ArcFace** (CVPR'2019):
   - State-of-the-art face recognition model
   - Uses additive angular margin loss
   - 512-dimensional face embeddings
   - Input size: 112x112 RGB image

2. **PartialFC** (CVPR'2022):
   - Efficient training for large-scale face recognition
   - Optimized for memory usage
   - 512-dimensional face embeddings
   - Input size: 112x112 RGB image

For face detection, we use:
- **RetinaFace**: A robust single-stage face detector that provides accurate facial landmarks

## Required Model Files

Download the following model files from [Google Drive](https://drive.google.com/drive/folders/1sQw9P-_ALSxLw0RsczbKufgEMNJMEOEj?usp=drive_link):

1. `arcface.onnx` (248.6 MB) - ArcFace model for face recognition
2. `partialfc.onnx` (166.3 MB) - PartialFC model for face recognition
3. `RRDB_ESRGAN_x4.pth` (63.8 MB) - Super-resolution model for face alignment

Place these files in the `weights` directory of the project:
```
project_root/
├── weights/
│   ├── arcface.onnx
│   ├── partialfc.onnx
│   └── RRDB_ESRGAN_x4.pth
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the model files as described above and place them in the `weights` directory.

## Project Structure

```
project_root/
├── weights/              # Model weights directory
├── align/               # Face alignment modules
├── static/             # Static files for web interface
├── templates/          # HTML templates
├── main.py            # Main application file
├── demo.py            # Demo script
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Usage

### Starting the Server

Run the main application:
```bash
python main.py
```
The server will start on `http://localhost:2000`

### Using the Web Interface

1. Open `interface.html` in a web browser
2. Upload two images for comparison
3. Adjust the similarity thresholds if needed
4. Click on either:
   - "ArcFace Analysis" for ArcFace model comparison
   - "Partial FC Analysis" for PartialFC model comparison
   - "Compare Both Models" to use both models simultaneously

### API Endpoints

#### Compare Faces
- **URL**: `/api/compare-faces`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "image1URL": "string (URL or Base64)",
    "image2URL": "string (URL or Base64)",
    "modelType": "string (arcface or partial_fc)",
    "threshold": "float (0.0-1.0)"
  }
  ```
- **Response**:
  ```json
  {
    "similarity": "float",
    "face1": "string (Base64)",
    "face2": "string (Base64)",
    "is_match": "boolean"
  }
  ```

## Model Details

### ArcFace
- Input size: 112x112 RGB image
- Output: 512-dimensional face embedding
- Optimal threshold: 0.5-0.6
- Based on InsightFace's implementation
- Uses additive angular margin loss for better feature discrimination

### PartialFC
- Input size: 112x112 RGB image
- Output: 512-dimensional face embedding
- Optimal threshold: 0.5-0.6
- Memory-efficient implementation
- Supports large-scale face recognition training

## Performance Optimization

The system includes several optimizations:
- Memory management for large-scale processing
- GPU acceleration support
- Image compression before processing
- Multi-angle face detection
- Duplicate face elimination

## Error Handling

Common errors and solutions:
1. "No faces detected" - Ensure the image contains a clear, front-facing face
2. Memory errors - Reduce image size or increase available RAM
3. Model loading errors - Verify model files are correctly placed in weights directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - State-of-the-art 2D and 3D face analysis project
  - ArcFace implementation (CVPR'2019)
  - PartialFC implementation (CVPR'2022)
  - RetinaFace detection model (CVPR'2020)
- [RetinaFace](https://github.com/serengil/retinaface) for Python implementation of face detection
- ESRGAN for super-resolution 