import cv2
import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from retinaface import RetinaFace
from torch.nn.functional import cosine_similarity
import align.RRDBNet_arch as arch
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import warnings
import logging
import sys
import tensorflow as tf
import psutil
import gc

# Memory Management
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Configure TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set PyTorch to release memory aggressively
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def check_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


align_model_path = 'weights/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
align_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
align_model.load_state_dict(torch.load(align_model_path, map_location=device), strict=True)
align_model.eval()
align_model = align_model.to(device)

# Memory Management
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    # Remove the "data:image/..." prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def url_to_image(url):
    """
    Download image from URL and convert directly to numpy array without saving
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert the image to a numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from URL")
            
        return image
    except Exception as e:
        raise ValueError(f"Failed to process image from URL: {str(e)}")

def extract_faces(image: np.ndarray, padding: int = 15, is_query: bool = False) -> list:
    """
    Enhanced face detection with multi-angle detection and pose handling.
    is_query: If True, only process the original image without variations
    """
    if image is None:
        raise ValueError("Invalid image input")

    def rotate_image(img, angle):
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        
        rotation_matrix[0, 2] += bound_w/2 - center[0]
        rotation_matrix[1, 2] += bound_h/2 - center[1]
        
        rotated = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated, rotation_matrix

    height, width = image.shape[:2]
    face_images = []
    
    # For query images, only process original orientation
    angles = [0] if is_query else [0, -30, -15, 15, 30]
    
    for angle in angles:
        if angle == 0:
            current_image = image
        else:
            current_image, _ = rotate_image(image, angle)
        
        resp = RetinaFace.detect_faces(current_image)
        if not resp:
            continue
            
        for face_info in resp.values():
            facial_area = face_info["facial_area"]
            landmarks = face_info.get("landmarks", {})
            
            # Only calculate side view for database processing
            is_side_view = False
            if not is_query and "left_eye" in landmarks and "right_eye" in landmarks:
                left_eye = np.array(landmarks["left_eye"])
                right_eye = np.array(landmarks["right_eye"])
                eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                is_side_view = abs(eye_angle) > 10
            
            x1, y1, x2, y2 = map(int, facial_area)
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(current_image.shape[1], x2 + padding)
            y2 = min(current_image.shape[0], y2 + padding)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            face = current_image[y1:y2, x1:x2]
            
            aspect_ratio = face.shape[1] / face.shape[0]
            if aspect_ratio > 1:
                new_width = 112
                new_height = int(112 / aspect_ratio)
            else:
                new_height = 112
                new_width = int(112 * aspect_ratio)
            
            face = cv2.resize(face, (new_width, new_height))
            
            if new_width != 112 or new_height != 112:
                top = (112 - new_height) // 2
                bottom = 112 - new_height - top
                left = (112 - new_width) // 2
                right = 112 - new_width - left
                face = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_REPLICATE)
            
            face_images.append(face)
            
            # Only add mirrored version for database processing
            if not is_query and is_side_view:
                mirrored_face = cv2.flip(face, 1)
                face_images.append(mirrored_face)
    
    # Remove duplicates only for database processing
    if not is_query and len(face_images) > 1:
        unique_faces = [face_images[0]]
        for face in face_images[1:]:
            is_duplicate = False
            for unique_face in unique_faces:
                diff = cv2.absdiff(face, unique_face)
                similarity = np.mean(diff)
                if similarity < 25:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append(face)
        face_images = unique_faces
    
    return face_images

def apply_sobel_filter(image):
    """
    Apply Sobel filter for edge detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel in x and y directions with optimized kernel size
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine x and y gradients
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255 range with adjusted scaling
    gradient_magnitude = np.uint8(gradient_magnitude * 255 / gradient_magnitude.max())
    
    # Convert back to 3-channel image
    edge_image = cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR)
    
    # Blend original image with edge detection - adjusted weights for better balance
    result = cv2.addWeighted(image, 0.75, edge_image, 0.25, 0)
    
    return result

def align_faces(face_images: list, is_train: bool = True) -> list:
    """
    Simple geometric alignment without color modifications
    """
    aligned_faces = []
    
    for face in face_images:
        # Ensure the face is properly sized
        if face.shape[:2] != (112, 112):
            face = cv2.resize(face, (112, 112))
        aligned_faces.append(face)
    
    return aligned_faces

def get_face_embedding(face_image: np.ndarray, session: onnxruntime.InferenceSession) -> torch.Tensor:
    """
    Get the embedding of a face image using the ONNX model.
    """
    image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image).unsqueeze(0).numpy()
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: input_tensor})[0]
    embedding_tensor = torch.tensor(embedding)
    
    # Verify embedding dimension
    if embedding_tensor.shape[1] != 512:
        raise ValueError(f"Model output dimension is {embedding_tensor.shape[1]}, expected 512")
    
    return embedding_tensor

def calculate_similarity(train_embedding: torch.Tensor, validate_embedding: torch.Tensor) -> float:
    """
    Calculate the cosine similarity between two embeddings.
    """
    train_embedding = torch.nn.functional.normalize(train_embedding, dim=1)
    validate_embedding = torch.nn.functional.normalize(validate_embedding, dim=1)
    return cosine_similarity(train_embedding, validate_embedding).item()

def compare_faces(image1_data, image2_data, model_type="arcface", threshold=0.5):
    """
    Compare two face images and return similarity score
    """
    try:
        # Load appropriate model based on type
        if model_type == "arcface":
            model_path = r"weights\arcface.onnx"
        else:  # partial_fc
            model_path = r"weights\partialfc.onnx"
            
        session = onnxruntime.InferenceSession(model_path)

        # Process first image
        if isinstance(image1_data, str) and (image1_data.startswith('http://') or image1_data.startswith('https://')):
            image1 = url_to_image(image1_data)
        else:
            # Handle base64 or file path
            if isinstance(image1_data, str) and image1_data.startswith('data:image/'):
                image1 = base64_to_image(image1_data)
            else:
                image1 = cv2.imread(image1_data)

        # Process second image
        if isinstance(image2_data, str) and (image2_data.startswith('http://') or image2_data.startswith('https://')):
            image2 = url_to_image(image2_data)
        else:
            # Handle base64 or file path
            if isinstance(image2_data, str) and image2_data.startswith('data:image/'):
                image2 = base64_to_image(image2_data)
            else:
                image2 = cv2.imread(image2_data)

        # Extract and align faces
        faces1 = extract_faces(image1, is_query=True)
        faces2 = extract_faces(image2, is_query=True)

        if not faces1 or not faces2:
            raise ValueError("No faces detected in one or both images")

        aligned_faces1 = align_faces(faces1)
        aligned_faces2 = align_faces(faces2)

        # Get embeddings
        embedding1 = get_face_embedding(aligned_faces1[0], session)
        embedding2 = get_face_embedding(aligned_faces2[0], session)

        # Calculate similarity and match status
        similarity = calculate_similarity(embedding1, embedding2)
        is_match = similarity >= threshold

        return {
            'similarity': float(similarity),
            'face1': aligned_faces1[0],
            'face2': aligned_faces2[0],
            'is_match': is_match,
            'threshold': threshold
        }

    except Exception as e:
        raise Exception(f"Error comparing faces: {str(e)}")

@app.route('/api/compare-faces', methods=['POST'])
def compare_faces_api():
    try:
        data = request.get_json()
        image1_url = data.get('image1URL')
        image2_url = data.get('image2URL')
        model_type = data.get('modelType', 'arcface')
        threshold = float(data.get('threshold', 0.5))  # Get threshold from request

        if not image1_url or not image2_url:
            print("[ERROR] Both images are required", file=sys.stderr)
            return jsonify({'error': 'Both images are required'}), 400

        result = compare_faces(image1_url, image2_url, model_type, threshold)
        print(f"[INFO] Successfully compared faces using {model_type} model", file=sys.stderr)

        # Convert face images to base64
        _, buffer1 = cv2.imencode('.jpg', result['face1'])
        _, buffer2 = cv2.imencode('.jpg', result['face2'])
        
        face1_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer1).decode('utf-8')}"
        face2_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer2).decode('utf-8')}"

        return jsonify({
            'similarity': result['similarity'],
            'face1': face1_b64,
            'face2': face2_b64,
            'is_match': result['is_match']
        })

    except Exception as e:
        error_msg = f"[ERROR] {str(e)}"
        print(error_msg, file=sys.stderr)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000)
