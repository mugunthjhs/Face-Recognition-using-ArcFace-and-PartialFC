import os
import warnings
import logging
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Suppress warnings
warnings.filterwarnings('ignore')

import cv2
import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
from retinaface import RetinaFace
import base64

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
            x1, y1, x2, y2 = map(int, facial_area)
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(current_image.shape[1], x2 + padding)
            y2 = min(current_image.shape[0], y2 + padding)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            face = current_image[y1:y2, x1:x2]
            face = cv2.resize(face, (112, 112))
            face_images.append(face)
    
    return face_images

def align_faces(face_images: list) -> list:
    """Simple geometric alignment without color modifications"""
    aligned_faces = []
    for face in face_images:
        if face.shape[:2] != (112, 112):
            face = cv2.resize(face, (112, 112))
        aligned_faces.append(face)
    return aligned_faces

def get_face_embedding(face_image: np.ndarray, session: onnxruntime.InferenceSession) -> torch.Tensor:
    """Get the embedding of a face image using the ONNX model."""
    image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image).unsqueeze(0).numpy()
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: input_tensor})[0]
    return torch.tensor(embedding)

def calculate_similarity(train_embedding: torch.Tensor, validate_embedding: torch.Tensor) -> float:
    """Calculate the cosine similarity between two embeddings."""
    train_embedding = torch.nn.functional.normalize(train_embedding, dim=1)
    validate_embedding = torch.nn.functional.normalize(validate_embedding, dim=1)
    return torch.nn.functional.cosine_similarity(train_embedding, validate_embedding).item()

def compare_faces(image1_path, image2_path, model_type="arcface"):
    """Compare two face images and return similarity score"""
    try:
        # Load appropriate model based on type
        if model_type == "arcface":
            model_path = r"weights\arcface.onnx"
        else:  # partial_fc
            model_path = r"weights\partialfc.onnx"
            
        session = onnxruntime.InferenceSession(model_path)

        # Load images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if image1 is None or image2 is None:
            raise ValueError("Failed to load one or both images")

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

        # Calculate similarity
        similarity = calculate_similarity(embedding1, embedding2)

        return {
            'similarity': float(similarity),
            'face1': aligned_faces1[0],
            'face2': aligned_faces2[0]
        }

    except Exception as e:
        raise Exception(f"Error comparing faces: {str(e)}")

def main():
    # Test images paths
    image1_path = r"test_images\test1.jpg"
    image2_path = r"test_images\test2.jpg"
    
    try:
        result = compare_faces(image1_path, image2_path, "arcface")
        print(f"ArcFace Similarity: {result['similarity']:.4f}")
    except Exception as e:
        print(f"Error with ArcFace: {e}")

    try:
        result = compare_faces(image1_path, image2_path, "partial_fc")
        print(f"Partial FC Similarity: {result['similarity']:.4f}")
    except Exception as e:
        print(f"Error with Partial FC: {e}")

if __name__ == "__main__":
    main() 