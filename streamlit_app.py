import streamlit as st
import base64
from PIL import Image
import io
import os
import gdown
import cv2
import torch
import onnxruntime
import numpy as np
from retinaface import RetinaFace
import warnings
import logging
import tensorflow as tf
from torchvision import transforms

# Suppress warnings and configure logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Must be the first Streamlit command
st.set_page_config(
    page_title="Face Recognition Analysis",
    page_icon="👤",
    layout="wide"
)

# Function to download weights
@st.cache_resource
def download_weights():
    """Download model weights if they don't exist"""
    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Dictionary of model files and their Google Drive IDs from secrets
    model_files = {
        'arcface.onnx': st.secrets["gcloud"]["arcface_id"],
        'partialfc.onnx': st.secrets["gcloud"]["partialfc_id"],
        'RRDB_ESRGAN_x4.pth': st.secrets["gcloud"]["esrgan_id"]
    }
    
    # Create a placeholder for the download progress
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        total_files = len(model_files)
        for idx, (file_name, drive_id) in enumerate(model_files.items(), 1):
            file_path = os.path.join(weights_dir, file_name)
            if not os.path.exists(file_path):
                progress_text.text(f"Downloading {file_name}... ({idx}/{total_files})")
                url = f'https://drive.google.com/uc?id={drive_id}'
                gdown.download(url, file_path, quiet=False)
            progress_bar.progress(idx/total_files)
        
        progress_text.text("All model weights downloaded successfully!")
        progress_bar.progress(1.0)
        return True
        
    except Exception as e:
        progress_text.error(f"Error downloading weights: {str(e)}")
        return False
    finally:
        # Clean up progress indicators after a short delay
        import time
        time.sleep(2)
        progress_text.empty()
        progress_bar.empty()

@st.cache_resource
def load_model(model_type):
    """Load ONNX model"""
    if model_type == "ArcFace":
        model_path = "weights/arcface.onnx"
    else:  # Partial FC
        model_path = "weights/partialfc.onnx"
    return onnxruntime.InferenceSession(model_path)

def extract_faces(image: np.ndarray, padding: int = 15, is_query: bool = False) -> list:
    """Extract faces from image with multi-angle detection"""
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
        return rotated

    face_images = []
    angles = [0] if is_query else [0, -30, -15, 15, 30]
    
    for angle in angles:
        current_image = image if angle == 0 else rotate_image(image, angle)
        
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

def get_face_embedding(face_image: np.ndarray, session: onnxruntime.InferenceSession) -> torch.Tensor:
    """Get face embedding using ONNX model"""
    image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(image).unsqueeze(0).numpy()
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: input_tensor})[0]
    return torch.tensor(embedding)

def calculate_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Calculate cosine similarity between embeddings"""
    emb1 = torch.nn.functional.normalize(emb1, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, dim=1)
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()

def process_image(image_file):
    """Convert uploaded file to OpenCV format"""
    if image_file is None:
        return None
    try:
        # Reset file pointer to beginning
        image_file.seek(0)
        
        # Read file content
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        
        # Decode image
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
            
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def compare_faces_local(image1_data, image2_data, model_type, threshold):
    """Compare faces using local processing"""
    try:
        # Load model
        session = load_model(model_type)
        
        # Extract faces
        faces1 = extract_faces(image1_data, is_query=True)
        faces2 = extract_faces(image2_data, is_query=True)
        
        if not faces1 or not faces2:
            raise ValueError("No faces detected in one or both images")
        
        # Get embeddings
        embedding1 = get_face_embedding(faces1[0], session)
        embedding2 = get_face_embedding(faces2[0], session)
        
        # Calculate similarity
        similarity = calculate_similarity(embedding1, embedding2)
        
        # Convert faces to base64 for display
        _, buffer1 = cv2.imencode('.jpg', faces1[0])
        _, buffer2 = cv2.imencode('.jpg', faces2[0])
        face1_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer1).decode('utf-8')}"
        face2_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer2).decode('utf-8')}"
        
        return {
            'similarity': float(similarity),
            'face1': face1_b64,
            'face2': face2_b64,
            'is_match': similarity >= threshold
        }
    except Exception as e:
        st.error(f"Error comparing faces: {str(e)}")
        return None

# Download weights at startup
if not download_weights():
    st.error("Failed to download model weights.")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .stTitle {
        color: #1a237e;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        padding-bottom: 1rem;
        border-bottom: 2px solid #1a237e;
        margin-bottom: 2rem;
        text-align: center !important;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #4a148c;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem;
        text-align: center !important;
    }

    /* Description text styling */
    .description-text {
        text-align: center !important;
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        width: 100%;
        background-color: #1a237e;
        color: white;
    }
    .stButton>button:hover {
        background-color: #4a148c;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<h1 class="stTitle">Face Recognition Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
    <div class="description-text">
        Compare facial features using state-of-the-art deep learning models. Upload two images and adjust the similarity thresholds to analyze face matching.
    </div>
""", unsafe_allow_html=True)

# Create two columns for image upload
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
    st.markdown('<h2 class="stSubheader">Reference Image</h2>', unsafe_allow_html=True)
    image1 = st.file_uploader("Upload reference image", type=['jpg', 'jpeg', 'png'], key="ref_img")
    if image1:
        # Convert to PIL Image and resize
        pil_img = Image.open(image1)
        # Calculate new size maintaining aspect ratio
        basewidth = 300
        wpercent = (basewidth/float(pil_img.size[0]))
        hsize = int((float(pil_img.size[1])*float(wpercent)))
        pil_img = pil_img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        st.image(pil_img, caption="Reference Image", use_container_width=False)

with col2:
    st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
    st.markdown('<h2 class="stSubheader">Comparison Image</h2>', unsafe_allow_html=True)
    image2 = st.file_uploader("Upload comparison image", type=['jpg', 'jpeg', 'png'], key="comp_img")
    if image2:
        # Convert to PIL Image and resize
        pil_img = Image.open(image2)
        # Calculate new size maintaining aspect ratio
        basewidth = 300
        wpercent = (basewidth/float(pil_img.size[0]))
        hsize = int((float(pil_img.size[1])*float(wpercent)))
        pil_img = pil_img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        st.image(pil_img, caption="Comparison Image", use_container_width=False)

# Thresholds
col3, col4 = st.columns(2)
with col3:
    arcface_threshold = st.slider("ArcFace Threshold", 0.0, 1.0, 0.5, 0.01)
with col4:
    partialfc_threshold = st.slider("Partial FC Threshold", 0.0, 1.0, 0.5, 0.01)

# Model selection
model_type = st.radio(
    "Select Model",
    ["ArcFace", "Partial FC", "Compare Both"],
    horizontal=True
)

def display_results(results, model_name):
    if results and 'error' not in results:
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            # Convert base64 to PIL Image and resize
            img_data = base64.b64decode(results['face1'].split(',')[1])
            pil_img = Image.open(io.BytesIO(img_data))
            basewidth = 200
            wpercent = (basewidth/float(pil_img.size[0]))
            hsize = int((float(pil_img.size[1])*float(wpercent)))
            pil_img = pil_img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
            st.image(pil_img, caption=f"Processed Face 1", use_container_width=False)
        
        with col2:
            # Display similarity score
            similarity = results['similarity'] * 100
            st.metric(
                label=f"{model_name} Similarity",
                value=f"{similarity:.2f}%"
            )
            
            # Display match status with styling
            if results['is_match']:
                st.markdown("""
                    <div style='text-align: center; padding: 10px; background-color: #4CAF50; color: white; 
                    border-radius: 5px; margin: 10px 0; font-weight: bold;'>
                        ✅ MATCH
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='text-align: center; padding: 10px; background-color: #f44336; color: white; 
                    border-radius: 5px; margin: 10px 0; font-weight: bold;'>
                        ❌ NO MATCH
                    </div>
                """, unsafe_allow_html=True)
        
        with col3:
            # Convert base64 to PIL Image and resize
            img_data = base64.b64decode(results['face2'].split(',')[1])
            pil_img = Image.open(io.BytesIO(img_data))
            basewidth = 200
            wpercent = (basewidth/float(pil_img.size[0]))
            hsize = int((float(pil_img.size[1])*float(wpercent)))
            pil_img = pil_img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
            st.image(pil_img, caption=f"Processed Face 2", use_container_width=False)
    else:
        st.error(results.get('error', 'Unknown error occurred'))

if st.button("Analyze Faces"):
    if image1 is None or image2 is None:
        st.warning("Please upload both images first.")
    else:
        with st.spinner("Processing images..."):
            try:
                # Convert uploaded files to OpenCV format
                image1_cv = process_image(image1)
                image2_cv = process_image(image2)
                
                if image1_cv is None or image2_cv is None:
                    st.error("Failed to process one or both images. Please try uploading them again.")
                    st.stop()
                
                if model_type == "Compare Both":
                    tab1, tab2 = st.tabs(["ArcFace Results", "Partial FC Results"])
                    
                    with tab1:
                        arcface_results = compare_faces_local(image1_cv, image2_cv, "ArcFace", arcface_threshold)
                        display_results(arcface_results, "ArcFace")
                    
                    with tab2:
                        partialfc_results = compare_faces_local(image1_cv, image2_cv, "Partial FC", partialfc_threshold)
                        display_results(partialfc_results, "Partial FC")
                else:
                    threshold = arcface_threshold if model_type == "ArcFace" else partialfc_threshold
                    results = compare_faces_local(image1_cv, image2_cv, model_type, threshold)
                    display_results(results, model_type)
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

# Add information about the project
with st.expander("About this project"):
    st.markdown("""
        This face recognition system uses state-of-the-art deep learning models for accurate face comparison:
        
        - **ArcFace**: Advanced face recognition model using additive angular margin loss
        - **Partial FC**: Memory-efficient implementation for large-scale face recognition
        - **RetinaFace**: Robust face detection and alignment
        
        Upload two images and adjust the similarity thresholds to compare faces.
        Higher threshold values require more similarity for a match.
    """)

# Footer
st.markdown("---")
