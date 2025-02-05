import os
import gdown
import shutil
import subprocess
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['weights', 'dist', 'static', 'templates']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def download_model_weights():
    """Download model weights from Google Drive."""
    print("Downloading model weights...")
    
    model_files = {
        'arcface.onnx': 'https://drive.google.com/file/d/1Yi9msAD_9pseDcf8Vucm87_8y49gJ-uU/view?usp=drive_link',
        'partialfc.onnx': 'https://drive.google.com/file/d/1_c3dp3CB2N3euqavKBXExrzM13yZnwlt/view?usp=drive_link',
        'RRDB_ESRGAN_x4.pth': 'https://drive.google.com/file/d/19x6pqwKVnNDG111IIUS0WNreM3uto7MB/view?usp=drive_link'
    }

    for filename, url in model_files.items():
        output_path = os.path.join('weights', filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            try:
                gdown.download(url, output_path, quiet=False)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                return False
        else:
            print(f"{filename} already exists, skipping download")
    
    return True

def setup_virtual_environment():
    """Create and activate virtual environment."""
    print("Setting up virtual environment...")
    try:
        if not os.path.exists('venv'):
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        
        # Activate virtual environment and install requirements
        if os.name == 'nt':  # Windows
            activate_script = os.path.join('venv', 'Scripts', 'activate')
            pip_path = os.path.join('venv', 'Scripts', 'pip')
        else:  # Linux/Mac
            activate_script = os.path.join('venv', 'bin', 'activate')
            pip_path = os.path.join('venv', 'bin', 'pip')
        
        # Install requirements
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("Virtual environment setup complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up virtual environment: {str(e)}")
        return False

def prepare_for_render():
    """Prepare files for Render deployment."""
    print("Preparing for Render deployment...")
    try:
        # Create dist directory for static files
        os.makedirs('dist', exist_ok=True)
        
        # Copy interface.html to dist
        shutil.copy2('interface.html', os.path.join('dist', 'index.html'))
        
        # Ensure render.yaml exists
        if not os.path.exists('render.yaml'):
            with open('render.yaml', 'w') as f:
                f.write('''services:
  - type: web
    name: face-recognition-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.8.0
    healthCheckPath: /
    autoDeploy: true

  - type: static
    name: face-recognition-frontend
    buildCommand: mkdir -p dist && cp interface.html dist/index.html
    publishDir: dist
    staticPublishPath: /
    autoDeploy: true''')
        
        print("Render deployment preparation complete")
        return True
    except Exception as e:
        print(f"Error preparing for Render deployment: {str(e)}")
        return False

def verify_setup():
    """Verify all required files and directories exist."""
    required_files = [
        'weights/arcface.onnx',
        'weights/partialfc.onnx',
        'weights/RRDB_ESRGAN_x4.pth',
        'requirements.txt',
        'main.py',
        'interface.html',
        'render.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"- {file_path}")
        return False
    
    print("All required files present")
    return True

def main():
    """Main deployment setup function."""
    print("Starting deployment setup...")
    
    # Create necessary directories
    create_directories()
    
    # Download model weights
    if not download_model_weights():
        print("Failed to download model weights")
        return
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("Failed to setup virtual environment")
        return
    
    # Prepare for Render deployment
    if not prepare_for_render():
        print("Failed to prepare for Render deployment")
        return
    
    # Verify setup
    if not verify_setup():
        print("Setup verification failed")
        return
    
    print("""
Deployment setup complete!

Next steps:
1. Push your code to GitHub
2. Go to render.com and create a new Web Service
3. Connect your GitHub repository
4. Use the following settings:
   - Build Command: pip install -r requirements.txt
   - Start Command: python main.py
   - Publish Directory: dist
5. Add environment variables if needed
6. Deploy!

Note: Make sure your repository includes all required files and the model weights
in the weights directory.
""")

if __name__ == "__main__":
    main() 