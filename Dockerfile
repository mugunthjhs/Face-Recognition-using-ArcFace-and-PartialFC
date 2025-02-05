# Use Python 3.8 slim image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    gcc \
    g++ \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages in steps to handle dependencies better
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create weights directory
RUN mkdir -p weights

# Download model weights using gdown
RUN pip install --no-cache-dir gdown==5.2.0 && \
    gdown --fuzzy "https://drive.google.com/file/d/1Yi9msAD_9pseDcf8Vucm87_8y49gJ-uU/view?usp=drive_link" -O weights/arcface.onnx && \
    gdown --fuzzy "https://drive.google.com/file/d/1_c3dp3CB2N3euqavKBXExrzM13yZnwlt/view?usp=drive_link" -O weights/partialfc.onnx && \
    gdown --fuzzy "https://drive.google.com/file/d/19x6pqwKVnNDG111IIUS0WNreM3uto7MB/view?usp=drive_link" -O weights/RRDB_ESRGAN_x4.pth

# Copy application code
COPY . .

# Create dist directory and copy interface
RUN mkdir -p dist && \
    cp interface.html dist/index.html

# Expose port
EXPOSE 2000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2

# Run the application
CMD ["python", "main.py"] 
