# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and models
COPY . .

# Create necessary directories
RUN mkdir -p weights test_images

# Expose the port the app runs on
EXPOSE 2000

# Command to run the application
CMD ["python", "pfc3_base.py"] 