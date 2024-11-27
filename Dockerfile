# Use NVIDIA CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables with defaults
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Model and server configuration
ENV MODEL_NAME="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
ENV SERVER_NAME="0.0.0.0"
ENV SERVER_PORT=7860

# Model parameters
ENV MODEL_TEMPERATURE=0.3
ENV MODEL_TOP_K=50
ENV MODEL_TOP_P=0.9
ENV MAX_OUTPUT_TOKENS=2048

# PyTorch configuration
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Image configuration
ENV MAX_IMAGE_WIDTH=1120
ENV MAX_IMAGE_HEIGHT=1120

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose port
EXPOSE ${SERVER_PORT}

# Run the application
CMD ["python3", "clean-ui.py"]
