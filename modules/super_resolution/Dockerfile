# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (include wget for downloading model weights)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure weights directory exists and download the GFPGAN model weights
RUN mkdir -p /app/weights \
    && wget -q https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P /app/weights

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Define the command to run the application
CMD ["python", "app.py"]
