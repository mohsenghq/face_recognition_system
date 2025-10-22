# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the insightface model
ARG EMBEDDING_MODEL_NAME=buffalo_l
COPY download_model.py .
RUN python download_model.py --model_name ${EMBEDDING_MODEL_NAME}

# Copy the rest of the application's code into the container
COPY . .

# Define the command to run the application
CMD ["python", "app.py"]
