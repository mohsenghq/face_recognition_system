# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential git\
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Define the command to run the application
CMD ["python", "app.py"]
