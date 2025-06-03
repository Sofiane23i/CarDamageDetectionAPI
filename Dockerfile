# Use a base image with Python 3.6
FROM python:3.6.13

# Set environment variables to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Clone and install Mask R-CNN (you can use your own fork or directory)
RUN git clone https://github.com/matterport/Mask_RCNN.git
WORKDIR /app/Mask_RCNN
RUN python setup.py install

# Go back to app
WORKDIR /app

# Copy your code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "your_script_name.py"]
