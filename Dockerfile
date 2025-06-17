# Use official Python image
FROM python:3.6.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libopencv-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy code
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip

# Install project dependencies
RUN pip install -r requirements.txt

# Install Mask R-CNN (Matterport)
#RUN git clone https://github.com/matterport/Mask_RCNN.git && \
#    cd Mask_RCNN && \
#    pip install -r requirements.txt && \
#    python setup.py install && \
#    cd .. && rm -rf Mask_RCNN

# Expose Flask default port
EXPOSE 5000

# Run Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]

