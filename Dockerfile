# Use a base image with Python 3.6
FROM python:3.6.13

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Go back to app
WORKDIR /app

# Copy your code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "Maskrcnn_inference.py"]
