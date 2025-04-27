FROM python:3.8-slim

WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file to leverage Docker cache
COPY requirements.txt .

# Create a modified requirements file without numpy and scipy
RUN grep -v "numpy\|scipy\|paddleocr" requirements.txt > modified_requirements.txt

# Install numpy and scipy using binary packages (avoiding compilation)
RUN pip install --no-cache-dir \
    numpy==1.20.3 \
    scipy==1.5.2 \
    paddlepaddle==2.4.2 \
    paddleocr==2.6.1.3

# Install other Python dependencies
RUN pip install --no-cache-dir -r modified_requirements.txt

# Download NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords')"

# Create necessary directories
RUN mkdir -p predicted_results images processed_test

# Copy application code
COPY . .

# Set port environment variable
ENV PORT=8080

# Expose the port
EXPOSE ${PORT}

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]