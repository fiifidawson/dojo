[WEBSITE](https://machinelearningmastery.com/step-by-step-guide-to-deploying-machine-learning-models-with-fastapi-and-docker/)
Next steps
1. Fix bug
2. Adapt output for usecase
(uvicorn app.main:app --reload --port 8000)
3. Requirements file
`pip freeze > requirements.txt`
4. Containerize with Docker
Dockerfile
`
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
`
4.1 Build Docker image
docker build -t heart-disease-predictor 

4.2 Run the container
docker run -d -p 8000:8000 heart-disease-predictor