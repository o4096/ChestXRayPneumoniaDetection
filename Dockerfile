FROM pytorch/pytorch:latest

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc &&\
    rm -rf /var/lib/apt/lists/* &&\
    pip install --upgrade pip &&\
    pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy application files
COPY app.py .
COPY utils.py .
COPY xray_classifier.json .
COPY xray_classifier.pth .
COPY index.html .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
