FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY app.py .
COPY model_entire.pt .
COPY best_model6.pt .
COPY label_encoder.pkl .

# Expose Flask port
EXPOSE 5000

# Jalankan Flask
CMD ["python", "app.py"]
