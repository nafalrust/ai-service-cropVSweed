# Gunakan image dasar
FROM python:3.10-slim

# Install dependencies untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam container
COPY . .

# Buka port untuk Flask
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "app.py"]
