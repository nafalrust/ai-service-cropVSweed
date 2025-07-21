# Gunakan image dasar Python yang ringan
FROM python:3.10-slim

# Install dependencies OS-level yang dibutuhkan untuk OpenCV, PIL, dsb.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Salin dan install dependensi Python lebih awal untuk cache layer lebih efisien
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek ke dalam container
COPY . .

# Buka port 7860 (wajib untuk Hugging Face Spaces)
EXPOSE 7860

# Jalankan aplikasi Flask di port 7860
CMD ["python", "app.py"]
