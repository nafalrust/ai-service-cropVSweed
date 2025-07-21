# Crop vs Weed API

API ini digunakan untuk mendeteksi gambar tanaman (crop) vs gulma (weed) menggunakan model AI.

## Setup & Menjalankan API

### 1. Build Docker Image

```sh
docker build -t crop-vs-weed-api .
```

### 2. Jalankan Docker Container

```sh
docker run -p 5000:5000 crop-vs-weed-api
```

API akan berjalan di `http://localhost:5000`

---

## Deploy ke Render.com

Sudah tersedia file `render.yaml` untuk deployment otomatis ke Render.com.  
Pastikan repository sudah terhubung ke Render dan deploy sesuai instruksi Render.

---

## Penggunaan API

### Endpoint: `/predict_url`

Menerima URL gambar (misal dari Firebase Storage) dan mengembalikan hasil prediksi.

- **Method:** `POST`
- **Content-Type:** `application/json`
- **Body:**
  ```json
  {
    "image_url": "https://firebasestorage.googleapis.com/..."
  }
  ```
- **Response:**
  ```json
  {
    "result": "hasil_prediksi"
  }
  ```

### Contoh Request dengan `curl`

```sh
curl -X POST http://localhost:5000/predict_url \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://firebasestorage.googleapis.com/your-image-url"}'
```

---

## Catatan

- Pastikan URL gambar dapat diakses secara publik.
- Untuk endpoint lain (misal upload file langsung), silakan tambahkan sesuai kebutuhan.