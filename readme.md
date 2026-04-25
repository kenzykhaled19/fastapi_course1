# 🔬 Gram Stain Classifier — API Documentation

## Overview
AI-powered Gram stain classifier using EfficientNet-B0 (98% accuracy).  
Classifies microscopy images as **gram_positive** or **gram_negative**.

---

## 📁 Project Structure
```
gram_api/
├── main.py                  ← FastAPI app (this is the backend)
├── predict.py               ← AI model logic (from AI team)
├── gram_classifier.pth      ← Trained model weights (from AI team)
├── requirements.txt         ← Python dependencies
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place model file
Make sure `gram_classifier.pth` is in the same folder as `main.py`.

### 3. Start the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: `http://localhost:8000`

---

## 📡 API Endpoints

### `GET /`
Health check — confirms server is running.

**Response:**
```json
{
  "status": "running",
  "model": "EfficientNet-B0",
  "accuracy": "98%",
  "endpoint": "POST /predict-gram"
}
```

---

### `GET /health`
Check if model is loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict-gram` ⭐ Main Endpoint

Upload an image to get Gram stain classification.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` → image file

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.gif`

---

**✅ Success Response (200):**
```json
{
  "prediction":   "gram_negative",
  "confidence":   99.87,
  "all_probs": {
    "gram_negative": 99.87,
    "gram_positive": 0.13
  },
  "is_confident": true,
  "warning":      null
}
```

**⚠️ Low Confidence Response (200):**
```json
{
  "prediction":   "gram_positive",
  "confidence":   72.3,
  "all_probs": {
    "gram_negative": 27.7,
    "gram_positive": 72.3
  },
  "is_confident": false,
  "warning": "Low confidence (72.3%). Please perform confirmatory biochemical tests."
}
```

**❌ Error Response (400):**
```json
{
  "detail": "Unsupported format '.pdf'. Supported: ['.bmp', '.gif', '.jpg', ...]"
}
```

---

## 💻 Frontend Integration Examples

### JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append("file", imageFile); // imageFile = File object from <input>

const response = await fetch("http://localhost:8000/predict-gram", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result.prediction);   // "gram_negative" or "gram_positive"
console.log(result.confidence);   // e.g. 99.87
console.log(result.warning);      // null or warning string
```

### Axios
```javascript
const formData = new FormData();
formData.append("file", imageFile);

const { data } = await axios.post("http://localhost:8000/predict-gram", formData, {
  headers: { "Content-Type": "multipart/form-data" },
});
```

### cURL (for testing)
```bash
curl -X POST "http://localhost:8000/predict-gram" \
  -F "file=@/path/to/image.jpg"
```

---

## 📖 Interactive API Docs
Once server is running, open in browser:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc:       `http://localhost:8000/redoc`

---

## 🔧 Environment Variables
| Variable     | Default                  | Description              |
|-------------|--------------------------|--------------------------|
| `MODEL_PATH` | `gram_classifier.pth`    | Path to the model file   |

---

## 📊 Response Fields Reference
| Field         | Type    | Description                                      |
|--------------|---------|--------------------------------------------------|
| `prediction`  | string  | `"gram_positive"` or `"gram_negative"`           |
| `confidence`  | float   | Confidence percentage (0–100)                    |
| `all_probs`   | object  | Probability % for each class                     |
| `is_confident`| boolean | `true` if confidence ≥ 90%                       |
| `warning`     | string/null | Warning message if low confidence, else `null` |
